"""
robot_cloth_spread.py — UR10e con gripper 2F-85 agarra y extiende una tela
===========================================================================

Uso:
    python robot_cloth_spread.py
    python robot_cloth_spread.py --no-viewer
"""

import argparse
import time
import re
import os
import numpy as np
import cv2
import genesis as gs

# ── Config ────────────────────────────────────────────────────────────────────

DT          = 1 / 120
SUBSTEPS    = 4

ROBOT_POS   = (-0.50, 0.0, 0.0)
CLOTH_DROP  = (0.30, 0.0, 0.01)
CLOTH_SCALE = 0.50

ARM_JOINTS  = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIP_JOINTS = ["left_driver_joint", "right_driver_joint"]   # ambos dedos
HOME_ARM    = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)

# Cámara wrist: offset respecto al wrist_3_link
# Montada 8 cm hacia adelante, mirando hacia abajo con 45° de tilt
CAM_RES     = (640, 480)
CAM_FOV     = 60


# ── Geometría de la tela ─────────────────────────────────────────────────────

def find_corner_particles(cloth, margin: float = 0.04):
    """
    Devuelve 4 grupos de partículas (una por esquina del bbox XY).
    Orden: SW, NW, SE, NE  →  (xmin,ymin), (xmin,ymax), (xmax,ymin), (xmax,ymax)
    """
    pos = cloth.get_particles_pos().numpy()
    x, y = pos[:, 0], pos[:, 1]
    corners_xy = [
        (x.min(), y.min()),
        (x.min(), y.max()),
        (x.max(), y.min()),
        (x.max(), y.max()),
    ]
    groups = []
    for cx, cy in corners_xy:
        mask = (np.abs(x - cx) < margin) & (np.abs(y - cy) < margin)
        idxs = np.where(mask)[0].astype(np.int32)
        if len(idxs) == 0:
            idxs = np.array([np.argmin(np.hypot(x - cx, y - cy))], dtype=np.int32)
        groups.append(idxs)
    return groups


def cloth_center_from_corners(cloth, corner_groups):
    """
    Centro de la tela = media de los centroides de las 4 esquinas.
    Más robusto que mean(all_particles) cuando la tela se deforma.
    Devuelve (center_xyz, corner_positions_4x3).
    """
    pos     = cloth.get_particles_pos().numpy()
    corners = np.array([pos[g].mean(axis=0) for g in corner_groups])
    return corners.mean(axis=0), corners


# ── Fix XML gripper RK4 ───────────────────────────────────────────────────────

def prepare_xml() -> str:
    d   = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    src = os.path.join(d, "ur10e_2f85.xml")
    out = os.path.join(d, "ur10e_2f85_rk4.xml")
    xml = open(src, encoding="utf-8").read()
    xml = re.sub(r'<option([^>]*)integrator="implicitfast"([^>]*)/>', r'<option\1integrator="RK4"\2/>', xml)
    open(out, "w", encoding="utf-8").write(xml)
    return out


# ── Movimiento ────────────────────────────────────────────────────────────────

def smoothstep(s):
    s = float(np.clip(s, 0, 1))
    return s * s * (3 - 2 * s)

def move_joints(robot, scene, cam, q_from, q_to, dofs, seconds):
    steps = max(2, int(seconds / (DT * SUBSTEPS)))
    for k in range(steps):
        a = smoothstep(k / (steps - 1))
        robot.control_dofs_position((1 - a) * q_from + a * q_to, dofs)
        scene.step()
        show_cam(cam)
        time.sleep(DT * SUBSTEPS)
    return q_to.copy()

def wait(scene, cam, seconds):
    for _ in range(max(1, int(seconds / (DT * SUBSTEPS)))):
        scene.step()
        show_cam(cam)
        time.sleep(DT * SUBSTEPS)

def ik(robot, tcp, pos, quat=None, dofs=None):
    q, err = robot.inverse_kinematics(
        link             = tcp,
        pos              = np.array(pos, dtype=float),
        quat             = np.array(quat, dtype=float) if quat is not None else None,
        dofs_idx_local   = dofs,
        max_samples      = 100,
        max_solver_iters = 50,
        return_error     = True,
    )
    pos_err = float(np.linalg.norm(err[:3]))
    print(f"  IK pos_err={pos_err:.4f} m  target={np.round(pos, 3)}")
    if pos_err > 0.02:
        print("  [!] IK no convergió bien")
    q = np.array(q, dtype=np.float32).flatten()
    if dofs is not None:
        q = q[dofs]
    return q


# ── Cámara wrist ──────────────────────────────────────────────────────────────

def make_wrist_cam(scene, wrist_link):
    """Cámara RGB adjunta al wrist_3_link — debe llamarse ANTES de scene.build()."""
    cam = scene.add_camera(
        res    = CAM_RES,
        pos    = (0.0, 0.0, 0.3),
        lookat = (0.0, 0.0, 0.0),
        fov    = CAM_FOV,
        GUI    = False,
    )

    th = np.deg2rad(45)       # 45° de tilt — ve gripper + zona delante
    R_x = np.array([
        [1,           0,            0],
        [0,  np.cos(th), -np.sin(th)],
        [0,  np.sin(th),  np.cos(th)],
    ], dtype=np.float32)

    offset_T = np.eye(4, dtype=np.float32)
    offset_T[:3, :3] = R_x
    offset_T[:3,  3] = np.array([0.0, -0.15, 0.10])  # 15 cm atrás, 10 cm arriba

    cam.attach(wrist_link, offset_T)
    return cam

def show_cam(cam):
    """Renderiza y muestra la imagen de la cámara wrist en una ventana OpenCV."""
    if cam is None:
        return
    cam.move_to_attach()
    rgb, *_ = cam.render()
    if rgb is None:
        return
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Wrist Camera", bgr)
    cv2.waitKey(1)


# ── Build scene ───────────────────────────────────────────────────────────────

def build_scene(show_viewer):
    mjcf = prepare_xml()
    gs.init()

    scene = gs.Scene(
        show_viewer    = show_viewer,
        sim_options    = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0, 0, -9.81)),
        pbd_options    = gs.options.PBDOptions(
            max_stretch_solver_iterations = 8,
            max_bending_solver_iterations = 4,
            particle_size                 = 8e-3,
        ),
        viewer_options = gs.options.ViewerOptions(
            res           = (1280, 720),
            camera_pos    = (1.6, -1.4, 1.2),
            camera_lookat = (0.1,  0.0, 0.3),
            camera_fov    = 45,
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(gs.morphs.MJCF(
        file  = mjcf,
        pos   = ROBOT_POS,
        euler = (0.0, 0.0, 0.0),
    ))

    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file  = "meshes/cloth.obj",
            pos   = CLOTH_DROP,
            scale = CLOTH_SCALE,
        ),
        material = gs.materials.PBD.Cloth(
            rho                = 5.0,
            stretch_compliance = 5e-10,
            bending_compliance = 6e-4,
            stretch_relaxation = 0.45,
            bending_relaxation = 0.15,
            static_friction    = 0.50,
            kinetic_friction   = 0.40,
            air_resistance     = 0.04,
        ),
    )

    wrist_link = robot.get_link("wrist_3_link")
    cam = make_wrist_cam(scene, wrist_link)

    scene.build()
    return scene, robot, cloth, cam


# ── Main ──────────────────────────────────────────────────────────────────────

def run(show_viewer):
    scene, robot, cloth, cam = build_scene(show_viewer)

    # DOFs
    dofs_arm  = np.array([j.dof_idx_local for j in robot.joints if j.name in ARM_JOINTS],  dtype=np.int32)
    dofs_grip = np.array([j.dof_idx_local for j in robot.joints if j.name in GRIP_JOINTS], dtype=np.int32)

    tcp         = robot.get_link("wrist_3_link")
    finger_r    = robot.get_link("right_pad")
    finger_l    = robot.get_link("left_pad")

    q_arm  = np.zeros(len(dofs_arm),  dtype=np.float32)
    q_grip = np.zeros(len(dofs_grip), dtype=np.float32)

    # ── 0. HOME ──────────────────────────────────────────────────────────────
    print("\n[0] HOME...")
    robot.set_dofs_position(HOME_ARM, dofs_arm)
    wait(scene, cam, 0.5)
    q_arm      = HOME_ARM.copy()
    grasp_quat = tcp.get_quat().numpy().flatten()

    # ── 1. Dejar asentar la tela y medir su centro real ──────────────────────
    wait(scene, cam, 1.0)   # esperar que la tela caiga y se asiente
    # Identificar las 4 esquinas una sola vez (índices estables durante toda la sim)
    corner_groups = find_corner_particles(cloth, margin=0.04)
    cloth_center, corner_positions = cloth_center_from_corners(cloth, corner_groups)
    cloth_z = float(cloth.get_particles_pos().numpy()[:, 2].min())
    print(f"[1] Centro tela (corners): {np.round(cloth_center, 3)}  cloth_z={cloth_z:.3f}")
    print(f"    SW={np.round(corner_positions[0],3)}  NW={np.round(corner_positions[1],3)}")
    print(f"    SE={np.round(corner_positions[2],3)}  NE={np.round(corner_positions[3],3)}")

    # ── 2. Pre-grasp: TCP sobre el centro de la tela ─────────────────────────
    pre_grasp = np.array([cloth_center[0], cloth_center[1], cloth_z + 0.30])
    print(f"\n[2] Pre-grasp {np.round(pre_grasp, 3)}...")
    q_pre = ik(robot, tcp, pre_grasp, grasp_quat, dofs=dofs_arm)
    q_arm = move_joints(robot, scene, cam, q_arm, q_pre, dofs_arm, seconds=2.5)
    wait(scene, cam, 0.5)

    # ── 3. Bajar al agarre usando offset 3D TCP→pads ──────────────────────────
    # El offset entre TCP y el centro de los dedos tiene componentes X, Y y Z.
    # Hay que compensar los tres para que los dedos queden sobre la tela.
    FINGER_MARGIN = 0.004   # 4 mm ≈ radio de partícula PBD

    def grip_center():
        pr = finger_r.get_pos().numpy().flatten()
        pl = finger_l.get_pos().numpy().flatten()
        return (pr + pl) / 2.0

    # ── 3a. Cerrar el gripper ANTES de bajar ─────────────────────────────────
    # Al cerrar, los dedos bajan ~6.5 cm extra. Hay que medir el offset
    # con el gripper ya cerrado para calcular el target correcto.
    robot.set_dofs_position(np.array([0.70, 0.70], dtype=np.float32), dofs_grip)
    wait(scene, cam, 0.4)   # dejar que el gripper se asiente

    tcp_pos   = tcp.get_pos().numpy().flatten()
    gc_pos    = grip_center()
    offset_3d = gc_pos - tcp_pos   # offset con gripper CERRADO
    print(f"\n[3] Offset TCP→pads (cerrado): {np.round(offset_3d, 4)}")

    # Re-medir cloth_z y centro real tras asentamiento
    cloth_z_settled = float(cloth.get_particles_pos().numpy()[:, 2].min())
    # Usar las mismas esquinas identificadas en paso 1 — más estable que bbox o mean
    cloth_center_settled, _ = cloth_center_from_corners(cloth, corner_groups)
    cloth_center_xy         = cloth_center_settled[:2]

    # TCP apunta directamente al centro de la tela en XY.
    # Solo corregir Z con el offset (los dedos cuelgan 22 cm bajo el TCP).
    target_tcp = np.array([
        cloth_center_xy[0],                              # X = centro tela
        cloth_center_xy[1],                              # Y = centro tela
        cloth_z_settled + FINGER_MARGIN - offset_3d[2], # Z corregido
    ])
    print(f"  cloth_center={np.round(cloth_center_xy,4)}  target_tcp={np.round(target_tcp,4)}")

    q_g   = ik(robot, tcp, target_tcp, grasp_quat, dofs=dofs_arm)
    q_arm = move_joints(robot, scene, cam, q_arm, q_g, dofs_arm, seconds=2.5)
    wait(scene, cam, 0.4)

    # Corrección iterativa solo en Z (TCP XY ya apunta al centro exacto)
    for corr in range(4):
        cloth_z_now = float(cloth.get_particles_pos().numpy()[:, 2].min())
        tcp_z_now   = float(tcp.get_pos().numpy()[2])
        # El TCP debe estar a cloth_z + FINGER_MARGIN - offset_3d[2] en Z
        target_z    = cloth_z_now + FINGER_MARGIN - offset_3d[2]
        err_z       = tcp_z_now - target_z
        print(f"  corr {corr}: tcp_z={tcp_z_now:.4f}  target_z={target_z:.4f}  err_z={err_z:.4f}")
        if abs(err_z) < 0.003:
            break
        target_tcp[2] -= err_z
        q_g   = ik(robot, tcp, target_tcp, grasp_quat, dofs=dofs_arm)
        q_arm = move_joints(robot, scene, cam, q_arm, q_g, dofs_arm, seconds=0.4)
        wait(scene, cam, 0.2)

    tcp_final       = tcp.get_pos().numpy().flatten()
    grasp_pos_final = target_tcp.copy()
    print(f"  TCP final: {np.round(tcp_final, 4)}")

    # ── 4. Adjuntar al TCP (wrist_3_link) — no al dedo ───────────────────────
    # Adjuntar al TCP garantiza que el punto de agarre es el centro XY de la tela.
    # Los dedos (offset -4.3cm X) crean el efecto visual del agarre.
    cloth_pos_now = cloth.get_particles_pos().numpy()
    # Seleccionar partículas más cercanas al centro XY de la tela en 2D
    dists_center = np.linalg.norm(cloth_pos_now[:, :2] - cloth_center_xy, axis=1)
    grip_idx = np.where(dists_center < 0.08)[0].astype(np.int32)
    if len(grip_idx) < 5:
        grip_idx = np.argsort(dists_center)[:20].astype(np.int32)

    print(f"\n[4] Adjuntando {len(grip_idx)} partículas al TCP (wrist_3_link, idx={tcp.idx})")
    cloth.fix_particles_to_link(tcp.idx, grip_idx)
    wait(scene, cam, 0.3)

    # ── 5. Levantar ───────────────────────────────────────────────────────────
    lift_pos = np.array([cloth_center_xy[0], cloth_center_xy[1], 0.65])
    print(f"\n[5] Levantando {np.round(lift_pos, 3)}...")
    q_lift = ik(robot, tcp, lift_pos, grasp_quat, dofs=dofs_arm)
    q_arm  = move_joints(robot, scene, cam, q_arm, q_lift, dofs_arm, seconds=3.0)
    wait(scene, cam, 1.0)

    # ── 6. Extender ───────────────────────────────────────────────────────────
    extend_pos = np.array([0.55, 0.0, 0.40])
    print(f"\n[6] Extendiendo {np.round(extend_pos, 3)}...")
    q_ext = ik(robot, tcp, extend_pos, grasp_quat, dofs=dofs_arm)
    q_arm = move_joints(robot, scene, cam, q_arm, q_ext, dofs_arm, seconds=3.5)

    lower_pos    = extend_pos.copy()
    lower_pos[2] = cloth_z + 0.08
    q_low = ik(robot, tcp, lower_pos, grasp_quat, dofs=dofs_arm)
    q_arm = move_joints(robot, scene, cam, q_arm, q_low, dofs_arm, seconds=1.5)

    # ── 7. Soltar ─────────────────────────────────────────────────────────────
    print("\n[7] Soltando...")
    cloth.release_particle(grip_idx)
    robot.set_dofs_position(np.array([0.0, 0.0], dtype=np.float32), dofs_grip)
    wait(scene, cam, 1.0)

    # ── 8. HOME ───────────────────────────────────────────────────────────────
    print("\n[8] HOME final...")
    q_ret = ik(robot, tcp, np.array([0.0, -0.35, 0.80]), grasp_quat, dofs=dofs_arm)
    q_arm = move_joints(robot, scene, cam, q_arm, q_ret,    dofs_arm, seconds=2.0)
    q_arm = move_joints(robot, scene, cam, q_arm, HOME_ARM, dofs_arm, seconds=2.0)

    print("\n[OK] Listo. Ctrl+C para salir.")
    try:
        while True:
            scene.step()
            show_cam(cam)
            time.sleep(DT * SUBSTEPS)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    run(not parser.parse_args().no_viewer)
