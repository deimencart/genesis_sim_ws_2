"""
robot_box_to_cup.py — Robot coge telas de dentro de la caja y las pasa al vaso
===============================================================================
Escena:
  · UR10e + 2F-85 en HOME
  · Caja poco profunda (paredes 10 cm) con las 5 telas apiladas dentro
  · Vaso cilíndrico alto (Y+) como destino

Las paredes de la caja son bajas (10 cm) para que el robot pueda meter
el gripper dentro y coger la tela de encima de la pila.

Secuencia automática:
  Por cada tela (de la de encima a la de abajo):
    1. Localizar centro con esquinas (corner markers)
    2. Pre-grasp 20 cm sobre el centro (poco margen: las paredes son bajas)
    3. Cerrar gripper → descender dentro de la caja → adjuntar al TCP
    4. Levantar y sacar de la caja
    5. Mover al vaso y depositar dentro
  HOME final

Uso:
    python robot_box_to_cup.py
    python robot_box_to_cup.py --no-viewer
"""

import argparse
import time
import re
import os
import numpy as np
import genesis as gs

# ── Config ────────────────────────────────────────────────────────────────────
DT       = 1 / 120
SUBSTEPS = 4

ROBOT_POS = np.array([-0.50, 0.0, 0.0])

# Caja fuente (paredes bajas para que el gripper quepa)
BOX_CX, BOX_CY = 0.30, -0.35
BOX_IW, BOX_ID = 0.38, 0.38   # interior ancho — telas de 21 cm caben bien
BOX_IH, BOX_T  = 0.10, 0.025  # paredes 10 cm (bajas = accesibles)
BOX_TOP = BOX_T + BOX_IH       # 0.125 m

# Vaso destino (Y+)
CUP_CX, CUP_CY = 0.30, +0.55
CUP_R, CUP_H, CUP_T, CUP_N = 0.20, 0.32, 0.025, 8
CUP_TOP = CUP_T + CUP_H        # 0.345 m

# Telas apiladas DENTRO de la caja (Z = suelo de la caja)
CLOTH_SCALE = 0.40
STACK_Z0    = BOX_T + 0.002    # justo sobre la base de la caja

CLOTH_CONFIGS = [
    {"name": "azul",     "color": (0.10, 0.30, 0.85, 1.0)},
    {"name": "verde",    "color": (0.10, 0.72, 0.20, 1.0)},
    {"name": "amarilla", "color": (0.90, 0.85, 0.05, 1.0)},
    {"name": "naranja",  "color": (0.95, 0.52, 0.05, 1.0)},
    {"name": "roja",     "color": (0.85, 0.12, 0.12, 1.0)},
]
STACK_OFFSETS = [
    {"dx":  0.000, "dy":  0.000, "dz": 0.00, "rz":  0.0},
    {"dx":  0.010, "dy": -0.008, "dz": 0.03, "rz":  8.0},
    {"dx": -0.012, "dy":  0.009, "dz": 0.06, "rz": -12.0},
    {"dx":  0.008, "dy":  0.012, "dz": 0.09, "rz":  18.0},
    {"dx": -0.006, "dy": -0.010, "dz": 0.12, "rz": -5.0},
]

ARM_JOINTS  = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
               "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIP_JOINTS = ["left_driver_joint", "right_driver_joint"]
HOME_ARM    = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
FINGER_MARGIN = 0.004


# ── Utilidades (idénticas a robot_pile_to_containers.py) ─────────────────────

def prepare_xml():
    d   = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    src = os.path.join(d, "ur10e_2f85.xml")
    out = os.path.join(d, "ur10e_2f85_rk4.xml")
    xml = open(src, encoding="utf-8").read()
    xml = re.sub(r'<option([^>]*)integrator="implicitfast"([^>]*)/>', r'<option\1integrator="RK4"\2/>', xml)
    open(out, "w", encoding="utf-8").write(xml)
    return out

def smoothstep(s):
    s = float(np.clip(s, 0, 1))
    return s * s * (3 - 2 * s)

def move_joints(robot, scene, dofs_arm, q_from, q_to, seconds):
    steps = max(2, int(seconds / (DT * SUBSTEPS)))
    for k in range(steps):
        a = smoothstep(k / (steps - 1))
        robot.control_dofs_position((1-a)*q_from + a*q_to, dofs_arm)
        scene.step()
        time.sleep(DT * SUBSTEPS)
    return q_to.copy()

def wait(scene, robot, dofs_arm, q_hold, seconds):
    for _ in range(max(1, int(seconds / (DT * SUBSTEPS)))):
        robot.control_dofs_position(q_hold, dofs_arm)
        scene.step()
        time.sleep(DT * SUBSTEPS)

def do_ik(robot, tcp, pos, quat, dofs_arm):
    q, err = robot.inverse_kinematics(
        link=tcp, pos=np.array(pos, dtype=float),
        quat=np.array(quat, dtype=float) if quat is not None else None,
        dofs_idx_local=dofs_arm, max_samples=120, max_solver_iters=60, return_error=True,
    )
    err_m = float(np.linalg.norm(np.array(err)[:3]))
    print(f"    IK err={err_m:.4f} m  target={np.round(pos,3)}")
    return np.array(q, dtype=np.float32).flatten()[dofs_arm]

def find_corner_particles(cloth, margin=0.04):
    pos = cloth.get_particles_pos().numpy()
    x, y = pos[:,0], pos[:,1]
    groups = []
    for cx, cy in [(x.min(),y.min()),(x.min(),y.max()),(x.max(),y.min()),(x.max(),y.max())]:
        mask = (np.abs(x-cx)<margin) & (np.abs(y-cy)<margin)
        idxs = np.where(mask)[0].astype(np.int32)
        if len(idxs)==0:
            idxs = np.array([np.argmin(np.hypot(x-cx,y-cy))], dtype=np.int32)
        groups.append(idxs)
    return groups

def cloth_center(cloth, corner_groups):
    pos = cloth.get_particles_pos().numpy()
    corners = np.array([pos[g].mean(axis=0) for g in corner_groups])
    return corners.mean(axis=0), corners

def find_top_cloth(remaining):
    best, best_z = None, -np.inf
    for entry in remaining:
        z = entry["cloth"].get_particles_pos().numpy()[:,2].mean()
        if z > best_z:
            best_z = z
            best = entry
    return best

def cloth_material():
    return gs.materials.PBD.Cloth(
        rho=5.0, stretch_compliance=5e-10, bending_compliance=6e-4,
        stretch_relaxation=0.45, bending_relaxation=0.15,
        static_friction=0.50, kinetic_friction=0.40, air_resistance=0.04,
    )


# ── Constructores de recipientes ──────────────────────────────────────────────

def make_box_container(scene, cx, cy, iw, id_, ih, t, color):
    ow, od = iw+2*t, id_+2*t
    for p in [
        dict(pos=(cx, cy, t/2),                      size=(ow, od, t)),
        dict(pos=(cx, cy-(id_/2+t/2), t+ih/2),       size=(ow, t,  ih)),
        dict(pos=(cx, cy+(id_/2+t/2), t+ih/2),       size=(ow, t,  ih)),
        dict(pos=(cx-(iw/2+t/2), cy, t+ih/2),        size=(t,  id_, ih)),
        dict(pos=(cx+(iw/2+t/2), cy, t+ih/2),        size=(t,  id_, ih)),
    ]:
        scene.add_entity(gs.morphs.Box(pos=p["pos"], size=p["size"], fixed=True),
                         surface=gs.surfaces.Default(color=color))

def make_cup(scene, cx, cy, r, h, t, n, color):
    bw = r * np.cos(np.pi/n)
    scene.add_entity(gs.morphs.Box(pos=(cx,cy,t/2), size=(2*bw,2*bw,t), fixed=True),
                     surface=gs.surfaces.Default(color=color))
    pw = 2*(r+t/2)*np.tan(np.pi/n)+0.008
    for i in range(n):
        ang = i*2*np.pi/n
        scene.add_entity(
            gs.morphs.Box(pos=(cx+(r+t/2)*np.cos(ang), cy+(r+t/2)*np.sin(ang), t+h/2),
                          size=(t, pw, h), euler=(0., 0., float(np.degrees(ang))), fixed=True),
            surface=gs.surfaces.Default(color=color))


# ── Grasp desde dentro de la caja ─────────────────────────────────────────────

def grasp_from_box(entry, robot, tcp, finger_r, finger_l,
                   dofs_arm, dofs_grip, scene, q_arm, grasp_quat):
    """
    Grasp con pre-grasp bajo (paredes bajas = poco margen vertical).
    El gripper desciende dentro de la caja desde 20 cm sobre el centro.
    """
    cloth = entry["cloth"]
    corner_groups = find_corner_particles(cloth)
    center, _ = cloth_center(cloth, corner_groups)
    cloth_z   = float(cloth.get_particles_pos().numpy()[:,2].min())

    # Pre-grasp bajo: solo 20 cm sobre la tela (paredes de 10 cm no estorban mucho)
    PRE_H = 0.22
    pre = np.array([center[0], center[1], cloth_z + PRE_H])
    q_arm = move_joints(robot, scene, dofs_arm, q_arm,
                        do_ik(robot, tcp, pre, grasp_quat, dofs_arm), seconds=2.5)
    wait(scene, robot, dofs_arm, q_arm, 0.3)

    # Cerrar gripper
    robot.set_dofs_position(np.array([0.70, 0.70], dtype=np.float32), dofs_grip)
    wait(scene, robot, dofs_arm, q_arm, 0.4)

    # Offset TCP→pads con gripper cerrado
    gc  = (finger_r.get_pos().numpy().flatten() + finger_l.get_pos().numpy().flatten()) / 2
    off = gc - tcp.get_pos().numpy().flatten()

    # Re-medir centro
    center2, _ = cloth_center(cloth, corner_groups)
    cz2 = float(cloth.get_particles_pos().numpy()[:,2].min())

    target = np.array([center2[0], center2[1], cz2 + FINGER_MARGIN - off[2]])
    q_arm = move_joints(robot, scene, dofs_arm, q_arm,
                        do_ik(robot, tcp, target, grasp_quat, dofs_arm), seconds=2.0)
    wait(scene, robot, dofs_arm, q_arm, 0.3)

    # Corrección Z
    for _ in range(4):
        cz_now   = float(cloth.get_particles_pos().numpy()[:,2].min())
        tz_now   = float(tcp.get_pos().numpy()[2])
        target_z = cz_now + FINGER_MARGIN - off[2]
        err = tz_now - target_z
        if abs(err) < 0.003:
            break
        target[2] -= err
        q_arm = move_joints(robot, scene, dofs_arm, q_arm,
                            do_ik(robot, tcp, target, grasp_quat, dofs_arm), seconds=0.4)
        wait(scene, robot, dofs_arm, q_arm, 0.2)

    # Adjuntar al TCP
    pos_now  = cloth.get_particles_pos().numpy()
    dists    = np.linalg.norm(pos_now[:,:2] - center2[:2], axis=1)
    grip_idx = np.where(dists < 0.08)[0].astype(np.int32)
    if len(grip_idx) < 5:
        grip_idx = np.argsort(dists)[:20].astype(np.int32)
    cloth.fix_particles_to_link(tcp.idx, grip_idx)
    wait(scene, robot, dofs_arm, q_arm, 0.2)
    print(f"    adjuntadas {len(grip_idx)} partículas")

    return q_arm, grip_idx, off


def lift_out_of_box(robot, tcp, dofs_arm, scene, q_arm, grasp_quat, box_cx, box_cy):
    """Sube verticalmente para sacar la tela de la caja antes de desplazarse."""
    exit_pos = np.array([box_cx, box_cy, 0.55])
    q_arm = move_joints(robot, scene, dofs_arm, q_arm,
                        do_ik(robot, tcp, exit_pos, grasp_quat, dofs_arm), seconds=2.0)
    wait(scene, robot, dofs_arm, q_arm, 0.3)
    return q_arm


def deposit_in_cup(cloth, grip_idx, robot, tcp, dofs_arm, dofs_grip,
                   scene, q_arm, grasp_quat, cup_xy, cup_top, off):
    MARGIN = 0.03
    # Moverse sobre el vaso
    over = np.array([cup_xy[0], cup_xy[1], cup_top + 0.25 - off[2]])
    q_arm = move_joints(robot, scene, dofs_arm, q_arm,
                        do_ik(robot, tcp, over, grasp_quat, dofs_arm), seconds=3.0)
    wait(scene, robot, dofs_arm, q_arm, 0.4)

    # Bajar hasta el borde del vaso
    dep = np.array([cup_xy[0], cup_xy[1], cup_top + MARGIN - off[2]])
    q_arm = move_joints(robot, scene, dofs_arm, q_arm,
                        do_ik(robot, tcp, dep, grasp_quat, dofs_arm), seconds=1.5)
    wait(scene, robot, dofs_arm, q_arm, 0.3)

    # Soltar
    cloth.release_particle(grip_idx)
    robot.set_dofs_position(np.array([0.0, 0.0], dtype=np.float32), dofs_grip)
    wait(scene, robot, dofs_arm, q_arm, 1.5)
    return q_arm


# ── Build scene ───────────────────────────────────────────────────────────────

def build_scene(show_viewer):
    mjcf = prepare_xml()
    gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer    = show_viewer,
        sim_options    = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0,0,-9.81)),
        pbd_options    = gs.options.PBDOptions(
            max_stretch_solver_iterations=8, max_bending_solver_iterations=4, particle_size=8e-3),
        viewer_options = gs.options.ViewerOptions(
            res=(1280,720), camera_pos=(1.8,-1.0,1.4),
            camera_lookat=(0.3,0.1,0.3), camera_fov=50),
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=mjcf, pos=tuple(ROBOT_POS)))

    # Caja fuente (paredes bajas, Y-)
    make_box_container(scene, BOX_CX, BOX_CY, BOX_IW, BOX_ID, BOX_IH, BOX_T,
                       color=(0.60, 0.42, 0.22, 1.0))
    print(f"[Caja fuente]  ({BOX_CX:.2f}, {BOX_CY:.2f})  interior {BOX_IW}×{BOX_ID} m  paredes {BOX_IH*100:.0f} cm")

    # Vaso destino (Y+)
    make_cup(scene, CUP_CX, CUP_CY, CUP_R, CUP_H, CUP_T, CUP_N,
             color=(0.25, 0.55, 0.72, 1.0))
    print(f"[Vaso destino] ({CUP_CX:.2f}, {CUP_CY:.2f})  radio {CUP_R} m  alto {CUP_H} m")

    # Telas dentro de la caja
    cloths = []
    print("\n[Telas] apiladas dentro de la caja:")
    for cfg, off in zip(CLOTH_CONFIGS, STACK_OFFSETS):
        cloth = scene.add_entity(
            gs.morphs.Mesh(
                file  = "meshes/cloth.obj",
                pos   = (BOX_CX + off["dx"], BOX_CY + off["dy"], STACK_Z0 + off["dz"]),
                euler = (0., 0., off["rz"]),
                scale = CLOTH_SCALE,
            ),
            material = cloth_material(),
            surface  = gs.surfaces.Default(color=cfg["color"]),
        )
        cloths.append({"name": cfg["name"], "cloth": cloth})
        print(f"  · {cfg['name']:10s}  Z={STACK_Z0+off['dz']:.3f}  rot={off['rz']:.1f}°")

    scene.build()
    return scene, robot, cloths


# ── Main ──────────────────────────────────────────────────────────────────────

def run(show_viewer):
    scene, robot, cloths = build_scene(show_viewer)

    dofs_arm  = np.array([j.dof_idx_local for j in robot.joints if j.name in ARM_JOINTS],  dtype=np.int32)
    dofs_grip = np.array([j.dof_idx_local for j in robot.joints if j.name in GRIP_JOINTS], dtype=np.int32)
    tcp      = robot.get_link("wrist_3_link")
    finger_r = robot.get_link("right_pad")
    finger_l = robot.get_link("left_pad")

    robot.set_dofs_position(HOME_ARM, dofs_arm)
    robot.control_dofs_position(HOME_ARM, dofs_arm)
    q_arm = HOME_ARM.copy()
    wait(scene, robot, dofs_arm, q_arm, 0.5)
    grasp_quat = tcp.get_quat().numpy().flatten()

    print("\n[Asentando telas 2 s...]")
    wait(scene, robot, dofs_arm, q_arm, 2.0)

    remaining = list(cloths)

    for i in range(len(cloths)):
        entry = find_top_cloth(remaining)
        remaining.remove(entry)

        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(cloths)}] Cogiendo '{entry['name']}' de la caja → vaso")
        print(f"{'='*55}")

        q_arm, grip_idx, off = grasp_from_box(
            entry, robot, tcp, finger_r, finger_l,
            dofs_arm, dofs_grip, scene, q_arm, grasp_quat,
        )

        # Subir para salir de la caja
        q_arm = lift_out_of_box(robot, tcp, dofs_arm, scene, q_arm, grasp_quat, BOX_CX, BOX_CY)

        # Depositar en el vaso
        q_arm = deposit_in_cup(
            entry["cloth"], grip_idx, robot, tcp, dofs_arm, dofs_grip,
            scene, q_arm, grasp_quat,
            np.array([CUP_CX, CUP_CY]), CUP_TOP, off,
        )
        print(f"  → '{entry['name']}' en el vaso")

    print("\n[HOME final...]")
    q_arm = move_joints(robot, scene, dofs_arm, q_arm, HOME_ARM, seconds=3.0)
    wait(scene, robot, dofs_arm, q_arm, 1.0)

    print("\n[OK] Transferencia completa. Ctrl+C para salir.")
    try:
        while True:
            robot.control_dofs_position(q_arm, dofs_arm)
            scene.step()
            time.sleep(DT * SUBSTEPS)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true")
    run(not parser.parse_args().no_viewer)
