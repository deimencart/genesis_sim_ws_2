"""
simple_pick_place.py — Rutina simple: agarra tela, ponla en la caja
====================================================================
Sin detección de esquinas, sin corrección iterativa.
Solo: HOME → bajar → cerrar → fix → subir → ir a caja (con waypoint) → soltar → HOME
"""

import argparse, time, re, os
import numpy as np
import genesis as gs

gs.init(backend=gs.gpu)

# ── Config ────────────────────────────────────────────────────────────────────
DT, SUBSTEPS = 1/120, 4
ROBOT_POS = (-0.50, 0.0, 0.0)

# Posiciones fijas
PILE_POS   = np.array([0.35, 0.0, 0.0])     # centro de la pila
BOX_CENTER = np.array([0.10, -0.50, 0.0])   # centro de la caja

# Caja
BOX_CX, BOX_CY = 0.10, -0.50
BOX_IW, BOX_ID, BOX_IH, BOX_T = 0.28, 0.28, 0.20, 0.025
BOX_TOP = BOX_T + BOX_IH  # ~0.225m

# Telas
CLOTH_CONFIGS = [
    {"name": "azul",     "color": (0.10, 0.30, 0.85, 1.0), "dz": 0.00},
    {"name": "verde",    "color": (0.10, 0.72, 0.20, 1.0), "dz": 0.03},
    {"name": "amarilla", "color": (0.90, 0.85, 0.05, 1.0), "dz": 0.06},
    {"name": "naranja",  "color": (0.95, 0.52, 0.05, 1.0), "dz": 0.09},
    {"name": "roja",     "color": (0.85, 0.12, 0.12, 1.0), "dz": 0.12},
]

ARM_JOINTS  = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
GRIP_JOINTS = ["left_driver_joint", "right_driver_joint"]
HOME_Q = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)


# ── Helpers simples ───────────────────────────────────────────────────────────

def prepare_xml():
    d = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    src, out = os.path.join(d, "ur10e_2f85.xml"), os.path.join(d, "ur10e_2f85_rk4.xml")
    xml = open(src, encoding="utf-8").read()
    xml = re.sub(r'integrator="implicitfast"', 'integrator="RK4"', xml)
    open(out, "w", encoding="utf-8").write(xml)
    return out

def smooth(s):
    s = float(np.clip(s, 0, 1))
    return s * s * (3 - 2 * s)

def ik(robot, tcp, pos, quat, dofs):
    q, err = robot.inverse_kinematics(
        link=tcp, pos=np.array(pos, dtype=float),
        quat=np.array(quat, dtype=float),
        dofs_idx_local=dofs, max_samples=200, max_solver_iters=80, return_error=True,
    )
    
    # Imprimir advertencia si el error es significativo (falla de cinemática inversa)
    error_val = float(err.cpu().numpy().mean())
    if error_val > 0.01:
        print(f"    ⚠️ [IK Warning] Alto error ({error_val:.4f}) al intentar alcanzar: {pos}")
        
    return q.cpu().numpy().astype(np.float32).flatten()[dofs]

def goto(robot, scene, dofs, q_from, q_to, steps=80):
    """Mover joints en N steps. Sin time.sleep."""
    for k in range(steps):
        a = smooth(k / (steps - 1))
        robot.control_dofs_position((1 - a) * q_from + a * q_to, dofs)
        scene.step()
    return q_to.copy()

def hold(robot, scene, dofs, q, steps=30):
    """Mantener posición N steps."""
    for _ in range(steps):
        robot.control_dofs_position(q, dofs)
        scene.step()

def open_gripper(robot, dofs_grip):
    robot.set_dofs_position(np.array([0.0, 0.0], dtype=np.float32), dofs_grip)

def close_gripper(robot, dofs_grip):
    robot.set_dofs_position(np.array([0.72, 0.72], dtype=np.float32), dofs_grip)


# ── Contenedor ────────────────────────────────────────────────────────────────

def make_box(scene, cx, cy, iw, id_, ih, t, color):
    ow, od = iw + 2*t, id_ + 2*t
    for p in [
        dict(pos=(cx, cy, t/2),                         size=(ow, od, t)),
        dict(pos=(cx, cy-(id_/2+t/2), t+ih/2),          size=(ow, t, ih)),
        dict(pos=(cx, cy+(id_/2+t/2), t+ih/2),          size=(ow, t, ih)),
        dict(pos=(cx-(iw/2+t/2), cy, t+ih/2),           size=(t, id_, ih)),
        dict(pos=(cx+(iw/2+t/2), cy, t+ih/2),           size=(t, id_, ih)),
    ]:
        scene.add_entity(gs.morphs.Box(pos=p["pos"], size=p["size"], fixed=True),
                         surface=gs.surfaces.Default(color=color))


# ── Build ─────────────────────────────────────────────────────────────────────

def build(show_viewer):
    scene = gs.Scene(
        show_viewer = show_viewer,
        sim_options = gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0,0,-9.81)),
        pbd_options = gs.options.PBDOptions(
            max_stretch_solver_iterations=8, max_bending_solver_iterations=4, particle_size=8e-3),
        viewer_options = gs.options.ViewerOptions(
            res=(1280,720), camera_pos=(1.8,-1.2,1.4),
            camera_lookat=(0.2,0.0,0.25), camera_fov=50),
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=prepare_xml(), pos=ROBOT_POS))

    make_box(scene, BOX_CX, BOX_CY, BOX_IW, BOX_ID, BOX_IH, BOX_T,
             color=(0.60, 0.42, 0.22, 1.0))

    cloths = []
    for cfg in CLOTH_CONFIGS:
        c = scene.add_entity(
            gs.morphs.Mesh(file="meshes/cloth.obj",
                           pos=(0.35, 0.0, 0.01 + cfg["dz"]),
                           scale=0.40),
            material=gs.materials.PBD.Cloth(
                rho=5.0, stretch_compliance=5e-10, bending_compliance=6e-4,
                stretch_relaxation=0.45, bending_relaxation=0.15,
                static_friction=0.70, kinetic_friction=0.55, air_resistance=0.04),
            surface=gs.surfaces.Default(color=cfg["color"]),
        )
        cloths.append((cfg["name"], c))

    scene.build()
    return scene, robot, cloths


# ── Pick and place routine ────────────────────────────────────────────────────

def pick_and_place(robot, scene, cloth, tcp, finger_r, finger_l,
                   dofs_arm, dofs_grip, q_arm, grasp_quat):
    """
    Rutina completa para UNA tela:
      1. HOME → sobre la tela
      2. Bajar al contacto
      3. Cerrar gripper + fix partículas
      4. Subir vertical
      5. Ir sobre la caja (con waypoint)
      6. Bajar al borde de la caja
      7. Soltar
      8. Subir
      9. HOME
    """

    # ── Leer posición actual de la tela ──
    cloth_pos = cloth.get_particles_pos().cpu().numpy()
    cx, cy = cloth_pos[:, 0].mean(), cloth_pos[:, 1].mean()
    cz = cloth_pos[:, 2].mean()
    print(f"    Tela en ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    # Offset TCP → pads (el gripper está ~21cm debajo del wrist)
    pad_c = (finger_r.get_pos().cpu().numpy().flatten() +
             finger_l.get_pos().cpu().numpy().flatten()) / 2
    off_z = pad_c[2] - tcp.get_pos().cpu().numpy().flatten()[2]

    # ── 1. Ir sobre la tela (25cm arriba) ──
    print("    → sobre la tela")
    open_gripper(robot, dofs_grip)
    q_over = ik(robot, tcp, [cx, cy, cz + 0.25 - off_z], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_over, steps=100)
    hold(robot, scene, dofs_arm, q_arm, steps=20)

    # ── 2. Bajar al contacto ──
    print("    → bajando al contacto")
    q_contact = ik(robot, tcp, [cx, cy, cz + 0.005 - off_z], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_contact, steps=80)
    hold(robot, scene, dofs_arm, q_arm, steps=20)

    # ── 3. Cerrar gripper ──
    print("    → cerrando gripper")
    close_gripper(robot, dofs_grip)
    hold(robot, scene, dofs_arm, q_arm, steps=40)

    # ── 4. Fix partículas al TCP ──
    cloth_pos = cloth.get_particles_pos().cpu().numpy()
    pad_now = (finger_r.get_pos().cpu().numpy().flatten() +
               finger_l.get_pos().cpu().numpy().flatten()) / 2
    dists = np.linalg.norm(cloth_pos - pad_now, axis=1)
    grip_idx = np.argsort(dists)[:20].astype(np.int32)
    cloth.fix_particles_to_link(tcp.idx, grip_idx)
    hold(robot, scene, dofs_arm, q_arm, steps=15)
    print(f"    → fijadas {len(grip_idx)} partículas")

    # ── 5. Subir vertical (sobre la pila) ──
    print("    → subiendo")
    q_up = ik(robot, tcp, [cx, cy, 0.60], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_up, steps=100)
    hold(robot, scene, dofs_arm, q_arm, steps=15)

    # ── 6. Ir sobre la caja (con Waypoint) ──
    print("    → moviendo a la caja (vía punto intermedio)")
    # Waypoint a mitad de camino en X e Y, ligeramente más alto
    mid_x = (cx + BOX_CX) / 2
    mid_y = (cy + BOX_CY) / 2
    q_mid = ik(robot, tcp, [mid_x, mid_y, 0.65], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_mid, steps=80)
    hold(robot, scene, dofs_arm, q_arm, steps=10)

    # Destino final sobre la caja
    q_over_box = ik(robot, tcp, [BOX_CX, BOX_CY, 0.60], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_over_box, steps=80)
    hold(robot, scene, dofs_arm, q_arm, steps=15)

    # ── 7. Bajar al borde de la caja ──
    print("    → bajando sobre la caja")
    q_drop = ik(robot, tcp, [BOX_CX, BOX_CY, BOX_TOP + 0.08], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_drop, steps=80)
    hold(robot, scene, dofs_arm, q_arm, steps=15)

    # ── 8. Soltar ──
    print("    → soltando")
    for method in ['unfix_particles', 'release_particles', 'release_particle']:
        if hasattr(cloth, method):
            try:
                getattr(cloth, method)(grip_idx)
                print(f"      {method} OK")
                break
            except Exception as e:
                print(f"      {method} falló: {e}")

    open_gripper(robot, dofs_grip)
    hold(robot, scene, dofs_arm, q_arm, steps=60)  # esperar que caiga

    # ── 9. Subir y HOME ──
    print("    → retirando")
    q_retreat = ik(robot, tcp, [BOX_CX, BOX_CY, 0.60], grasp_quat, dofs_arm)
    q_arm = goto(robot, scene, dofs_arm, q_arm, q_retreat, steps=80)

    print("    → HOME")
    q_arm = goto(robot, scene, dofs_arm, q_arm, HOME_Q, steps=100)
    hold(robot, scene, dofs_arm, q_arm, steps=30)

    return q_arm


# ── Main ──────────────────────────────────────────────────────────────────────

def run(show_viewer):
    scene, robot, cloths = build(show_viewer)

    dofs_arm  = np.array([j.dof_idx_local for j in robot.joints if j.name in ARM_JOINTS])
    dofs_grip = np.array([j.dof_idx_local for j in robot.joints if j.name in GRIP_JOINTS])
    tcp       = robot.get_link("wrist_3_link")
    finger_r  = robot.get_link("right_pad")
    finger_l  = robot.get_link("left_pad")

    # Init
    robot.set_dofs_position(HOME_Q, dofs_arm)
    robot.control_dofs_position(HOME_Q, dofs_arm)
    q_arm = HOME_Q.copy()
    hold(robot, scene, dofs_arm, q_arm, steps=30)
    grasp_quat = tcp.get_quat().cpu().numpy().flatten()

    # Asentar telas
    print("[Asentando telas...]")
    hold(robot, scene, dofs_arm, q_arm, steps=120)

    # ── Procesar tela por tela (de arriba a abajo) ──
    remaining = list(cloths)

    for i in range(len(cloths)):
        # Encontrar la tela más alta
        best, best_z = None, -np.inf
        for name, c in remaining:
            z = c.get_particles_pos().cpu().numpy()[:, 2].mean()
            if z > best_z:
                best_z = z
                best = (name, c)
        name, cloth = best
        remaining.remove(best)

        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(cloths)}] Tela '{name}'")
        print(f"{'='*50}")

        q_arm = pick_and_place(
            robot, scene, cloth, tcp, finger_r, finger_l,
            dofs_arm, dofs_grip, q_arm, grasp_quat,
        )

    print("\n[DONE] Todas las telas procesadas.")

    # Idle loop
    try:
        while True:
            robot.control_dofs_position(q_arm, dofs_arm)
            scene.step()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-viewer", action="store_true")
    run(not p.parse_args().no_viewer)