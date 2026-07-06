import numpy as np
import genesis as gs
import time

DT = 1/60

# ── Tus waypoints (cada fila = una posición del brazo) ──────────────────────
waypoints = [
    [-1.57, -1.57,  1.57, -1.57, -1.57,  0.0],   # HOME
    [ 0.0,  -1.57,  1.57, -1.57, -1.57,  0.0],   # girar base
    [ 0.0,  -1.0,   1.0,  -1.57, -1.57,  0.0],   # subir hombro
    [ 0.0,  -1.57,  1.57, -1.0,  -1.57,  1.57],  # mover muñecas
    [-1.57, -1.57,  1.57, -1.57, -1.57,  0.0],   # volver a HOME
]

def smoothstep(t):
    """Interpolación suave: arranca y frena despacio"""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def mover_a(robot, scene, q_desde, q_hasta, dofs, segundos=2.0):
    pasos = int(segundos / DT)
    for k in range(pasos):
        alpha = smoothstep(k / (pasos - 1))
        q = (1 - alpha) * q_desde + alpha * q_hasta
        robot.control_dofs_position(q, dofs)
        scene.step()
        time.sleep(DT)
    return q_hasta.copy()

# ── Setup ────────────────────────────────────────────────────────────────────
gs.init()
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file="xml/universal_robots_ur10e/ur10e_2f85_rk4.xml", pos=(0,0,0)))
scene.build()

dofs_arm = np.array([
    j.dof_idx_local for j in robot.joints
    if j.name in [
        "shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
        "wrist_1_joint","wrist_2_joint","wrist_3_joint"
    ]
], dtype=np.int32)

# ── Ejecutar trayectoria ─────────────────────────────────────────────────────
waypoints = [np.array(w, dtype=np.float32) for w in waypoints]
q_actual  = waypoints[0]

print("Iniciando trayectoria...")
for i, q_siguiente in enumerate(waypoints):
    print(f"  Waypoint {i+1}/{len(waypoints)}")
    q_actual = mover_a(robot, scene, q_actual, q_siguiente, dofs_arm, segundos=2.0)

print("Trayectoria completada.")
