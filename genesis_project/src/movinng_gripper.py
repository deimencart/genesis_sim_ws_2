import numpy as np
import genesis as gs
import time

DT = 1/60

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

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

def mover_a(q_desde, q_hasta, segundos=2.0):
    pasos = int(segundos / DT)
    for k in range(pasos):
        alpha = smoothstep(k / (pasos - 1))
        q = (1 - alpha) * q_desde + alpha * q_hasta
        robot.control_dofs_position(q, dofs_arm)
        scene.step()
        time.sleep(DT)
    return q_hasta.copy()

# ── Define tus puntos y cuánto tarda cada uno ────────────────────────────────
trayectoria = [
    ([-1.57, -1.57,  1.57, -1.57, -1.57, 0.0],  2.0),  # HOME
    ([-0.8,  -1.57,  1.57, -1.57, -1.57, 0.0],  1.0),  # izquierda
    ([ 0.8,  -1.57,  1.57, -1.57, -1.57, 0.0],  1.0),  # derecha
    ([-1.57, -1.0,   1.0,  -1.57, -1.57, 0.0],  1.5),  # subir
    ([-1.57, -1.57,  1.57, -1.57, -1.57, 0.0],  2.0),  # HOME
]

N_VECES = 5   # ← cambia este número, o pon 0 para infinito

# ── Ejecutar ──────────────────────────────────────────────────────────────────
q_actual = np.zeros(6, dtype=np.float32)

repeticiones = 0
try:
    while N_VECES == 0 or repeticiones < N_VECES:
        print(f"Vuelta {repeticiones + 1}")
        for i, (punto, segundos) in enumerate(trayectoria):
            q_siguiente = np.array(punto, dtype=np.float32)
            print(f"  -> Punto {i + 1}/{len(trayectoria)}")
            q_actual = mover_a(q_actual, q_siguiente, segundos)
        repeticiones += 1

    print("Terminado.")
    while True:          # se queda abierto al terminar
        scene.step()
        time.sleep(DT)

except KeyboardInterrupt:
    print("Saliendo.")
    scene.close()