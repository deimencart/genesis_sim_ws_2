import time
import math
import numpy as np
import genesis as gs

gs.init()

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/franka_emika_panda/panda.xml",
        pos=(0, 0, 0),
    )
)

scene.build()

# --- DOFs del brazo (7 joints principales) ---
arm_joints = robot.joints[:7]  # joint1..joint7
dofs = np.array([j.dof_idx_local for j in arm_joints], dtype=np.int32)

# Postura base (segura) para que se vea "humanoide" y levantado
# (No es la única, pero suele verse bien)
q_base = np.array([1.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.7], dtype=np.float32)

# Parámetros del saludo
bow_amp = 0.35     # cuánto “se inclina” (rad)
wave_amp = 0.6     # amplitud de “hola” (rad)
wave_speed = 3.0   # velocidad del “hola”
bow_speed = 1.2    # velocidad de la “reverencia”

t = 0.0
dt = 1.0 / 60.0

print("Saludo iniciado. Cierra la ventana o Ctrl+C para salir.")

while True:
    q = q_base.copy()

    # 1) “Reverencia”: baja un poco el brazo (más flexión/estiramiento)
    #    Ajustamos principalmente joint2 y joint4 para que se vea como inclinar
    bow = bow_amp * (0.5 + 0.5 * math.sin(bow_speed * t))  # de 0 a bow_amp
    q[1] -= bow            # joint2
    q[3] -= 0.6 * bow      # joint4 (acompaña)

    # 2) “Hola”: agitar la muñeca (joint7 suele funcionar bien)
    wave = wave_amp * math.sin(wave_speed * t)
    q[6] = q_base[6] + wave   # joint7 (muñeca)

    robot.control_dofs_position(q, dofs)

    scene.step()
    time.sleep(dt)
    t += dt
