import numpy as np
import genesis as gs

# 1. INICIALIZAR
gs.init()
scene = gs.Scene(show_viewer=True)

# 2. AÑADIR ROBOT
scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.MJCF(file="xml/universal_robots_ur10e/ur10e_2f85_rk4.xml", pos=(0,0,0))
)

# 3. COMPILAR (una sola vez)
scene.build()

# 4. OBTENER ÍNDICES DE LOS JOINTS
dofs_arm = np.array([
    j.dof_idx_local for j in robot.joints
    if j.name in [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
], dtype=np.int32)

# 5. LOOP PRINCIPAL
# Posición home (brazo apuntando arriba, estable)
HOME = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])

# --- Prueba cada línea por separado ---

# Girar la BASE (joint 0) → el brazo rota sobre sí mismo
q = HOME.copy(); q[0] = 0.0          # neutro
q = HOME.copy(); q[0] = 1.57         # 90° a la derecha

# Subir/bajar el HOMBRO (joint 1)
q = HOME.copy(); q[1] = -0.5         # más arriba
q = HOME.copy(); q[1] = -2.5         # más abajo

# Doblar el CODO (joint 2)
q = HOME.copy(); q[2] = 0.5          # más cerrado
q = HOME.copy(); q[2] = 2.5          # más abierto

# Inclinar MUÑECA 1 (joint 3) → pitch de la mano
q = HOME.copy(); q[3] = 0.0
q = HOME.copy(); q[3] = -2.5

# Rotar MUÑECA 2 (joint 4) → roll de la mano  
q = HOME.copy(); q[4] = 0.0
q = HOME.copy(); q[4] = 1.57

# Girar MUÑECA 3 (joint 5) → spin final del tool
q = HOME.copy(); q[5] = 1.57
q = HOME.copy(); q[5] = -1.57

while True:
    robot.control_dofs_position(q, dofs_arm)  # ENVIAR COMANDO
    scene.step()                                         # AVANZAR SIMULACIÓN