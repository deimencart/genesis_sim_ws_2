import time
import genesis as gs
import numpy as np

gs.init()

scene = gs.Scene(
    show_viewer=True,  # <- fuerza el viewer
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 720),
        camera_pos=(2.0, 2.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.6),
    ),
)

scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.MJCF(
        #file="xml/franka_emika_panda/panda.xml",
        file="xml/universal_robots_ur10e/ur10e.xml",
        pos=(0, 0, 0),
    )
)

scene.build()

print("\nJOINTS DEL ROBOT:")
for j in robot.joints:
    print(" -", j.name)

joint = robot.joints[0]
dof = joint.dof_idx_local
print("\nMoviendo joint:", joint.name)

# Mueve el joint (objetivo)
robot.control_dofs_position(
    np.array([0.5], dtype=np.float32),
    np.array([dof], dtype=np.int32),
)

# Loop "infinito" para que la ventana se quede abierta
# (cierra la ventana o Ctrl+C)
dt = 1.0 / 60.0
while True:
    scene.step()
    time.sleep(dt)
