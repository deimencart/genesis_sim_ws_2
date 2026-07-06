# test_cam.py
import genesis as gs
import numpy as np
import cv2


gs.init()

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.MJCF(file="xml/universal_robots_ur10e/ur10e_2f85.xml")
)

# ── CAMARA ANTES DEL BUILD ──
cam = scene.add_camera(
    res=(640, 480),
    pos=(1.5, 0.0, 1.5),
    lookat=(0.0, 0.0, 0.5),
    fov=60,
    GUI=False,
)

scene.build()  # ahora sí

HOME = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0], dtype=np.float32)

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

dofs = []
for j in robot.joints:
    if j.name in ARM_JOINTS:
        dofs.append(j.dof_idx_local)

dofs = np.array(dofs, dtype=np.int32)

# Mover a home inmediatamente
robot.set_dofs_position(HOME, dofs)

# Ver links
print("\nLinks del robot:")
for link in robot.links:
    print(f"  {link.name}")

# Ver métodos de la cámara
print("\nMétodos de la cámara:")
print([m for m in dir(cam) if not m.startswith("_")])

# Probar attach
ee = robot.get_link("wrist_3_link")
offset_T = np.eye(4, dtype=np.float32)
offset_T[:3, 3] = [0.0, 0.01, 0.0]  # pequeño offset hacia adelante
tilt = np.deg2rad(90)
offset_T[:3, :3] = np.array([
    [1,  0,           0          ],
    [0,  np.cos(tilt), -np.sin(tilt)],
    [0,  np.sin(tilt),  np.cos(tilt)],
], dtype=np.float32)
try:
    cam.attach(ee, offset_T)
    print("\ncam.attach() -> OK")
except Exception as e:
    print(f"\ncam.attach() -> ERROR: {e}")

# Correr pasos y probar render
for i in range(5):
    scene.step()
    try:
        cam.move_to_attach()
    except Exception as e:
        print(f"move_to_attach() -> ERROR: {e}")
        break

try:
    result = cam.render()
    print(f"\ncam.render() -> type={type(result)}")
    if hasattr(result, '__len__'):
        for i, r in enumerate(result):
            arr = np.asarray(r) if r is not None else None
            print(f"  [{i}] shape={arr.shape if arr is not None else None}, dtype={arr.dtype if arr is not None else None}")
except Exception as e:
    print(f"\ncam.render() -> ERROR: {e}")

# al final del test_cam.py, después de todo
print("\nLoop infinito, Ctrl+C para salir")
try:
    i = 0
    while True:
        scene.step()
        cam.move_to_attach()
        
        # Mostrar imagen cada 10 frames
        if i % 10 == 0:
            rgb, _, _, _ = cam.render()
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            # BGR para OpenCV
            cv2.imshow("Eye-in-hand camera", rgb[:, :, ::-1])
            cv2.waitKey(1)
        i += 1
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("Saliendo.")