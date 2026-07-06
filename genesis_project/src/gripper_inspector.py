"""
gripper_inspector.py — ¿A dónde va exactamente el gripper?
===========================================================
Reproduce el descenso del robot hasta el punto de agarre y
muestra en detalle la posición de cada link relevante vs la tela.

Uso:
    python gripper_inspector.py
"""

import re, os, time
import numpy as np
import genesis as gs

DT        = 1 / 120
SUBSTEPS  = 4
ROBOT_POS = (-0.50, 0.0, 0.0)
CLOTH_POS = (0.30,  0.0, 0.01)
CLOTH_SCALE = 0.50
HOME_ARM  = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
ARM_JOINTS  = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
               "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIP_JOINTS = ["left_driver_joint","right_driver_joint"]


def prepare_xml():
    d   = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    src = os.path.join(d, "ur10e_2f85.xml")
    out = os.path.join(d, "ur10e_2f85_rk4.xml")
    xml = open(src, encoding="utf-8").read()
    xml = re.sub(r'<option([^>]*)integrator="implicitfast"([^>]*)/>', r'<option\1integrator="RK4"\2/>', xml)
    open(out, "w", encoding="utf-8").write(xml)
    return out


def sep(title=""):
    w = 60
    pad = (w - len(title) - 2) // 2
    print("\n" + "─"*pad + f" {title} " + "─"*(w-pad-len(title)-2))


gs.init(backend=gs.cpu, logging_level="warning")

scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0,0,-9.81)),
    pbd_options=gs.options.PBDOptions(max_stretch_solver_iterations=8,
                                      max_bending_solver_iterations=4,
                                      particle_size=8e-3),
    viewer_options=gs.options.ViewerOptions(
        res=(1280,720),
        camera_pos=(1.4,-1.2,1.0),
        camera_lookat=(0.3,0.0,0.3),
        camera_fov=45,
    ),
)

scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file=prepare_xml(), pos=ROBOT_POS))
cloth = scene.add_entity(
    gs.morphs.Mesh(file="meshes/cloth.obj", pos=CLOTH_POS, scale=CLOTH_SCALE),
    material=gs.materials.PBD.Cloth(rho=5.0, stretch_compliance=5e-10,
                                    bending_compliance=6e-4, stretch_relaxation=0.45,
                                    bending_relaxation=0.15, static_friction=0.5,
                                    kinetic_friction=0.4, air_resistance=0.04),
)
scene.build()

dofs_arm  = np.array([j.dof_idx_local for j in robot.joints if j.name in ARM_JOINTS],  dtype=np.int32)
dofs_grip = np.array([j.dof_idx_local for j in robot.joints if j.name in GRIP_JOINTS], dtype=np.int32)

tcp      = robot.get_link("wrist_3_link")
finger_r = robot.get_link("right_pad")
finger_l = robot.get_link("left_pad")

sep("TODOS LOS LINKS DEL ROBOT")
for lk in robot.links:
    print(f"  idx={lk.idx:3d}  {lk.name}")

sep("LINKS USADOS EN EL AGARRE")
print(f"  TCP   → wrist_3_link  idx={tcp.idx}")
print(f"  pad R → right_pad     idx={finger_r.idx}")
print(f"  pad L → left_pad      idx={finger_l.idx}")
print(f"\n  fix_particles_to_link usa:  finger_r.idx = {finger_r.idx}  (solo el dedo DERECHO)")
print("  ⚠  Las partículas se pegan a UN solo dedo, no al centro del gripper")

# HOME
robot.set_dofs_position(HOME_ARM, dofs_arm)
for _ in range(5): scene.step()

# Cerrar gripper
robot.set_dofs_position(np.array([0.70, 0.70], dtype=np.float32), dofs_grip)
for _ in range(20): scene.step()

sep("POSICIONES CON GRIPPER CERRADO (HOME)")
t  = tcp.get_pos().numpy().flatten()
pr = finger_r.get_pos().numpy().flatten()
pl = finger_l.get_pos().numpy().flatten()
gc = (pr + pl) / 2
cloth_pos = cloth.get_particles_pos().numpy()
cloth_bbox_center = (cloth_pos.min(axis=0) + cloth_pos.max(axis=0)) / 2

print(f"  TCP (wrist_3_link): {np.round(t, 4)}")
print(f"  right_pad:          {np.round(pr, 4)}")
print(f"  left_pad:           {np.round(pl, 4)}")
print(f"  grip_center (medio):{np.round(gc, 4)}")
print(f"  cloth bbox center:  {np.round(cloth_bbox_center, 4)}")

sep("OFFSET 3D  (cada link respecto al TCP)")
print(f"  right_pad - TCP:    {np.round(pr - t, 4)}")
print(f"  left_pad  - TCP:    {np.round(pl - t, 4)}")
print(f"  grip_center - TCP:  {np.round(gc - t, 4)}")

sep("SEPARACIÓN ENTRE DEDOS")
diff = pr - pl
print(f"  right_pad - left_pad: {np.round(diff, 4)}")
print(f"  Distancia 3D:         {np.linalg.norm(diff):.4f} m")
print(f"  Separación en X:      {abs(diff[0]):.4f} m")
print(f"  Separación en Y:      {abs(diff[1]):.4f} m")
print(f"  Separación en Z:      {abs(diff[2]):.4f} m")

sep("IK: HACIA DÓNDE VA EL TCP vs DÓNDE QUEDA EL GRIPPER")
cloth_z    = float(cloth_pos[:,2].min())
offset_z   = float(gc[2] - t[2])
target_tcp = np.array([cloth_bbox_center[0], cloth_bbox_center[1],
                        cloth_z + 0.004 - offset_z])
print(f"  cloth_bbox_center XY: ({cloth_bbox_center[0]:.4f}, {cloth_bbox_center[1]:.4f})")
print(f"  offset_z (gc-TCP):    {offset_z:.4f} m")
print(f"  target TCP:           {np.round(target_tcp, 4)}")
print(f"\n  El IK lleva el TCP a target_tcp.")
print(f"  Los pads deberían quedar en:")
print(f"    right_pad ≈ TCP + ({pr[0]-t[0]:.3f}, {pr[1]-t[1]:.3f}, {pr[2]-t[2]:.3f})")
print(f"    left_pad  ≈ TCP + ({pl[0]-t[0]:.3f}, {pl[1]-t[1]:.3f}, {pl[2]-t[2]:.3f})")
print(f"    grip_center ≈ TCP + (0, 0, {offset_z:.3f})")

# Calcular dónde quedará cada pad en el agarre
est_pr_grasp = target_tcp + (pr - t)
est_pl_grasp = target_tcp + (pl - t)
est_gc_grasp = target_tcp + (gc - t)
print(f"\n  Posición estimada en el agarre:")
print(f"    right_pad:   {np.round(est_pr_grasp, 4)}")
print(f"    left_pad:    {np.round(est_pl_grasp, 4)}")
print(f"    grip_center: {np.round(est_gc_grasp, 4)}")
print(f"    cloth_bbox:  {np.round(cloth_bbox_center, 4)}")
print(f"\n  Error XY estimado right_pad vs centro tela: "
      f"dX={est_pr_grasp[0]-cloth_bbox_center[0]:.4f}  "
      f"dY={est_pr_grasp[1]-cloth_bbox_center[1]:.4f}")
print(f"  Error XY estimado grip_center vs centro tela: "
      f"dX={est_gc_grasp[0]-cloth_bbox_center[0]:.4f}  "
      f"dY={est_gc_grasp[1]-cloth_bbox_center[1]:.4f}")

sep("VIEWER — Ctrl+C para salir")
try:
    while True:
        scene.step()
        time.sleep(DT * SUBSTEPS)
except KeyboardInterrupt:
    print("Saliendo.")
