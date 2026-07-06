"""
cloth_center_markers.py — Tela con 4 marcadores de esquina para tracking del centro
====================================================================================
Añade 4 esferas de referencia en las esquinas de la tela para calcular
dinámicamente su centro geométrico y obtener transformaciones de posición
respecto a la base del robot y al TCP del gripper (wrist_3_link).

Coordenadas reportadas en cada ciclo:
  • Posición de las 4 esquinas (frame mundial)
  • Centro de la tela  (frame mundial)
  • Centro de la tela  (frame base del robot)
  • Desplazamiento centro → TCP  (frame TCP, con norma)
  • Target sugerido para el TCP  (agarre en el centro exacto)

Nota sobre las esferas: son entidades rígidas fijas que marcan las posiciones
iniciales de las esquinas. El tracking dinámico usa las partículas PBD reales,
que sí se mueven con la tela.

Uso:
    python cloth_center_markers.py
    python cloth_center_markers.py --no-viewer
    python cloth_center_markers.py --no-robot      # solo tela + marcadores
    python cloth_center_markers.py --steps 300     # simular antes del loop
"""

import argparse
import time
import re
import os
import numpy as np
import genesis as gs

# ── Constantes de escena ──────────────────────────────────────────────────────
DT          = 1 / 120
SUBSTEPS    = 4

ROBOT_POS   = np.array([-0.50, 0.0, 0.0], dtype=np.float64)
CLOTH_POS   = np.array([ 0.30, 0.0, 0.01], dtype=np.float64)
CLOTH_SCALE = 0.50

MARKER_RADIUS = 0.015   # 15 mm — bien visibles en el viewer

ARM_JOINTS  = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
HOME_ARM = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)

# Offset empírico TCP → centro del gripper cerrado (de gripper_inspector.py)
GRIP_OFFSET = np.array([-0.043, 0.002, -0.224], dtype=np.float64)


# ── Helpers XML ───────────────────────────────────────────────────────────────

def prepare_xml() -> str:
    """Parcha el XML del robot para usar integrador RK4."""
    d   = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    src = os.path.join(d, "ur10e_2f85.xml")
    out = os.path.join(d, "ur10e_2f85_rk4.xml")
    xml = open(src, encoding="utf-8").read()
    xml = re.sub(
        r'<option([^>]*)integrator="implicitfast"([^>]*)/>',
        r'<option\1integrator="RK4"\2/>',
        xml,
    )
    open(out, "w", encoding="utf-8").write(xml)
    return out


# ── Geometría de la tela ──────────────────────────────────────────────────────

def find_corner_particles(cloth, margin: float = 0.04):
    """
    Identifica las partículas PBD más cercanas a cada esquina del bbox XY.

    Devuelve:
        corner_particles : lista de 4 arrays de índices (NW, NE, SW, SE)
        corner_labels    : nombres de las esquinas
        corner_positions : posición media de cada grupo de partículas (4×3)
    """
    pos = cloth.get_particles_pos().numpy()
    x, y = pos[:, 0], pos[:, 1]

    # Orden: (xmin,ymin), (xmin,ymax), (xmax,ymin), (xmax,ymax)
    corners_xy = [
        (x.min(), y.min()),
        (x.min(), y.max()),
        (x.max(), y.min()),
        (x.max(), y.max()),
    ]
    labels = ["SW", "NW", "SE", "NE"]

    corner_particles = []
    corner_positions = []
    for (cx, cy), label in zip(corners_xy, labels):
        mask = (np.abs(x - cx) < margin) & (np.abs(y - cy) < margin)
        idxs = np.where(mask)[0].astype(np.int32)
        if len(idxs) == 0:
            # fallback: la partícula más cercana
            dist = np.hypot(x - cx, y - cy)
            idxs = np.array([np.argmin(dist)], dtype=np.int32)
        corner_particles.append(idxs)
        corner_positions.append(pos[idxs].mean(axis=0))

    return corner_particles, labels, np.array(corner_positions)


def cloth_center_from_corners(cloth, corner_particle_groups):
    """
    Calcula el centro de la tela como media de los centroides de las 4 esquinas.
    Más robusto que la media de todas las partículas cuando la tela se deforma.
    """
    pos = cloth.get_particles_pos().numpy()
    corners = np.array([pos[g].mean(axis=0) for g in corner_particle_groups])
    center  = corners.mean(axis=0)
    return center, corners


# ── Transformaciones de frame ─────────────────────────────────────────────────

def world_to_robot_base(point_world: np.ndarray, robot_base_pos=ROBOT_POS) -> np.ndarray:
    """
    Transforma un punto del frame mundial al frame de la base del robot.
    Asume que la base del robot NO está rotada respecto al mundo (euler=0).
    Transformación: p_base = p_world - T_robot_base
    """
    return np.asarray(point_world) - robot_base_pos


def point_relative_to_link(point_world: np.ndarray, link) -> np.ndarray:
    """
    Desplazamiento de un punto mundial en el frame de un link del robot.
    Nota: se devuelve en coordenadas mundiales (solo resta de posición).
    Para un frame rotado habría que aplicar R^T; aquí se reporta como vector diferencia.
    """
    link_pos = link.get_pos().numpy().flatten()
    return np.asarray(point_world) - link_pos


def grasp_target_for_tcp(cloth_center: np.ndarray, grip_offset=GRIP_OFFSET) -> np.ndarray:
    """
    Posición objetivo del TCP para que el centro del gripper cerrado
    quede justo encima del centro de la tela.

    TCP_target = cloth_center - grip_offset
      → grip_offset[2] ≈ -0.224 m  (dedos 22 cm bajo el TCP)
      → grip_offset[0] ≈ -0.043 m  (dedos 4.3 cm adelante del TCP en X)
    """
    target = cloth_center.copy()
    target[2] = cloth_center[2] - grip_offset[2]   # subimos el TCP para que los dedos bajen hasta la tela
    # XY: apuntar el TCP al centro XY exacto (offset X no se compensa en IK, según lecciones aprendidas)
    return target


# ── Construcción de la escena ─────────────────────────────────────────────────

def build_scene(show_viewer: bool, with_robot: bool):
    """
    Construye la escena con:
      - Suelo plano
      - Tela PBD algodón
      - 4 esferas de referencia en las esquinas estimadas de la tela
      - (Opcional) UR10e con gripper 2F-85 en HOME
    """
    mjcf = prepare_xml() if with_robot else None
    gs.init(backend=gs.cpu, logging_level="warning")

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
            camera_pos    = (1.5, -1.4, 1.2),
            camera_lookat = (0.3,  0.0, 0.2),
            camera_fov    = 45,
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    # ── Robot (opcional) ──────────────────────────────────────────────────────
    robot = None
    if with_robot:
        robot = scene.add_entity(gs.morphs.MJCF(
            file  = mjcf,
            pos   = tuple(ROBOT_POS),
            euler = (0.0, 0.0, 0.0),
        ))

    # ── Tela PBD ─────────────────────────────────────────────────────────────
    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file  = "meshes/cloth.obj",
            pos   = tuple(CLOTH_POS),
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

    # ── Esferas de referencia en las esquinas ─────────────────────────────────
    # Posiciones estimadas a partir de la geometría conocida de cloth.obj:
    #   cloth.obj centrado en (0,0,0), scale=0.50 → 25×25 cm
    #   bbox en escena: X:[0.176, 0.426], Y:[-0.124, 0.126], Z≈0.01
    # Se colocan ANTES del build (Genesis no permite añadir entidades después).
    # Son FIJAS: sirven como marcadores visuales de la posición INICIAL de las esquinas.
    # El tracking dinámico usa las partículas PBD reales (ver loop principal).
    corner_estimates = np.array([
        [0.176, -0.124, 0.015],   # SW
        [0.176,  0.126, 0.015],   # NW
        [0.426, -0.124, 0.015],   # SE
        [0.426,  0.126, 0.015],   # NE
    ], dtype=np.float32)

    marker_entities = []
    for pos in corner_estimates:
        m = scene.add_entity(
            gs.morphs.Sphere(
                pos    = tuple(pos),
                radius = MARKER_RADIUS,
                fixed  = True,
            )
        )
        marker_entities.append(m)

    scene.build()
    return scene, robot, cloth, marker_entities


# ── Main ──────────────────────────────────────────────────────────────────────

def run(show_viewer: bool, with_robot: bool, pre_steps: int):
    scene, robot, cloth, _marker_entities = build_scene(show_viewer, with_robot)

    tcp      = None
    dofs_arm = None
    if with_robot:
        dofs_arm = np.array(
            [j.dof_idx_local for j in robot.joints if j.name in ARM_JOINTS],
            dtype=np.int32,
        )
        # Teleportar a HOME y fijar el target PD desde el primer instante.
        # Sin control_dofs_position en cada paso, la gravedad tira el brazo.
        robot.set_dofs_position(HOME_ARM, dofs_arm)
        robot.control_dofs_position(HOME_ARM, dofs_arm)
        tcp = robot.get_link("wrist_3_link")

    # ── Identificar partículas de esquina ─────────────────────────────────────
    # Esperar 1 paso para que Genesis inicialice las posiciones
    scene.step()
    corner_particles, corner_labels, corner_pos_init = find_corner_particles(cloth, margin=0.04)

    print("\n" + "═" * 65)
    print("  MARCADORES DE ESQUINA — PARTÍCULAS IDENTIFICADAS")
    print("═" * 65)
    for label, particles, cpos in zip(corner_labels, corner_particles, corner_pos_init):
        print(f"  {label}:  {len(particles):3d} partículas  posición inicial = {np.round(cpos, 3)}")
    center_init, _ = cloth_center_from_corners(cloth, corner_particles)
    print(f"\n  Centro inicial (bbox corners): {np.round(center_init, 4)}")
    if with_robot:
        center_in_robot = world_to_robot_base(center_init)
        print(f"  Centro en frame robot base:    {np.round(center_in_robot, 4)}")
    print("═" * 65)

    # ── Simular N pasos opcionales antes del loop interactivo ─────────────────
    if pre_steps > 0:
        print(f"\n[Pre-simulando {pre_steps} pasos ({pre_steps * DT * SUBSTEPS:.2f} s)...]")
        for i in range(pre_steps):
            if dofs_arm is not None:
                robot.control_dofs_position(HOME_ARM, dofs_arm)
            scene.step()
            time.sleep(DT * SUBSTEPS)

    # ── Loop principal: tracking del centro ───────────────────────────────────
    print("\n[Loop] Tracking activo. Ctrl+C para salir.\n")
    REPORT_STEPS = int(1.0 / (DT * SUBSTEPS))  # imprimir cada ~1 s de simulación
    step_n = 0

    try:
        while True:
            if dofs_arm is not None:
                robot.control_dofs_position(HOME_ARM, dofs_arm)
            scene.step()
            time.sleep(DT * SUBSTEPS)
            step_n += 1

            if step_n % REPORT_STEPS != 0:
                continue

            t_sim = step_n * DT * SUBSTEPS

            # ── Calcular centro dinámico ──────────────────────────────────────
            center_world, corner_positions = cloth_center_from_corners(cloth, corner_particles)

            print(f"┌─ t = {t_sim:.1f} s " + "─" * 45)

            # Posiciones de las 4 esquinas
            for label, cpos in zip(corner_labels, corner_positions):
                print(f"│  {label}: {np.round(cpos, 4)}")

            print(f"│")
            print(f"│  Centro (mundo):              {np.round(center_world, 4)}")

            # Frame base del robot
            center_robot_base = world_to_robot_base(center_world)
            print(f"│  Centro (frame base robot):   {np.round(center_robot_base, 4)}")

            if with_robot and tcp is not None:
                # Frame TCP
                tcp_pos   = tcp.get_pos().numpy().flatten()
                delta_tcp = point_relative_to_link(center_world, tcp)
                dist_tcp  = float(np.linalg.norm(delta_tcp))
                print(f"│  TCP pos (mundo):             {np.round(tcp_pos, 4)}")
                print(f"│  Δ(centro → TCP) en mundo:    {np.round(delta_tcp, 4)}  ‖·‖={dist_tcp:.4f} m")

                # Target sugerido para el TCP (agarre en el centro)
                target = grasp_target_for_tcp(center_world)
                target_robot = world_to_robot_base(target)
                print(f"│  Target TCP para agarre:      {np.round(target, 4)}  (mundo)")
                print(f"│                               {np.round(target_robot, 4)}  (frame robot)")

            print(f"└" + "─" * 55)

    except KeyboardInterrupt:
        print("\nDetenido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tela con marcadores de esquina para tracking del centro")
    parser.add_argument("--no-viewer", action="store_true", help="Sin ventana gráfica")
    parser.add_argument("--no-robot",  action="store_true", help="Solo tela y marcadores, sin robot")
    parser.add_argument("--steps", type=int, default=120,
                        help="Pasos de pre-simulación antes del loop (default: 120 = 1 s)")
    args = parser.parse_args()

    run(
        show_viewer = not args.no_viewer,
        with_robot  = not args.no_robot,
        pre_steps   = args.steps,
    )
