"""
realistic_cloth.py — Simulación de tela realista con Genesis (PBD)
===================================================================

Escenario:
  - Tela de algodón (~50x50 cm) cae sobre una esfera rígida
  - Esquinas superiores fijadas 2 segundos, luego soltadas
  - Parámetros ajustados para comportamiento tipo algodón

Uso:
    python realistic_cloth.py           # cae sobre esfera
    python realistic_cloth.py --hang    # tela colgada (2 esquinas fijas permanentes)
    python realistic_cloth.py --free    # caída libre sin esfera
"""

import argparse
import time
import numpy as np
import genesis as gs


# ── Constantes de simulación ──────────────────────────────────────────────────

DT        = 1 / 120        # paso temporal (120 Hz → más estable)
SUBSTEPS  = 4              # sub-pasos por frame de física
CLOTH_H   = 1.2            # altura inicial de la tela
SPHERE_R  = 0.22           # radio de la esfera
SPHERE_Z  = SPHERE_R + 0.01


# ── Parámetros de material ── algodón medio ───────────────────────────────────
# rho              : densidad superficial kg/m²  (algodón ~4-6)
# stretch_compliance : rigidez a la tracción  (muy bajo = casi inextensible)
# bending_compliance : resistencia al doblado (alto = más suave/caído)
# static_friction  : fricción estática partícula-partícula
# kinetic_friction : fricción cinética
# air_resistance   : amortiguación por aire (hace las ondas más naturales)

CLOTH_MATERIAL = gs.materials.PBD.Cloth(
    rho                 = 5.0,
    stretch_compliance  = 5e-10,   # casi inextensible
    bending_compliance  = 6e-4,    # drape suave tipo algodón
    stretch_relaxation  = 0.45,
    bending_relaxation  = 0.15,
    static_friction     = 0.45,
    kinetic_friction    = 0.35,
    air_resistance      = 0.04,
)


def find_corner_particles(cloth, margin: float = 0.04):
    """
    Devuelve los índices de las 4 partículas más cercanas a las esquinas
    del bounding-box de la tela (en XY).  Se usa margin para coger
    un pequeño grupo en vez de solo el vértice más extremo.
    """
    pos = cloth.get_particles_pos().numpy()   # (N, 3)
    x, y = pos[:, 0], pos[:, 1]

    corners = [
        (x.min(), y.min()),
        (x.min(), y.max()),
        (x.max(), y.min()),
        (x.max(), y.max()),
    ]

    idx = []
    for cx, cy in corners:
        mask = (np.abs(x - cx) < margin) & (np.abs(y - cy) < margin)
        idx.extend(np.where(mask)[0].tolist())

    return np.array(list(set(idx)), dtype=np.int32)


def find_top_edge_particles(cloth, top_frac: float = 0.05):
    """Partículas en el borde superior (Y máximo) — para modo --hang."""
    pos   = cloth.get_particles_pos().numpy()
    y     = pos[:, 1]
    y_top = y.max()
    mask  = y > (y_top - top_frac * (y.max() - y.min()))
    return np.where(mask)[0].astype(np.int32)


def build_scene(show_viewer: bool, with_sphere: bool):
    gs.init()

    scene = gs.Scene(
        show_viewer = show_viewer,
        sim_options = gs.options.SimOptions(
            dt       = DT,
            substeps = SUBSTEPS,
            gravity  = (0.0, 0.0, -9.81),
        ),
        pbd_options = gs.options.PBDOptions(
            max_stretch_solver_iterations = 8,
            max_bending_solver_iterations = 4,
            particle_size                 = 8e-3,
        ),
        viewer_options = gs.options.ViewerOptions(
            res          = (1280, 720),
            camera_pos   = (1.4, -1.4, 1.2),
            camera_lookat= (0.0,  0.0, 0.4),
            camera_fov   = 45,
        ),
    )

    # ── Suelo ─────────────────────────────────────────────────────────────────
    scene.add_entity(gs.morphs.Plane())

    # ── Esfera rígida (target de caída) ───────────────────────────────────────
    sphere = None
    if with_sphere:
        sphere = scene.add_entity(
            gs.morphs.Sphere(
                pos   = (0.0, 0.0, SPHERE_Z),
                radius= SPHERE_R,
                fixed = True,
            )
        )

    # ── Tela ──────────────────────────────────────────────────────────────────
    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file  = "meshes/cloth.obj",
            pos   = (0.0, 0.0, CLOTH_H),
            scale = 0.55,
        ),
        material = CLOTH_MATERIAL,
    )

    scene.build()

    return scene, cloth, sphere


def run_drape(show_viewer: bool):
    """Tela cae sobre una esfera; esquinas fijadas 2 s y luego soltadas."""
    scene, cloth, _ = build_scene(show_viewer, with_sphere=True)

    # Fijar esquinas durante la caída inicial
    corners = find_corner_particles(cloth)
    print(f"[drape] Fijando {len(corners)} partículas de esquina...")
    cloth.fix_particles(corners)

    print("[drape] Simulando caída... (esquinas fijas 2 s)")
    t_sim   = 0.0
    pinned  = True
    dt_real = DT * SUBSTEPS

    try:
        while True:
            scene.step()
            t_sim += dt_real

            # Soltar esquinas a los 2 s
            if pinned and t_sim >= 2.0:
                print("[drape] Soltando esquinas...")
                cloth.release_particle(corners)
                pinned = False

            time.sleep(dt_real)
    except KeyboardInterrupt:
        print("\nDetenido.")


def run_hang(show_viewer: bool):
    """Tela colgada por el borde superior (fijo permanente)."""
    scene, cloth, _ = build_scene(show_viewer, with_sphere=False)

    top = find_top_edge_particles(cloth)
    print(f"[hang] Fijando {len(top)} partículas del borde superior.")
    cloth.fix_particles(top)

    print("[hang] Tela colgante — Ctrl+C para salir")
    dt_real = DT * SUBSTEPS
    try:
        while True:
            scene.step()
            time.sleep(dt_real)
    except KeyboardInterrupt:
        print("\nDetenido.")


def run_free(show_viewer: bool):
    """Caída libre sin fijar nada."""
    scene, cloth, _ = build_scene(show_viewer, with_sphere=False)

    print("[free] Caída libre — Ctrl+C para salir")
    dt_real = DT * SUBSTEPS
    try:
        while True:
            scene.step()
            time.sleep(dt_real)
    except KeyboardInterrupt:
        print("\nDetenido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación realista de tela")
    parser.add_argument("--hang", action="store_true", help="Tela colgada por borde superior")
    parser.add_argument("--free", action="store_true", help="Caída libre sin esfera")
    parser.add_argument("--no-viewer", action="store_true", help="Sin ventana gráfica")
    args = parser.parse_args()

    viewer = not args.no_viewer

    if args.hang:
        run_hang(viewer)
    elif args.free:
        run_free(viewer)
    else:
        run_drape(viewer)
