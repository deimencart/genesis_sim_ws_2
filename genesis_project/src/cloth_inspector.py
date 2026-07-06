"""
cloth_inspector.py — Exploración de la tela PBD en Genesis
===========================================================
Muestra cómo está definida la tela: partículas, geometría,
centro, atributos del material y del morfismo.

Uso:
    python cloth_inspector.py
    python cloth_inspector.py --steps 120   # simular N pasos antes de medir
"""

import argparse
import numpy as np
import genesis as gs

DT       = 1 / 120
SUBSTEPS = 4

CLOTH_POS   = (0.30, 0.0, 0.01)
CLOTH_SCALE = 0.50

CLOTH_MATERIAL = gs.materials.PBD.Cloth(
    rho                = 5.0,
    stretch_compliance = 5e-10,
    bending_compliance = 6e-4,
    stretch_relaxation = 0.45,
    bending_relaxation = 0.15,
    static_friction    = 0.50,
    kinetic_friction   = 0.40,
    air_resistance     = 0.04,
)


def separator(title=""):
    w = 60
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        print("\n" + "─" * w)


def inspect_cloth(steps: int):
    gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS, gravity=(0, 0, -9.81)),
        pbd_options=gs.options.PBDOptions(
            max_stretch_solver_iterations=8,
            max_bending_solver_iterations=4,
            particle_size=8e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(1.2, -1.2, 1.0),
            camera_lookat=(0.3, 0.0, 0.3),
            camera_fov=45,
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file  = "meshes/cloth.obj",
            pos   = CLOTH_POS,
            scale = CLOTH_SCALE,
        ),
        material=CLOTH_MATERIAL,
    )

    scene.build()

    # ── 1. Atributos del objeto cloth ─────────────────────────────────────────
    separator("OBJETO CLOTH")
    print(f"  Tipo Python:          {type(cloth)}")
    print(f"  cloth.idx:            {cloth.idx}")
    print(f"  Métodos relevantes:")
    metodos = [m for m in dir(cloth) if not m.startswith("_")]
    for m in metodos:
        print(f"    · {m}")

    # ── 2. Morfismo (Mesh) ────────────────────────────────────────────────────
    separator("MORFISMO (Mesh)")
    print(f"  Archivo:              meshes/cloth.obj")
    print(f"  pos en escena:        {CLOTH_POS}")
    print(f"  scale:                {CLOTH_SCALE}")

    # ── 3. Material PBD.Cloth ─────────────────────────────────────────────────
    separator("MATERIAL PBD.Cloth")
    mat_attrs = [a for a in dir(CLOTH_MATERIAL) if not a.startswith("_")]
    for a in mat_attrs:
        try:
            val = getattr(CLOTH_MATERIAL, a)
            if not callable(val):
                print(f"  {a:30s} = {val}")
        except Exception:
            pass

    # ── 4. Partículas en t=0 (antes de simular) ───────────────────────────────
    separator("PARTÍCULAS — t=0 (sin simular)")
    pos0 = cloth.get_particles_pos().numpy()
    _print_particle_stats(pos0, "t=0")

    # ── 5. Simular N pasos ────────────────────────────────────────────────────
    if steps > 0:
        separator(f"SIMULANDO {steps} pasos ({steps * DT * SUBSTEPS:.2f} s)")
        for i in range(steps):
            scene.step()
            if i % 30 == 0:
                pos_i = cloth.get_particles_pos().numpy()
                print(f"  step {i:4d}  cloth_z_min={pos_i[:,2].min():.4f}  "
                      f"center_xy=({pos_i[:,0].mean():.4f}, {pos_i[:,1].mean():.4f})")

        separator("PARTÍCULAS — tras simulación")
        pos_final = cloth.get_particles_pos().numpy()
        _print_particle_stats(pos_final, "final")

    # ── 6. Diferencia entre centros ───────────────────────────────────────────
    if steps > 0:
        separator("COMPARACIÓN DE CENTROS")
        c_mean0  = pos0.mean(axis=0)
        c_bbox0  = (pos0.min(axis=0) + pos0.max(axis=0)) / 2
        c_mean_f = pos_final.mean(axis=0)
        c_bbox_f = (pos_final.min(axis=0) + pos_final.max(axis=0)) / 2

        print(f"  {'Método':<20} {'t=0':>20} {'final':>20}")
        print(f"  {'mean (XYZ)':<20} {str(np.round(c_mean0,4)):>20} {str(np.round(c_mean_f,4)):>20}")
        print(f"  {'bbox center (XYZ)':<20} {str(np.round(c_bbox0,4)):>20} {str(np.round(c_bbox_f,4)):>20}")
        shift = c_mean_f[:2] - c_mean0[:2]
        print(f"\n  Desplazamiento XY del mean durante la simulación: {np.round(shift, 4)} m")

    separator("VIEWER")
    print("  Ctrl+C para salir")
    import time
    try:
        while True:
            scene.step()
            time.sleep(DT * SUBSTEPS)
    except KeyboardInterrupt:
        print("Saliendo.")


def _print_particle_stats(pos: np.ndarray, label: str):
    n = len(pos)
    c_mean = pos.mean(axis=0)
    c_bbox = (pos.min(axis=0) + pos.max(axis=0)) / 2

    print(f"  Nº partículas:        {n}")
    print(f"  ─── Extensión ────────────────────────────────────")
    for ax, name in enumerate(["X", "Y", "Z"]):
        lo, hi = pos[:, ax].min(), pos[:, ax].max()
        print(f"  {name}: [{lo:.4f}, {hi:.4f}]  ancho={hi-lo:.4f} m  "
              f"mean={pos[:,ax].mean():.4f}")
    print(f"  ─── Centro ───────────────────────────────────────")
    print(f"  Centro por MEAN:      {np.round(c_mean, 4)}")
    print(f"  Centro por BBOX:      {np.round(c_bbox, 4)}")
    print(f"  Diferencia mean-bbox: {np.round(c_mean - c_bbox, 5)}")

    # Distribución Z (¿está toda la tela en el suelo?)
    z_unique = np.unique(np.round(pos[:, 2], 4))
    if len(z_unique) <= 5:
        print(f"  Valores Z únicos:     {z_unique}")
    else:
        print(f"  Z percentiles [0,25,50,75,100]: "
              f"{np.round(np.percentile(pos[:,2],[0,25,50,75,100]),4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120,
                        help="Pasos de simulación antes de medir (default: 120 = 1 s)")
    args = parser.parse_args()
    inspect_cloth(args.steps)
