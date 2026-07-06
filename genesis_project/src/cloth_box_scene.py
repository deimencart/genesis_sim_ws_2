"""
cloth_box_scene.py — Telas apiladas con contenedor hueco y vaso cilíndrico
==========================================================================
Escena:
  · 5 telas de colores apiladas en orden y orientación específicos
  · Contenedor rectangular hueco (4 paredes + base, abierto por arriba)
  · Vaso cilíndrico grande (8 paneles octagonales + base, abierto por arriba)
  Las telas caen y pueden entrar/salir por los bordes de ambos recipientes.

Uso:
    python cloth_box_scene.py
    python cloth_box_scene.py --no-viewer
    python cloth_box_scene.py --spread   # telas separadas espacialmente (más dramático)
"""

import argparse
import time
import numpy as np
import genesis as gs

# ── Simulación ────────────────────────────────────────────────────────────────
DT       = 1 / 120
SUBSTEPS = 4

# ── Colores de las telas (orden de la pila: abajo → arriba) ───────────────────
CLOTH_STACK = [
    {"name": "azul",     "color": (0.10, 0.30, 0.85, 1.0)},
    {"name": "verde",    "color": (0.10, 0.72, 0.20, 1.0)},
    {"name": "amarilla", "color": (0.90, 0.85, 0.05, 1.0)},
    {"name": "naranja",  "color": (0.95, 0.52, 0.05, 1.0)},
    {"name": "roja",     "color": (0.85, 0.12, 0.12, 1.0)},
]

CLOTH_SCALE  = 0.42     # cada tela ~21×21 cm
STACK_X      = 0.38     # centro de la pila en X
STACK_Y      = 0.0      # centro de la pila en Y
STACK_Z0     = 0.01    # altura de la tela inferior

# Cada tela en la pila tiene un desplazamiento XY y rotación propios
# para que parezcan mal apiladas (como las dejaría un humano)
STACK_OFFSETS = [
    {"dx":  0.000, "dy":  0.000, "dz": 0.00, "rot_z":  0.0},   # azul   (fondo)
    {"dx":  0.015, "dy": -0.010, "dz": 0.03, "rot_z":  8.0},   # verde
    {"dx": -0.020, "dy":  0.012, "dz": 0.06, "rot_z": -14.0},  # amarilla
    {"dx":  0.010, "dy":  0.018, "dz": 0.09, "rot_z":  20.0},  # naranja
    {"dx": -0.008, "dy": -0.015, "dz": 0.12, "rot_z": -6.0},   # roja   (cima)
]

# ── Contenedor rectangular hueco ──────────────────────────────────────────────
BOX_CX   = 0.30          # centro X
BOX_CY   = -0.42         # centro Y (a un lado)
BOX_IW   = 0.32          # ancho interior (X)
BOX_ID   = 0.32          # profundidad interior (Y)
BOX_IH   = 0.22          # altura interior
BOX_T    = 0.025         # grosor de paredes
BOX_COLOR = (0.60, 0.42, 0.22, 1.0)   # marrón cartón

# ── Vaso cilíndrico grande ────────────────────────────────────────────────────
CUP_CX     = 0.30        # centro X
CUP_CY     = +0.52       # centro Y (al otro lado)
CUP_R      = 0.17        # radio interior
CUP_H      = 0.30        # altura interior
CUP_T      = 0.025       # grosor de paredes
CUP_N      = 8           # número de paneles (octágono)
CUP_COLOR  = (0.25, 0.55, 0.72, 1.0)  # azul plástico


# ── Constructores de recipientes ──────────────────────────────────────────────

def make_box_container(scene, cx, cy, iw, id_, ih, t, color):
    """
    Caja rectangular hueca: base + 4 paredes, abierta por arriba.
    Los paneles se montan con un ligero solapamiento en las esquinas.
    """
    ow = iw + 2 * t  # ancho exterior
    od = id_ + 2 * t

    panels = [
        # Base
        dict(pos=(cx,               cy,                    t / 2),
             size=(ow,              od,                    t)),
        # Pared Y-  (frente)
        dict(pos=(cx,               cy - (id_/2 + t/2),   t + ih/2),
             size=(ow,              t,                     ih)),
        # Pared Y+  (fondo)
        dict(pos=(cx,               cy + (id_/2 + t/2),   t + ih/2),
             size=(ow,              t,                     ih)),
        # Pared X-  (izquierda)
        dict(pos=(cx - (iw/2+t/2),  cy,                   t + ih/2),
             size=(t,               id_,                   ih)),
        # Pared X+  (derecha)
        dict(pos=(cx + (iw/2+t/2),  cy,                   t + ih/2),
             size=(t,               id_,                   ih)),
    ]

    for p in panels:
        scene.add_entity(
            gs.morphs.Box(pos=p["pos"], size=p["size"], fixed=True),
            surface=gs.surfaces.Default(color=color),
        )

    return t + ih   # altura del borde superior del contenedor


def make_cup(scene, cx, cy, r_inner, height, t, n_panels, color):
    """
    Vaso cilíndrico: base circular (approx. cuadrada) + N paneles en polígono regular.
    Los paneles se rotan con euler Z para apuntar radialmente.
    """
    # Base: cuadrado que cabe dentro del polígono
    base_hw = r_inner * np.cos(np.pi / n_panels)   # apotema del polígono
    scene.add_entity(
        gs.morphs.Box(
            pos=(cx, cy, t / 2),
            size=(2 * base_hw, 2 * base_hw, t),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=color),
    )

    # Paneles laterales
    panel_w = 2 * (r_inner + t/2) * np.tan(np.pi / n_panels) + 0.008  # solape pequeño
    for i in range(n_panels):
        angle   = i * (2 * np.pi / n_panels)
        px      = cx + (r_inner + t / 2) * np.cos(angle)
        py      = cy + (r_inner + t / 2) * np.sin(angle)
        pz      = t + height / 2
        euler_z = float(np.degrees(angle))   # rotar para apuntar radialmente

        scene.add_entity(
            gs.morphs.Box(
                pos   = (px, py, pz),
                size  = (t, panel_w, height),
                euler = (0.0, 0.0, euler_z),
                fixed = True,
            ),
            surface=gs.surfaces.Default(color=color),
        )

    return t + height   # altura del borde superior del vaso


# ── Escena completa ───────────────────────────────────────────────────────────

def build_scene(show_viewer: bool, spread: bool):
    gs.init(backend=gs.cpu, logging_level="warning")

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
            res           = (1280, 720),
            camera_pos    = (1.5, -1.1, 1.3),
            camera_lookat = (0.35, 0.0, 0.25),
            camera_fov    = 55,
        ),
    )

    # Suelo
    scene.add_entity(gs.morphs.Plane())

    # Contenedor rectangular (a un lado, Y-)
    make_box_container(
        scene, BOX_CX, BOX_CY, BOX_IW, BOX_ID, BOX_IH, BOX_T, BOX_COLOR,
    )
    print(f"[Contenedor] caja rectangular en ({BOX_CX:.2f}, {BOX_CY:.2f})")

    # Vaso cilíndrico (al otro lado, Y+)
    make_cup(
        scene, CUP_CX, CUP_CY, CUP_R, CUP_H, CUP_T, CUP_N, CUP_COLOR,
    )
    print(f"[Contenedor] vaso cilíndrico en ({CUP_CX:.2f}, {CUP_CY:.2f})")

    # Telas apiladas
    cloths = []
    print("\n[Telas] pila (de abajo a arriba):")
    for cfg, off in zip(CLOTH_STACK, STACK_OFFSETS):
        if spread:
            # Modo --spread: telas distribuidas en el espacio, no apiladas
            i    = len(cloths)
            x    = STACK_X + (i - 2) * 0.08
            y    = off["dy"] * 3
            z    = STACK_Z0 + i * 0.15
            rot  = off["rot_z"] * 2
        else:
            x    = STACK_X + off["dx"]
            y    = STACK_Y + off["dy"]
            z    = STACK_Z0 + off["dz"]
            rot  = off["rot_z"]

        cloth = scene.add_entity(
            gs.morphs.Mesh(
                file  = "meshes/cloth.obj",
                pos   = (x, y, z),
                euler = (0.0, 0.0, rot),
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
            surface = gs.surfaces.Default(color=cfg["color"]),
        )
        cloths.append((cfg["name"], cloth))
        print(f"  {len(cloths)}. {cfg['name']:10s}  pos=({x:.3f}, {y:.3f}, {z:.3f})  rot_z={rot:.1f}°")

    scene.build()
    return scene, cloths


# ── Loop de simulación ────────────────────────────────────────────────────────

def run(show_viewer: bool, spread: bool):
    scene, cloths = build_scene(show_viewer, spread)
    print("\nSimulando... Ctrl+C para salir.")

    dt_real = DT * SUBSTEPS
    try:
        while True:
            scene.step()
            time.sleep(dt_real)
    except KeyboardInterrupt:
        print("\nDetenido.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telas apiladas con contenedor y vaso")
    parser.add_argument("--no-viewer", action="store_true", help="Sin ventana gráfica")
    parser.add_argument("--spread",    action="store_true", help="Telas separadas en el espacio")
    args = parser.parse_args()

    run(not args.no_viewer, args.spread)
