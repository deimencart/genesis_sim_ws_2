import numpy as np
import genesis as gs
import time

DT = 1/60

gs.init()
scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=DT),
    pbd_options=gs.options.PBDOptions(          # ← añadir esto
        max_stretch_solver_iterations = 10,     # más iteraciones = mejor colisión
        max_bending_solver_iterations = 5,
        particle_size                 = 0.02,   # partículas más pequeñas
    ),
)

scene.add_entity(gs.morphs.Plane())

# ── Cilindro fijo ─────────────────────────────────────────────────────────────
cilindro = scene.add_entity(
    gs.morphs.Cylinder(
        height = 0.5,
        radius = 0.15,
        pos    = (0.0, 0.0, 0.25),
        fixed  = True,
    )
)

# ── Tela encima del cilindro ──────────────────────────────────────────────────
tela = scene.add_entity(
    gs.morphs.Mesh(
        file  = "meshes/cloth.obj",
        pos   = (0.0, 0.0, 2.0),
        scale = 0.6,
    ),
    material = gs.materials.PBD.Cloth(
         rho                = 4.0,
         stretch_compliance = 1e-7,    # ← mucho más rígido
         bending_compliance = 1e-4,
         #stretch_relaxation = 0.9,     # ← más alto = más estable
         #bending_relaxation = 0.5,
         air_resistance     = 0.01,
    ),
)

scene.build()

try:
    while True:
        scene.step()
        time.sleep(DT)
except KeyboardInterrupt:
    scene.close()