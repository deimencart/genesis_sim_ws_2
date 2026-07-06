import numpy as np
import genesis as gs
import time

DT = 1/60

gs.init()
scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(dt=DT),
)

scene.add_entity(gs.morphs.Plane())

# ── Tela con Mesh + material PBD.Cloth ───────────────────────────────────────
tela = scene.add_entity(
    gs.morphs.Mesh(
        file     = "meshes/cloth.obj",   # Genesis incluye este mesh de ejemplo
        pos      = (0.0, 0.0, 1.5),
        scale    = 0.6,
    ),
    material = gs.materials.PBD.Cloth(
        rho                = 4.0,
        stretch_compliance = 1e-7,
        bending_compliance = 1e-3,
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