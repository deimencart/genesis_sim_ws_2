import time
import numpy as np
import genesis as gs
import curses

# ---- Ajustes ----
DT = 1.0 / 60.0         # 60 FPS
STEP = 0.04             # tamaño de paso por tecla (radianes)

# Pose base (se ve bien para saludar y es estable)
Q_BASE = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.2, 0.7], dtype=np.float32)

# Mapa tecla -> (indice_joint, delta)
KEYMAP = {
    ord('a'): (0, -STEP),
    ord('d'): (0, +STEP),

    ord('w'): (1, +STEP),
    ord('s'): (1, -STEP),

    ord('i'): (2, +STEP),
    ord('k'): (2, -STEP),

    ord('j'): (3, -STEP),
    ord('l'): (3, +STEP),

    ord('q'): (4, -STEP),
    ord('e'): (4, +STEP),

    ord('u'): (5, +STEP),
    ord('o'): (5, -STEP),

    ord('r'): (6, +STEP),
    ord('f'): (6, -STEP),
}

def clamp_to_limits(robot, q):
    """Intenta limitar q a los límites si están disponibles."""
    # En algunas versiones, joint.limit existe; en otras no.
    for i, j in enumerate(robot.joints[:7]):
        lim = getattr(j, "limit", None)
        if lim is None:
            continue
        try:
            lo, hi = lim
            q[i] = float(np.clip(q[i], lo, hi))
        except Exception:
            pass
    return q

def main(stdscr):
    # ---- Terminal setup ----
    curses.cbreak()
    stdscr.nodelay(True)     # no bloquear esperando tecla
    stdscr.keypad(True)

    # ---- Genesis setup ----
    gs.init()
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)))
    scene.build()

    dofs = np.array([j.dof_idx_local for j in robot.joints[:7]], dtype=np.int32)
    q = Q_BASE.copy()

    # Mensaje en terminal
    stdscr.clear()
    stdscr.addstr(0, 0, "Control Panda (teclas en la TERMINAL)\n")
    stdscr.addstr(1, 0, "A/D:j1  W/S:j2  I/K:j3  J/L:j4  Q/E:j5  U/O:j6  R/F:j7\n")
    stdscr.addstr(2, 0, "SPACE: reset   X: salir\n")
    stdscr.refresh()

    # Aplica pose inicial
    robot.control_dofs_position(q, dofs)

    while True:
        key = stdscr.getch()

        if key == ord('x'):
            break

        if key == ord(' '):  # reset
            q = Q_BASE.copy()

        if key in KEYMAP:
            idx, delta = KEYMAP[key]
            q[idx] += delta
            q = clamp_to_limits(robot, q)

        # Enviar comando al robot (posición objetivo)
        robot.control_dofs_position(q, dofs)

        # Sim step
        scene.step()
        time.sleep(DT)

        # Mostrar estado
        stdscr.addstr(4, 0, f"q = {np.array2string(q, precision=2, suppress_small=True)}   ")
        stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(main)
