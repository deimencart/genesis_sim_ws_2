import time
import re
import numpy as np
import genesis as gs

DT = 1.0 / 60.0

def parse_home_qpos(xml_path: str):
    try:
        txt = open(xml_path, "r", encoding="utf-8", errors="ignore").read()
        m = re.search(r'<key[^>]*name="home"[^>]*qpos="([^"]+)"', txt)
        if not m:
            return None
        vals = [float(x) for x in m.group(1).strip().split()]
        return np.array(vals, dtype=np.float32)
    except Exception:
        return None

def smoothstep(s):
    s = max(0.0, min(1.0, s))
    return s * s * (3.0 - 2.0 * s)

def move_to(robot, scene, q_from, q_to, dofs, seconds=2.0):
    steps = max(2, int(seconds / DT))
    for k in range(steps):
        a = smoothstep(k / (steps - 1))
        q = (1 - a) * q_from + a * q_to
        robot.control_dofs_position(q, dofs)
        scene.step()
        time.sleep(DT)
    return q_to.copy()

def wave(robot, scene, q, dofs, idx=5, amp=0.8, freq=2.0, seconds=3.0):
    steps = max(1, int(seconds / DT))
    t = 0.0
    for _ in range(steps):
        q2 = q.copy()
        q2[idx] = q[idx] + amp * np.sin(2 * np.pi * freq * t)
        robot.control_dofs_position(q2, dofs)
        scene.step()
        time.sleep(DT)
        t += DT

def main():
    gs.init()

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    # 🔥 ESTO ES LO CORRECTO EN TU VERSIÓN
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/universal_robots_ur10e/ur10e.xml",
            pos=(0, 0, 0),
        )
    )

    scene.build()

    print("\nJOINTS UR10e:")
    for i, j in enumerate(robot.joints):
        print(i, j.name)

    dofs = np.array([j.dof_idx_local for j in robot.joints], dtype=np.int32)
    n = len(dofs)

    # Home desde keyframe o fallback
    xml_full_path = (
        gs.__file__.replace("__init__.py", "") +
        "assets/xml/universal_robots_ur10e/ur10e.xml"
    )

    q_home = parse_home_qpos(xml_full_path)
    if q_home is None or len(q_home) != n:
        q_home = np.zeros(n, dtype=np.float32)
        if n >= 6:
            q_home[:6] = np.array(
                [-1.57, -1.57, 1.57, -1.57, -1.57, 0.0],
                dtype=np.float32,
            )

    q = np.zeros(n, dtype=np.float32)

    # HOME
    q = move_to(robot, scene, q, q_home, dofs, seconds=2.0)

    # SALUDO
    q2 = q_home.copy()
    if n >= 6:
        q2[1] += 0.6
        q2[2] -= 0.4
        q2[3] -= 0.3

    q = move_to(robot, scene, q, q2, dofs, seconds=2.0)

    # WAVE
    if n >= 6:
        wave(robot, scene, q, dofs, idx=5, seconds=4.0)

    # LOOP FINAL
    print("\nUR10e saludando infinitamente 😄 (Ctrl+C para salir)")
    while True:
        if n >= 6:
            wave(robot, scene, q_home, dofs, idx=5, seconds=2.0)
        else:
            scene.step()
            time.sleep(DT)

if __name__ == "__main__":
    main()
