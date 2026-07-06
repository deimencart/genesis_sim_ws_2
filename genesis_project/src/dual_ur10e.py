"""
Dual UR10e + Mesa  —  Genesis Environment
==========================================
- Dos brazos UR10e con gripper 2F-85 montados sobre una mesa
- Fix del gripper: integrador RK4 (copiado del issue #40 mujoco_menagerie)
- DOFs separados: brazo (6) y gripper (1 actuado)
- Camara eye-in-hand en wrist_3_link de cada robot

Uso:
    python dual_ur10e_env.py --vis           # con viewer
    python dual_ur10e_env.py --vis --demo    # demo de movimiento + gripper
"""

from __future__ import annotations

import argparse
import shutil
import time
import re
import os
import numpy as np
import genesis as gs

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DT            = 1.0 / 60.0
TABLE_HEIGHT  = 0.75
ROBOT_SPACING = 1.00
MJCF_ORIG     = "xml/universal_robots_ur10e/ur10e_2f85.xml"   # path relativo en assets/
# MJCF_LOCAL: el XML modificado se guarda junto al original en el paquete Genesis

# Home del brazo (6 DOF), el gripper empieza abierto (0.0)
HOME_ARM = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)

# Joints del brazo (por nombre) — se usan para filtrar DOFs
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
# Joint actuado del gripper
GRIPPER_JOINT_NAME = "right_driver_joint"


# ──────────────────────────────────────────────────────────────────────────────
# Fix XML: cambiar integrador a RK4 y guardar copia local
# ──────────────────────────────────────────────────────────────────────────────

def prepare_xml() -> str:
    """
    Genera un XML con integrador RK4 guardado JUNTO AL XML original de Genesis.
    Asi los paths relativos de assets siguen funcionando sin cambios.
    Fix del gripper 2F-85 (issue #40 mujoco_menagerie).
    """
    genesis_xml_dir  = os.path.join(
        os.path.dirname(gs.__file__),
        "assets", "xml", "universal_robots_ur10e"
    )
    src_xml   = os.path.join(genesis_xml_dir, "ur10e_2f85.xml")
    out_xml   = os.path.join(genesis_xml_dir, "ur10e_2f85_rk4.xml")

    with open(src_xml, "r", encoding="utf-8") as f:
        xml = f.read()

    # Cambiar integrador implicitfast -> RK4
    xml = re.sub(
        r'<option([^>]*)integrator="implicitfast"([^>]*)/>'
,
        r'<option\1integrator="RK4"\2/>'
,
        xml,
    )

    with open(out_xml, "w", encoding="utf-8") as f:
        f.write(xml)

    print(f"[prepare_xml] XML con RK4 guardado -> {out_xml}")
    return out_xml


def smoothstep(s: float) -> float:
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


def get_arm_and_gripper_dofs(robot):
    """
    Separa los DOFs en:
      dofs_arm     -> 6 joints del brazo (shoulder ... wrist_3)
      dofs_gripper -> 1 joint actuado del gripper (right_driver)
    """
    dofs_arm, dofs_gripper = [], []
    for j in robot.joints:
        if j.name in ARM_JOINT_NAMES:
            dofs_arm.append(j.dof_idx_local)
        elif j.name == GRIPPER_JOINT_NAME:
            dofs_gripper.append(j.dof_idx_local)

    print(f"  Brazo DOFs    : {dofs_arm}")
    print(f"  Gripper DOFs  : {dofs_gripper}")
    return (
        np.array(dofs_arm,     dtype=np.int32),
        np.array(dofs_gripper, dtype=np.int32),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────────────────

class DualUR10eEnv:
    """
    Entorno dual UR10e con:
      - Mesa estatica
      - Gripper 2F-85 visible (fix RK4)
      - DOFs separados: brazo y gripper
      - Camaras eye-in-hand en wrist_3_link
    """

    def __init__(self, show_viewer: bool = False):
        # ── Preparar XML con fix del gripper ─────────────────────────────────
        mjcf_path = prepare_xml()

        gs.init()

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            viewer_options=gs.options.ViewerOptions(
                camera_pos    = (0.0, -2.8, 2.0),
                camera_lookat = (0.0,  0.0, TABLE_HEIGHT),
                camera_fov    = 50,
            ),
        )

        # ── Suelo ─────────────────────────────────────────────────────────────
        self.scene.add_entity(gs.morphs.Plane())

        # ── Mesa ──────────────────────────────────────────────────────────────
        THICKNESS = 0.05
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                size  = (ROBOT_SPACING + 0.60, 0.90, THICKNESS),
                pos   = (0.0, 0.0, TABLE_HEIGHT - THICKNESS / 2),
                fixed = True,
            )
        )

        # ── Robots ────────────────────────────────────────────────────────────
        half = ROBOT_SPACING / 2.0

        self.robot_l = self.scene.add_entity(
            gs.morphs.MJCF(
                file  = mjcf_path,
                pos   = (-half, 0.0, TABLE_HEIGHT),
                euler = (0.0, 0.0, 0.0),
            )
        )
        self.robot_r = self.scene.add_entity(
            gs.morphs.MJCF(
                file  = mjcf_path,
                pos   = ( half, 0.0, TABLE_HEIGHT),
                euler = (0.0, 0.0, 180.0),
            )
        )

        # ── Camaras eye-in-hand ───────────────────────────────────────────────
        # Se adjuntan despues del build(); aqui solo se declaran
        self.cam_l = None
        self.cam_r = None

        # ── Build ─────────────────────────────────────────────────────────────
        self.scene.build()

        # ── DOFs separados ────────────────────────────────────────────────────
        print("\n[Robot L]")
        self.dofs_arm_l, self.dofs_grip_l = get_arm_and_gripper_dofs(self.robot_l)
        print("[Robot R]")
        self.dofs_arm_r, self.dofs_grip_r = get_arm_and_gripper_dofs(self.robot_r)

        # ── Adjuntar camaras al wrist_3_link ─────────────────────────────────
        self._attach_cameras()

        # ── Estado inicial ────────────────────────────────────────────────────
        self.q_arm_l  = np.zeros(len(self.dofs_arm_l),  dtype=np.float32)
        self.q_arm_r  = np.zeros(len(self.dofs_arm_r),  dtype=np.float32)
        self.q_grip_l = np.array([0.0], dtype=np.float32)   # abierto
        self.q_grip_r = np.array([0.0], dtype=np.float32)

    def _attach_cameras(self):
        """Adjunta camaras RGB al link wrist_3_link de cada robot."""
        try:
            link_l = self.robot_l.get_link("wrist_3_link")
            link_r = self.robot_r.get_link("wrist_3_link")

            self.cam_l = self.scene.add_camera(
                res       = (640, 480),
                pos       = (0.0, 0.1, 0.0),   # offset desde el wrist
                lookat    = (0.0, 0.2, 0.0),
                fov       = 60,
                attach_to = link_l,
                name      = "cam_left",
            )
            self.cam_r = self.scene.add_camera(
                res       = (640, 480),
                pos       = (0.0, 0.1, 0.0),
                lookat    = (0.0, 0.2, 0.0),
                fov       = 60,
                attach_to = link_r,
                name      = "cam_right",
            )
            print("\n[Camaras] cam_left y cam_right adjuntadas al wrist_3_link")
        except Exception as e:
            print(f"\n[Camaras] No se pudieron adjuntar: {e}")
            print("  (Genesis puede no soportar attach_to en esta version)")

    # ── API publica ───────────────────────────────────────────────────────────

    def reset(self, seconds: float = 2.5):
        """Mueve ambos brazos a home y abre grippers."""
        # Brazo izquierdo
        self.q_arm_l = move_to(
            self.robot_l, self.scene,
            self.q_arm_l, HOME_ARM, self.dofs_arm_l, seconds
        )
        # Brazo derecho
        self.q_arm_r = move_to(
            self.robot_r, self.scene,
            self.q_arm_r, HOME_ARM, self.dofs_arm_r, seconds
        )

    def set_gripper(self, side: str, value: float, seconds: float = 1.0):
        """
        Controla el gripper.
        side  : 'left' o 'right'
        value : 0.0 = abierto, 0.8 = cerrado
        """
        value = float(np.clip(value, 0.0, 0.8))
        q_target = np.array([value], dtype=np.float32)
        if side == "left":
            self.q_grip_l = move_to(
                self.robot_l, self.scene,
                self.q_grip_l, q_target, self.dofs_grip_l, seconds
            )
        else:
            self.q_grip_r = move_to(
                self.robot_r, self.scene,
                self.q_grip_r, q_target, self.dofs_grip_r, seconds
            )

    def move_arm(self, side: str, q_target: np.ndarray, seconds: float = 2.0):
        """Mueve el brazo (6 DOF) a la posicion objetivo."""
        if side == "left":
            self.q_arm_l = move_to(
                self.robot_l, self.scene,
                self.q_arm_l, q_target, self.dofs_arm_l, seconds
            )
        else:
            self.q_arm_r = move_to(
                self.robot_r, self.scene,
                self.q_arm_r, q_target, self.dofs_arm_r, seconds
            )

    def get_camera_rgb(self, side: str):
        """Devuelve imagen RGB (H,W,3) de la camara eye-in-hand."""
        cam = self.cam_l if side == "left" else self.cam_r
        if cam is None:
            return None
        cam.render()
        return cam.get_rgb()

    def wave_arm(self, side: str, seconds: float = 3.0):
        robot = self.robot_l if side == "left" else self.robot_r
        q     = self.q_arm_l if side == "left" else self.q_arm_r
        dofs  = self.dofs_arm_l if side == "left" else self.dofs_arm_r
        wave(robot, self.scene, q, dofs, idx=5, seconds=seconds)

    def step(self, n: int = 1):
        for _ in range(n):
            self.scene.step()
            time.sleep(DT)

    def close(self):
        self.scene.close()


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def run_demo(show_viewer: bool):
    env = DualUR10eEnv(show_viewer=show_viewer)

    print("\n-> HOME ...")
    env.reset()

    print("-> Cerrando gripper izquierdo ...")
    env.set_gripper("left",  value=0.8, seconds=1.5)

    print("-> Cerrando gripper derecho ...")
    env.set_gripper("right", value=0.8, seconds=1.5)

    print("-> Abriendo ambos grippers ...")
    env.set_gripper("left",  value=0.0, seconds=1.5)
    env.set_gripper("right", value=0.0, seconds=1.5)

    print("-> Wave brazo izquierdo ...")
    env.wave_arm("left",  seconds=3.0)

    print("-> Wave brazo derecho ...")
    env.wave_arm("right", seconds=3.0)

    print("-> Loop infinito (Ctrl+C para salir)")
    try:
        while True:
            env.step()
    except KeyboardInterrupt:
        print("\nDetenido.")

    env.close()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual UR10e + Gripper + Mesa")
    parser.add_argument("--vis",  action="store_true", help="Activar viewer 3D")
    parser.add_argument("--demo", action="store_true", help="Ejecutar demo completo")
    args = parser.parse_args()

    if args.demo:
        run_demo(show_viewer=args.vis)
    else:
        env = DualUR10eEnv(show_viewer=args.vis)
        print("\n[OK] Entorno listo.")
        env.reset()
        print("[OK] Home alcanzado. Ctrl+C para salir.")
        try:
            while True:
                env.step()
        except KeyboardInterrupt:
            pass
        env.close()