import os
import numpy as np
import genesis as gs

def main():
    # 1. Inicializar
    gs.init(backend=gs.gpu)

    # 2. Crear la escena
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    # 3. Cargar el robot (UR10e) - ¡Lo ponemos a 1 metro de altura!
    d = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    xml_path = os.path.join(d, "ur10e_2f85.xml")
    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0.0, 0.0, 1.0)))

    # Construir la escena
    scene.build()

    # 4. Mapear articulaciones
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    dofs_arm = np.array([j.dof_idx_local for j in robot.joints if j.name in arm_joint_names])

    # 5. Definir estados
    # Un array de ceros en el UR10e lo deja totalmente estirado horizontalmente
    #pose_estirada = np.zeros(len(dofs_arm), dtype=np.float32)
    # 5. Definir estados
    # Levantamos el hombro -90 grados (-1.57 rad) para que apunte al techo
    pose_estirada = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Un array de ceros para la fuerza (apagará los motores)
    cero_fuerza = np.zeros(len(dofs_arm), dtype=np.float32)

    # Forzar la posición inicial al instante (Teletransporte)
    robot.set_dofs_position(pose_estirada, dofs_arm)

    print("🤖 Manteniendo el robot estirado con los motores a tope...")

    # 6. Bucle de simulación
    t = 0
    try:
        while True:
            if t < 150: 
                # FASE 1: Motores encendidos manteniendo la posición (aprox 2.5 segundos)
                robot.control_dofs_position(pose_estirada, dofs_arm)
                
            elif t == 150:
                # Transición
                print("💥 ¡APAGANDO MOTORES! ¡Caída libre!")
                robot.control_dofs_force(cero_fuerza, dofs_arm)
                
            else:
                # FASE 2: Motores apagados, aplicando 0 torque constantemente
                robot.control_dofs_force(cero_fuerza, dofs_arm)
            
            # Avanzar la simulación
            scene.step()
            t += 1
            
    except KeyboardInterrupt:
        print("\nSimulación terminada.")

if __name__ == "__main__":
    main()