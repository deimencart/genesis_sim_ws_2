import os
import time
import numpy as np
import genesis as gs

def main():
    # 1. Inicializar Genesis (Usa CPU si GPU te da problemas para probar)
    gs.init(backend=gs.gpu)

    # 2. Crear la escena básica
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    # 3. Cargar el robot (usando la ruta de tus assets de Genesis)
    d = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    xml_path = os.path.join(d, "ur10e_2f85.xml")
    
    print(f"Cargando robot desde: {xml_path}")
    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0, 0, 0)))

    # 4. Construir la escena (Obligatorio antes de simular)
    scene.build()

    # 5. Identificar las articulaciones del brazo
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    # Obtener los índices de los grados de libertad (DOFs)
    dofs_arm = np.array([j.dof_idx_local for j in robot.joints if j.name in arm_joint_names])

    # Definir dos posiciones articulares para alternar
    pos_A = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)
    pos_B = np.array([1.5, -1.00, 1.00, -1.00, -1.57, 0.0], dtype=np.float32)

    print("Iniciando simulación...")

    # 6. Bucle de control
    t = 0
    try:
        while True:
            # Usar una onda senoidal para interpolar suavemente entre pos_A y pos_B
            # Esto genera un movimiento oscilante de ida y vuelta
            alpha = (np.sin(t * 0.02) + 1) / 2  # Valor entre 0 y 1
            
            target_q = (1 - alpha) * pos_A + alpha * pos_B
            
            # Enviar el comando al robot
            robot.control_dofs_position(target_q, dofs_arm)
            
            # Avanzar un paso en el simulador
            scene.step()
            t += 1
            
    except KeyboardInterrupt:
        print("\nSimulación terminada por el usuario.")

if __name__ == "__main__":
    main()