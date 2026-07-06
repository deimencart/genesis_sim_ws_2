import os
import numpy as np
import genesis as gs

def main():
    # 1. Inicializar Genesis 
    # (Si la GPU te dio problemas antes, cambia backend=gs.gpu por backend=gs.cpu)
    gs.init(backend=gs.gpu)

    # 2. Crear la escena
    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    # 3. Cargar el robot (UR10e)
    d = os.path.join(os.path.dirname(gs.__file__), "assets", "xml", "universal_robots_ur10e")
    xml_path = os.path.join(d, "ur10e_2f85.xml")
    robot = scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(0, 0, 0)))

    # Construir la escena
    scene.build()

    # 4. Mapear las articulaciones del brazo
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    dofs_arm = np.array([j.dof_idx_local for j in robot.joints if j.name in arm_joint_names])

    print("¡El robot está listo para saludar! (Presiona Ctrl+C para salir)")

    # 5. Bucle de animación
    t = 0
    try:
        while True:
            # Calculamos el ángulo de la muñeca para que vaya de lado a lado
            # np.sin() crea el movimiento de vaivén. Multiplicamos por 1.2 para que el saludo sea amplio.
            angulo_saludo = np.sin(t * 0.05) * 1.2 
            
            # Definimos la postura en tiempo real
            target_q = np.array([
                0.0,           # Hombro base (quieto)
                -1.57,         # Levanta el brazo (90 grados hacia arriba)
                0.0,           # Codo (recto)
                -1.57,         # Muñeca 1 (apuntando hacia adelante)
                angulo_saludo, # Muñeca 2 (¡El movimiento de saludar!)
                0.0            # Muñeca 3 (quieta)
            ], dtype=np.float32)
            
            # Enviar el comando al robot
            robot.control_dofs_position(target_q, dofs_arm)
            
            # Avanzar la simulación
            scene.step()
            t += 1
            
    except KeyboardInterrupt:
        print("\n¡Adiós!")

if __name__ == "__main__":
    main()