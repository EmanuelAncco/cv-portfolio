import os
import logging
from roboflow import Roboflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

def deploy_to_version(api_key, workspace_id, project_id, version_number, model_run_path):
    """
    Sube un modelo YOLOv8 como un checkpoint a una versión específica de un dataset.

    Args:
        api_key (str): Tu API Key de Roboflow.
        workspace_id (str): El ID de tu workspace.
        project_id (str): El ID de tu proyecto.
        version_number (int): El número de la versión del dataset a la que se anclará el modelo.
        model_run_path (str): La ruta a la carpeta principal de la ejecución del entrenamiento.
    """
    logging.info(f"Iniciando despliegue a la versión {version_number} del proyecto {project_id}.")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_id).project(project_id)
        version = project.version(version_number)

        logging.info(f"Ruta de la ejecución del modelo: {model_run_path}")

        # Desplegar el modelo directamente a la versión del dataset
        version.deploy(model_type="yolov8", model_path=model_run_path)

        logging.info("¡Despliegue completado exitosamente!")
        print("\n¡Éxito! El modelo ha sido subido y asociado a la versión del dataset.")
        print("Revisa la sección 'Models' o 'Deployments' en tu proyecto de Roboflow.")

    except Exception as e:
        logging.error(f"Error inesperado durante el despliegue a versión: {e}", exc_info=True)
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    YOUR_API_KEY = "2ZAzAa1cz8ncqfiH7xjW"
    YOUR_WORKSPACE_ID = "emairc-vision" # Reemplaza si es necesario

    # El ID del proyecto donde creaste las clases y subiste la imagen
    YOUR_PROJECT_ID = "emarc-v3-production-deploy-5-jl8ne" # Reemplaza con el ID correcto

    # El número de la versión que generaste en el Paso 1
    DATASET_VERSION_NUMBER = 1

    # La ruta a la carpeta principal de la ejecución
    YOUR_MODEL_RUN_DIRECTORY = r"D:\Python_proyectos_2025\GAIATECH\runs\detect\emarc_vision_v3_run13"

    deploy_to_version(
        api_key=YOUR_API_KEY,
        workspace_id=YOUR_WORKSPACE_ID,
        project_id=YOUR_PROJECT_ID,
        version_number=DATASET_VERSION_NUMBER,
        model_run_path=YOUR_MODEL_RUN_DIRECTORY
    )