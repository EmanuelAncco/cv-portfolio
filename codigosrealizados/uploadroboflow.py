import os
import logging
from roboflow import Roboflow

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("roboflow_upload_final.log", mode='w'),
        logging.StreamHandler()
    ]
)


def upload_model_with_manifest(api_key, workspace_id, project_id, model_run_path, model_name):
    """
    Sube un modelo YOLOv8 a Roboflow, asegurándose de incluir el contexto
    del directorio de la ejecución (que contiene el archivo .yaml).

    Args:
        api_key (str): Tu API Key de Roboflow.
        workspace_id (str): El ID de tu workspace en Roboflow (de la URL).
        project_id (str): El ID de tu proyecto de despliegue virgen (de la URL).
        model_run_path (str): La ruta a la carpeta principal de la ejecución del entrenamiento
                                (ej. '.../runs/detect/emarc_vision_v3_run13').
        model_name (str): El nombre que le darás a tu modelo en Roboflow.
    """
    logging.info("Iniciando la subida del modelo a Roboflow (con manifiesto .yaml).")
    try:
        # Inicializar Roboflow
        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace_id)

        # Validar la ruta principal de la ejecución
        full_run_path = os.path.abspath(model_run_path)
        weights_file_path = os.path.join(full_run_path, 'weights', 'best.pt')

        if not os.path.exists(full_run_path):
            raise FileNotFoundError(f"El directorio de la ejecución no existe: {full_run_path}")
        if not os.path.exists(weights_file_path):
            raise FileNotFoundError(f"El archivo 'best.pt' no se encontró dentro de: {full_run_path}")

        logging.info(f"Ruta de la ejecución del modelo validada: {full_run_path}")

        # La función deploy_model es suficientemente inteligente.
        # Al pasarle la ruta de la ejecución, buscará el archivo .yaml
        # y los pesos en la subcarpeta 'weights'.
        workspace.deploy_model(
            model_type="yolov8",
            model_path=full_run_path,  # <-- CORRECCIÓN CLAVE
            project_ids=[project_id],
            model_name=model_name,
            filename="weights/best.pt"  # Opcional, pero bueno ser explícito
        )
        logging.info(f"Modelo '{model_name}' subido exitosamente al proyecto '{project_id}'.")
        print("\n¡Subida completada! Revisa tu proyecto en Roboflow. Las clases deberían haber aparecido.")

    except Exception as e:
        logging.error(f"Error inesperado al subir el modelo: {e}", exc_info=True)
        print(f"\nOcurrió un error inesperado: {e}")
        print("Asegúrate de que tu API Key, IDs de workspace/proyecto y la ruta a la ejecución son correctos.")


if __name__ == "__main__":
    # --- CONFIGURACIÓN PARA TU PROYECTO EMARC VISIÓN ---
    YOUR_API_KEY = "2ZAzAa1cz8ncqfiH7xjW"

    # --- IDs de tu workspace y del proyecto de despliegue virgen ---
    # Reemplaza con los IDs que ves en la URL de Roboflow.
    YOUR_WORKSPACE_ID = "emairc-vision"  # Reemplaza con tu ID real
    YOUR_PROJECT_ID = "emarc-v3-production-deploy-5-jl8ne"  # Reemplaza con el ID de tu proyecto VIRGEN

    # --- CORRECCIÓN DE RUTA ---
    # Apunta a la carpeta principal de la ejecución, NO a la subcarpeta 'weights'.
    YOUR_MODEL_RUN_DIRECTORY = r"D:\Python_proyectos_2025\GAIATECH\runs\detect\emarc_vision_v3_run13"

    # Un nombre nuevo y final para tu modelo en Roboflow.
    YOUR_MODEL_NAME_IN_ROBOFLOW = "EMARC-V3-PRODUCCION"

    # --- EJECUCIÓN ---
    upload_model_with_manifest(
        api_key=YOUR_API_KEY,
        workspace_id=YOUR_WORKSPACE_ID,
        project_id=YOUR_PROJECT_ID,
        model_run_path=YOUR_MODEL_RUN_DIRECTORY,
        model_name=YOUR_MODEL_NAME_IN_ROBOFLOW
    )
