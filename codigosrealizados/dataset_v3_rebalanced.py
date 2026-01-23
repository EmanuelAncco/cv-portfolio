# train_model_v3.py (v3.1 - Lógica de búsqueda de modelo corregida)
import os
import yaml
import torch
import logging
from ultralytics import YOLO
import traceback

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training_v3.log", mode='w'),
        logging.StreamHandler()
    ]
)


def find_latest_v2_model():
    """
    Encuentra la ruta al 'best.pt' de la ejecución MÁS RECIENTE que contenga 'v2' en su nombre.
    Esta es la corrección del bug que impedía el fine-tuning.
    """
    try:
        runs_dir = 'runs/detect'
        # 1. Filtrar solo las carpetas que contienen 'v2' en su nombre.
        v2_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and 'v2' in d]

        if not v2_runs:
            logging.warning("No se encontraron carpetas de ejecución de v2 en 'runs/detect'.")
            return None

        # 2. Encontrar la más reciente de las carpetas v2.
        latest_v2_run = max(v2_runs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))

        model_path = os.path.join(runs_dir, latest_v2_run, 'weights', 'best.pt')

        if os.path.exists(model_path):
            logging.info(f"Modelo v2 'best.pt' encontrado para fine-tuning en: {model_path}")
            return model_path
        else:
            logging.warning(f"No se encontró 'best.pt' en la carpeta de ejecución v2 más reciente: {latest_v2_run}")
            return None
    except FileNotFoundError:
        logging.error("'runs/detect' no encontrado. Asegúrate de estar en el directorio correcto.")
        return None


def main():
    """
    Función principal que orquesta el entrenamiento del Modelo v3.
    """
    logging.info("--- Iniciando Proceso de Entrenamiento para EMARC VISIÓN v3 ---")

    # --- FASE 0: CONFIGURACIÓN Y VERIFICACIONES ---
    if not torch.cuda.is_available():
        logging.warning("No se detectó GPU. El entrenamiento será muy lento.")
        device = "cpu"
    else:
        device = 0
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}. Usando CUDA.")

    dataset_v3_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced'
    data_yaml_path = os.path.join(dataset_v3_path, 'data.yaml')

    if not os.path.exists(data_yaml_path):
        logging.critical(f"Error fatal: El archivo 'data.yaml' del dataset v3 no se encuentra en: {data_yaml_path}")
        return

    # --- FASE I: ENTRENAMIENTO INCREMENTAL (FINE-TUNING v2 -> v3) ---
    logging.info("--- Fase I: Iniciando entrenamiento del Modelo v3 ---")

    # Cargar el mejor modelo de la v2 para iniciar el fine-tuning
    model_v2_path = find_latest_v2_model()
    if not model_v2_path:
        logging.warning(
            "No se encontró el modelo v2 ('best.pt'). Empezando desde pesos pre-entrenados de YOLOv8s. El entrenamiento será más largo.")
        model_v2_path = 'yolov8s.pt'

    logging.info(f"Iniciando entrenamiento desde pesos: {model_v2_path}")
    model = YOLO(model_v2_path)
    model.to(device)

    try:
        # El entrenamiento ahora tiene más datos, por lo que le damos más épocas para aprender
        results = model.train(
            data=data_yaml_path,
            epochs=75,
            patience=15,
            batch=16,
            imgsz=640,
            device=device,
            name='emarc_vision_v3_run1',
            cache=True  # Usar caché para acelerar la carga de datos en épocas sucesivas
        )

        logging.info("¡Entrenamiento del Modelo v3 completado!")

        results_dir = results.save_dir
        best_model_path = os.path.join(results_dir, 'weights', 'best.pt')

        logging.info("--- Resumen del Entrenamiento v3 ---")
        logging.info(f"El mejor modelo v3 se encuentra en: {best_model_path}")
        logging.info(f"Todos los artefactos están en: {results_dir}")

    except Exception as e:
        logging.critical(f"Ocurrió un error crítico durante el entrenamiento: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
