# train_model_v3.py
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


def find_latest_best_model():
    """
    Encuentra la ruta al modelo 'best.pt' de la ejecución de entrenamiento MÁS RECIENTE.
    """
    try:
        runs_dir = 'runs/detect'
        all_run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not all_run_dirs:
            logging.warning("No se encontraron carpetas de ejecución en 'runs/detect'.")
            return None

        latest_run_dir = max(all_run_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))

        model_path = os.path.join(runs_dir, latest_run_dir, 'weights', 'best.pt')

        if os.path.exists(model_path):
            logging.info(f"Modelo 'best.pt' encontrado en la última ejecución: {model_path}")
            return model_path
        else:
            logging.warning(f"No se encontró 'best.pt' en la carpeta de ejecución más reciente: {latest_run_dir}")
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
    model_v2_path = find_latest_best_model()
    if not model_v2_path or 'v2' not in model_v2_path:
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
            epochs=50,
            patience=10,
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
