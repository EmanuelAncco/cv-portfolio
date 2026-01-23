# train_model_v3.1.py (v3.4 - Control Manual del Optimizador para Fine-Tuning)
import os
import yaml
import torch
import logging
import traceback
from ultralytics import YOLO

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training_v3.1_manual_lr.log", mode='w'),
        logging.StreamHandler()
    ]
)


def find_latest_v3_model():
    """
    Encuentra la ruta al 'best.pt' de la ejecución MÁS RECIENTE que contenga 'v3'
    en su nombre, excluyendo las ejecuciones de fine-tuning para evitar errores.
    """
    try:
        runs_dir = 'runs/detect'
        v3_base_runs = [
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d)) and 'v3' in d and 'finetune' not in d
        ]

        if not v3_base_runs:
            logging.error("No se encontraron carpetas de ejecución de v3 base (que no sean de fine-tuning).")
            return None

        latest_v3_run = max(v3_base_runs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
        model_path = os.path.join(runs_dir, latest_v3_run, 'weights', 'best.pt')

        if os.path.exists(model_path):
            logging.info(f"Modelo v3 'best.pt' encontrado para fine-tuning en: {model_path}")
            return model_path
        else:
            logging.warning(f"No se encontró 'best.pt' en la carpeta de ejecución v3 más reciente: {latest_v3_run}")
            return None
    except FileNotFoundError:
        logging.error("'runs/detect' no encontrado. Asegúrate de estar en el directorio correcto.")
        return None


def main():
    """
    Función principal que orquesta el re-entrenamiento para crear el Modelo v3.1.
    """
    logging.info("--- Iniciando Proceso de Re-entrenamiento para EMARC VISIÓN v3.1 (Control Manual) ---")

    # --- FASE 0: CONFIGURACIÓN Y VERIFICACIONES ---
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logging.warning("No se detectó GPU. El entrenamiento será muy lento.")
    else:
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}. Usando CUDA.")

    dataset_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced'
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')

    if not os.path.exists(data_yaml_path):
        logging.critical(f"Error fatal: El archivo 'data.yaml' del dataset no se encuentra en: {data_yaml_path}")
        return

    # --- FASE I: ENTRENAMIENTO DE AJUSTE FINO (FINE-TUNING v3 -> v3.1) ---
    logging.info("--- Fase I: Iniciando re-entrenamiento del Modelo v3.1 ---")

    model_v3_path = find_latest_v3_model()
    if not model_v3_path:
        logging.error("CRÍTICO: No se encontró el modelo v3 ('best.pt') para el fine-tuning. El proceso no puede continuar.")
        return

    logging.info(f"Iniciando entrenamiento desde pesos: {model_v3_path}")
    model = YOLO(model_v3_path)
    model.to(device)

    try:
        # AJUSTE CRÍTICO FINAL: Tomamos el control total del optimizador.
        # Al especificar un optimizador, forzamos a la librería a usar
        # la tasa de aprendizaje que nosotros definimos.
        results = model.train(
            data=data_yaml_path,
            epochs=30,
            patience=15,
            batch=16,
            imgsz=640,
            device=device,
            name='emarc_vision_v3.1_finetune_manual_lr', # Nuevo nombre para esta ejecución.
            cache=True,
            optimizer='AdamW',  # Especificamos el optimizador manualmente.
            lr0=0.001,      # Ahora esta tasa de aprendizaje SÍ será utilizada.
            lrf=0.0001      # Podemos hacer que la tasa baje aún más hacia el final.
        )

        logging.info("¡Re-entrenamiento del Modelo v3.1 completado!")

        results_dir = results.save_dir
        best_model_path = os.path.join(results_dir, 'weights', 'best.pt')

        logging.info("--- Resumen del Re-entrenamiento v3.1 ---")
        logging.info(f"El mejor modelo v3.1 se encuentra en: {best_model_path}")
        logging.info(f"Todos los artefactos están en: {results_dir}")

    except Exception as e:
        logging.critical(f"Ocurrió un error crítico durante el entrenamiento: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
