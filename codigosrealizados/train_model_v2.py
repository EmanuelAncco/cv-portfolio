# train_model_v2.py
import os
import yaml
import torch
import logging
import random
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training_v2.log", mode='w'),
        logging.StreamHandler()
    ]
)


def analyze_dataset(data_yaml_path):
    """
    Analiza el nuevo dataset para generar estadísticas y gráficos.
    Es una función crucial para la sección de metodología del paper.
    """
    logging.info("--- Fase I (v2): Iniciando análisis del nuevo dataset ---")
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        class_names = data.get('names', [])
        if not class_names:
            logging.critical("No se encontraron 'names' en el archivo YAML.")
            return False

        all_labels = []

        for subset_name, yaml_key in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            if yaml_key not in data:
                logging.warning(f"La clave '{yaml_key}' no se encontró en data.yaml. Saltando subset.")
                continue

            base_dir = os.path.dirname(data_yaml_path)
            # Roboflow exporta rutas relativas, así que las construimos desde la base del yaml
            label_dir = os.path.join(base_dir, data[yaml_key].replace('../', '').replace('images', 'labels'))

            if not os.path.isdir(label_dir):
                logging.warning(f"Directorio de etiquetas no encontrado para '{subset_name}': {label_dir}")
                continue

            for label_file in os.listdir(label_dir):
                if not label_file.endswith('.txt'): continue
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts: all_labels.append(int(parts[0]))

        if not all_labels:
            logging.error("Análisis completado, pero no se encontraron etiquetas válidas.")
            return False

        class_counts = Counter(all_labels)
        class_indices = sorted(class_counts.keys())
        counts = [class_counts[i] for i in class_indices]
        names = [class_names[i] for i in class_indices]

        if not counts:
            logging.warning("No hay datos para graficar. Saltando la creación del gráfico de distribución.")
        else:
            plt.figure(figsize=(14, 10))
            # FIX: Generar el gráfico en escala lineal y luego cambiar la escala del eje X a logarítmica.
            # Este enfoque es más robusto que usar 'log=True' directamente en barplot.
            ax = sns.barplot(x=counts, y=names, palette="viridis")
            ax.set_xscale('log')

            # Usar el objeto 'ax' para configurar el gráfico
            ax.set_title('Distribución de Clases en el Dataset v2 (Escala Log)')
            ax.set_xlabel('Número de Instancias (escala logarítmica)')
            ax.set_ylabel('Clases')
            plt.tight_layout()

            output_graph_path = os.path.join(os.path.dirname(data_yaml_path), 'class_distribution_v2.png')
            plt.savefig(output_graph_path)
            logging.info(f"Gráfico de distribución de clases v2 guardado en '{output_graph_path}'")
            plt.close()

        output_stats_path = os.path.join(os.path.dirname(data_yaml_path), 'dataset_stats_v2.txt')
        with open(output_stats_path, 'w') as f:
            f.write("--- Estadísticas del Dataset EMARC VISIÓN v2 ---\n")
            f.write(f"Número total de clases: {data.get('nc', 'No especificado')}\n")
            f.write(f"Nombres de clases: {class_names}\n\n")
            f.write("Distribución de instancias por clase:\n")
            for i in class_indices:
                f.write(f"  - {class_names[i]} (ID: {i}): {class_counts[i]} instancias\n")
        logging.info(f"Estadísticas del dataset v2 guardadas en '{output_stats_path}'")

    except Exception as e:
        logging.critical(f"Error fatal durante el análisis del dataset: {e}\n{traceback.format_exc()}")
        return False
    return True


def find_best_model_v1():
    """Encuentra la ruta al modelo 'best.pt' del entrenamiento anterior."""
    try:
        runs_dir = 'runs/detect'
        all_runs = sorted(
            [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))],
            key=os.path.getmtime,
            reverse=True
        )
        if not all_runs: return None
        latest_run_dir = all_runs[0]
        model_path = os.path.join(latest_run_dir, 'weights', 'best.pt')
        return model_path if os.path.exists(model_path) else None
    except FileNotFoundError:
        return None


def main():
    logging.info("--- Iniciando Proceso de Entrenamiento para EMARC VISIÓN v2 ---")

    if not torch.cuda.is_available():
        logging.warning("No se detectó GPU. El entrenamiento será muy lento.")
        device = "cpu"
    else:
        device = 0
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}. Usando CUDA.")

    # --- FASE 0: CONFIGURACIÓN DE RUTAS ---
    new_dataset_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0\PPE Detection.v1-v2.ppe2025-07-22-12-29am.yolov8'
    data_yaml_path = os.path.join(new_dataset_path, 'data.yaml')

    if not os.path.exists(data_yaml_path):
        logging.critical(f"Error fatal: El archivo 'data.yaml' del nuevo dataset no se encuentra en: {data_yaml_path}")
        return

    # --- FASE I: ANÁLISIS DEL NUEVO DATASET ---
    if not analyze_dataset(data_yaml_path):
        logging.critical("El análisis del dataset v2 falló. El entrenamiento no puede continuar.")
        return

    # --- FASE II: ENTRENAMIENTO INCREMENTAL (FINE-TUNING) ---
    logging.info("--- Fase II: Iniciando entrenamiento del Modelo v2 ---")

    # Cargar el mejor modelo de la v1 para iniciar el fine-tuning
    model_v1_path = find_best_model_v1()
    if not model_v1_path:
        logging.warning(
            "No se encontró el modelo v1 ('best.pt'). Empezando desde pesos pre-entrenados de YOLOv8s. El entrenamiento será más largo.")
        model_v1_path = 'yolov8s.pt'

    logging.info(f"Iniciando entrenamiento desde pesos: {model_v1_path}")
    model = YOLO(model_v1_path)
    model.to(device)

    try:
        results = model.train(
            data=data_yaml_path,
            epochs=50,  # Reducimos las épocas gracias al fine-tuning
            patience=10,  # Menos paciencia, esperamos una convergencia más rápida
            batch=16,
            imgsz=640,
            device=device,
            name='emarc_vision_v2_run1'  # Nuevo nombre para esta ejecución
        )

        logging.info("¡Entrenamiento del Modelo v2 completado!")

        results_dir = results.save_dir
        best_model_path = os.path.join(results_dir, 'weights', 'best.pt')

        logging.info("--- Resumen del Entrenamiento v2 ---")
        logging.info(f"El mejor modelo v2 se encuentra en: {best_model_path}")
        logging.info(f"Todos los artefactos están en: {results_dir}")

    except Exception as e:
        logging.critical(f"Ocurrió un error crítico durante el entrenamiento: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()