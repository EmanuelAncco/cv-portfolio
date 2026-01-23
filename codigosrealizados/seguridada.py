# train_model_v1.py
import os
import yaml
import torch
import logging
import shutil
import random
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# --- CONFIGURACIÓN DEL LOGGING ---
# Configura un sistema de logging para registrar información y errores.
# Se guardará en 'training.log' y se mostrará en la consola.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),  # 'w' para sobreescribir el log en cada ejecución
        logging.StreamHandler()
    ]
)


def analyze_dataset(data_yaml_path):
    """
    Analiza el dataset para generar estadísticas y gráficos de distribución de clases.
    Es robusto ante claves faltantes o inconsistencias ('val' vs 'valid') en el YAML.
    """
    logging.info("--- Fase I: Iniciando análisis del dataset ---")
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        class_names = data.get('names', [])
        if not class_names:
            logging.critical("No se encontraron 'names' en el archivo YAML. No se puede continuar.")
            return False

        all_labels = []

        # Bucle robusto que maneja inconsistencias y claves faltantes
        for subset_name, yaml_key in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
            logging.info(f"Procesando subset: {subset_name}")

            if yaml_key not in data:
                logging.warning(f"La clave '{yaml_key}' no se encontró en data.yaml. Saltando este subset.")
                continue

            # Construir la ruta al directorio de etiquetas
            images_path = data[yaml_key]
            # Asumimos que la ruta en el YAML es relativa al propio YAML
            base_dir = os.path.dirname(data_yaml_path)
            label_dir = os.path.join(base_dir, images_path.replace('images', 'labels'))

            if not os.path.isdir(label_dir):
                logging.warning(f"Directorio de etiquetas no encontrado para '{subset_name}': {label_dir}")
                continue

            for label_file in os.listdir(label_dir):
                if not label_file.endswith('.txt'): continue
                try:
                    with open(os.path.join(label_dir, label_file), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                all_labels.append(int(parts[0]))
                except Exception as e:
                    logging.error(f"Error procesando el archivo {label_file}: {e}")

        if not all_labels:
            logging.error("Análisis completado, pero no se encontraron etiquetas válidas en ningún subset.")
            return False

        # Generar y guardar gráfico de distribución de clases
        class_counts = Counter(all_labels)
        class_indices = sorted(class_counts.keys())
        counts = [class_counts[i] for i in class_indices]
        names = [class_names[i] for i in class_indices]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=counts, y=names, palette="viridis",
                    log=True)  # Usar escala logarítmica para mejor visualización si hay desbalance
        plt.title('Distribución de Clases en el Dataset Completo (Escala Log)')
        plt.xlabel('Número de Instancias (log scale)')
        plt.ylabel('Clases')
        plt.tight_layout()
        # Guardar el gráfico en el directorio del proyecto, no donde se ejecuta el script
        output_graph_path = os.path.join(os.path.dirname(data_yaml_path), 'class_distribution.png')
        plt.savefig(output_graph_path)
        logging.info(f"Gráfico de distribución de clases guardado como '{output_graph_path}'")
        plt.close()

        # Guardar estadísticas del dataset en un archivo de texto
        output_stats_path = os.path.join(os.path.dirname(data_yaml_path), 'dataset_stats.txt')
        with open(output_stats_path, 'w') as f:
            f.write("--- Estadísticas del Dataset EMARC VISIÓN v1 ---\n")
            f.write(f"Número total de clases: {data.get('nc', 'No especificado')}\n")
            f.write(f"Nombres de clases: {class_names}\n\n")
            f.write("Distribución de instancias por clase:\n")
            for i in class_indices:
                f.write(f"  - {class_names[i]} (ID: {i}): {class_counts[i]} instancias\n")
        logging.info(f"Estadísticas del dataset guardadas en '{output_stats_path}'")

    except Exception as e:
        logging.critical(f"Error fatal durante el análisis del dataset: {e}")
        logging.critical(f"Traceback completo:\n{traceback.format_exc()}")
        return False
    return True


def run_qualitative_evaluation(model_path, data_yaml_path, num_images=15):
    """
    Ejecuta el modelo sobre una muestra de imágenes de test para evaluación cualitativa.
    """
    logging.info("--- Fase III (Post): Iniciando evaluación cualitativa ---")
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'test' not in data:
            logging.warning("No se encontró el conjunto de 'test' en el YAML. Saltando evaluación cualitativa.")
            return

        test_images_dir = os.path.join(os.path.dirname(data_yaml_path), data['test'])

        if not os.path.isdir(test_images_dir):
            logging.error(f"Directorio de imágenes de test no encontrado: {test_images_dir}")
            return

        image_files = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            logging.warning("No se encontraron imágenes en el directorio de test.")
            return

        sample_images = random.sample(image_files, min(num_images, len(image_files)))

        model = YOLO(model_path)
        results = model(sample_images)

        # Guardar resultados en la carpeta del modelo, no en la raíz
        model_run_dir = os.path.dirname(os.path.dirname(model_path))  # Sube dos niveles desde 'best.pt'
        output_dir = os.path.join(model_run_dir, 'qualitative_results')
        os.makedirs(output_dir, exist_ok=True)

        for i, r in enumerate(results):
            r.save(filename=os.path.join(output_dir, f'result_{i}.jpg'))

        logging.info(f"Resultados de evaluación cualitativa guardados en: {output_dir}")

    except Exception as e:
        logging.error(f"Error durante la evaluación cualitativa: {e}")
        logging.error(f"Traceback completo:\n{traceback.format_exc()}")


def main():
    """
    Función principal que orquesta el análisis, entrenamiento y evaluación del modelo.
    """
    logging.info("--- Iniciando Proceso de Entrenamiento para EMARC VISIÓN v1 ---")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info(f"GPU detectada: {torch.cuda.get_device_name(0)}. Usando CUDA.")
    else:
        device = torch.device("cpu")
        logging.warning("No se detectó GPU. El entrenamiento se ejecutará en la CPU y será muy lento.")

    base_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0'
    dataset_path = os.path.join(base_path, 'dataset_filtrado')
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')

    if not os.path.exists(data_yaml_path):
        logging.critical(f"Error fatal: El archivo 'data.yaml' no se encuentra en la ruta esperada: {data_yaml_path}")
        return

    if not analyze_dataset(data_yaml_path):
        logging.critical("El análisis del dataset falló. El entrenamiento no puede continuar.")
        return

    logging.info("--- Fase II: Iniciando entrenamiento del modelo ---")

    model = YOLO('yolov8s.pt')
    model.to(device)

    try:
        results = model.train(
            data=data_yaml_path,
            epochs=100,
            imgsz=640,
            device=0,
            patience=20,
            batch=16,
            name='emarc_vision_v1_run1'
        )

        logging.info("¡Entrenamiento completado exitosamente!")

        results_dir = results.save_dir
        best_model_path = os.path.join(results_dir, 'weights', 'best.pt')

        logging.info("--- Resumen del Entrenamiento ---")
        logging.info(f"Métricas finales guardadas en la carpeta de resultados.")
        logging.info(f"El mejor modelo se encuentra en: {best_model_path}")
        logging.info(f"Todos los gráficos y artefactos están en: {results_dir}")

        run_qualitative_evaluation(best_model_path, data_yaml_path)

    except Exception as e:
        logging.critical(f"Ocurrió un error crítico durante el entrenamiento: {e}")
        logging.critical(f"Traceback completo:\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
