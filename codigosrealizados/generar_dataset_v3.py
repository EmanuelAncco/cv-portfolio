# generar_dataset_v3.py (v3.5 - Análisis de estructura de directorios robusto)
import os
import shutil
import random
import logging
import yaml
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("generacion_dataset_v3.log", mode='w'),
        logging.StreamHandler()
    ]
)


def analizar_contenido_dataset(dataset_path):
    """
    Analiza un dataset en formato YOLO que CONTIENE un archivo data.yaml.
    Devuelve el conteo de instancias y los nombres de las clases.
    """
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        logging.error(f"No se encontró data.yaml en {dataset_path}")
        return None, None

    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get('names', [])
    conteo_clases = defaultdict(int)

    for subset in ['train', 'valid', 'test']:
        yaml_key = 'val' if subset == 'valid' else subset
        if yaml_key not in data_config or not data_config[yaml_key]:
            continue

        relative_path = data_config[yaml_key].replace('../', '')
        label_dir = os.path.join(os.path.dirname(data_yaml_path), relative_path.replace('images', 'labels'))

        if not os.path.isdir(label_dir):
            continue

        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            class_id = int(line.strip().split()[0])
                            conteo_clases[class_id] += 1
                        except (ValueError, IndexError):
                            continue

    return conteo_clases, class_names


def analizar_directorio_yolo_raw(dataset_path):
    """
    Analiza un dataset en formato YOLO que puede tener una estructura anidada (train/valid/test)
    o una estructura plana (images/labels). Devuelve el mapeo de imagen a clases.
    """
    imagen_a_clases = defaultdict(set)

    # Determinar la estructura del directorio
    if os.path.isdir(os.path.join(dataset_path, 'train')):
        subsets_a_revisar = ['train', 'valid', 'test']
    else:
        subsets_a_revisar = ['']  # Indica una estructura plana

    for subset in subsets_a_revisar:
        labels_path = os.path.join(dataset_path, subset, 'labels')
        images_path = os.path.join(dataset_path, subset, 'images')
        if not os.path.isdir(labels_path) or not os.path.isdir(images_path):
            continue

        for label_filename in os.listdir(labels_path):
            if label_filename.endswith('.txt'):
                image_name_base, _ = os.path.splitext(label_filename)

                possible_exts = ['.jpg', '.jpeg', '.png']
                image_filename_with_ext = None
                for ext in possible_exts:
                    if os.path.exists(os.path.join(images_path, image_name_base + ext)):
                        image_filename_with_ext = image_name_base + ext
                        break

                if image_filename_with_ext:
                    # La ruta relativa preserva la estructura de subcarpetas si existe
                    relative_path = os.path.join(subset, 'images', image_filename_with_ext)
                    with open(os.path.join(labels_path, label_filename), 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                class_id = int(line.strip().split()[0])
                                imagen_a_clases[relative_path].add(class_id)
                            except (ValueError, IndexError):
                                continue

    return imagen_a_clases


def generar_dataset_v3():
    """
    Crea un dataset v3 equilibrado fusionando de forma inteligente el dataset v2
    con subconjuntos de dos datasets de refuerzo diferentes.
    """
    # --- CONFIGURACIÓN ---
    base_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0'
    dataset_v2_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0\PPE Detection.v1-v2.ppe2025-07-22-12-29am.yolov8'
    dataset_refuerzo_general_path = os.path.join(base_path, 'dataset_refuerzo_v3')
    dataset_refuerzo_noboots_path = os.path.join(base_path, 'dataset_refuerzo_noboots')
    dataset_v3_final_path = os.path.join(base_path, 'dataset_v3_final')

    OBJETIVO_INSTANCIAS_POR_CLASE = 4000

    logging.info("--- Iniciando la creación del Dataset v3 Equilibrado (v3.5) ---")

    # --- FASE 1: ANÁLISIS DE LOS DATASETS EXISTENTES ---
    logging.info("Analizando el Dataset v2 (nuestro dataset base)...")
    conteo_v2, nombres_clases_v2 = analizar_contenido_dataset(dataset_v2_path)
    if conteo_v2 is None: return

    logging.info("Analizando el Dataset de Refuerzo General...")
    imagen_a_clases_ref_gral = analizar_directorio_yolo_raw(dataset_refuerzo_general_path)
    if not imagen_a_clases_ref_gral: return

    logging.info("Analizando el Dataset de Refuerzo 'No-Boots'...")
    imagen_a_clases_ref_noboots = analizar_directorio_yolo_raw(dataset_refuerzo_noboots_path)
    if not imagen_a_clases_ref_noboots: return

    # --- FASE 2: SELECCIÓN INTELIGENTE Y RE-MAPEO ---
    logging.info("Calculando necesidades y seleccionando imágenes de refuerzo...")

    mapa_traduccion_v2_a_ref = {
        'No-Boots': ('No-Boots', 0), 'No-Gloves': ('General', 6), 'No-Specs': ('General', 7),
        'Shades': ('General', 2), 'No-Vest': ('General', 10), 'Gloves': ('General', 1),
        'No-Helmet': ('General', 8)
    }

    mapa_ref_a_v2 = {
        'General': {1: 1, 2: 10, 6: 5, 7: 7, 8: 6, 10: 8},  # ID_Ref: ID_v2
        'No-Boots': {0: 4}
    }

    imagenes_a_copiar = defaultdict(lambda: {'fuente': None, 'etiquetas_originales': set()})

    for id_v2, count_v2 in conteo_v2.items():
        if count_v2 < OBJETIVO_INSTANCIAS_POR_CLASE:
            clase_nombre_v2 = nombres_clases_v2[id_v2]
            if clase_nombre_v2 not in mapa_traduccion_v2_a_ref: continue

            necesitadas = OBJETIVO_INSTANCIAS_POR_CLASE - count_v2
            fuente, id_refuerzo_buscado = mapa_traduccion_v2_a_ref[clase_nombre_v2]
            dataset_a_usar = imagen_a_clases_ref_noboots if fuente == "No-Boots" else imagen_a_clases_ref_gral

            candidatos = [img for img, classes in dataset_a_usar.items() if id_refuerzo_buscado in classes]
            random.shuffle(candidatos)

            instancias_añadidas = 0
            for img_path_relativo in candidatos:
                if instancias_añadidas >= necesitadas: break

                imagenes_a_copiar[img_path_relativo]['fuente'] = fuente
                imagenes_a_copiar[img_path_relativo]['etiquetas_originales'].update(dataset_a_usar[img_path_relativo])

                instancias_añadidas += list(dataset_a_usar[img_path_relativo]).count(id_refuerzo_buscado)

    # --- FASE 3: CREACIÓN DEL DATASET v3 FINAL ---
    logging.info(f"Creando la estructura de carpetas para '{dataset_v3_final_path}'...")
    if os.path.exists(dataset_v3_final_path): shutil.rmtree(dataset_v3_final_path)
    shutil.copytree(dataset_v2_path, dataset_v3_final_path)

    logging.info(f"Añadiendo {len(imagenes_a_copiar)} imágenes de refuerzo al conjunto de entrenamiento...")
    destino_train_images = os.path.join(dataset_v3_final_path, 'train', 'images')
    destino_train_labels = os.path.join(dataset_v3_final_path, 'train', 'labels')

    for img_path_relativo, info in imagenes_a_copiar.items():
        fuente = info['fuente']
        origen_base_path = dataset_refuerzo_noboots_path if fuente == "No-Boots" else dataset_refuerzo_general_path

        origen_img_path = os.path.join(origen_base_path, img_path_relativo)

        nombre_base, ext = os.path.splitext(os.path.basename(img_path_relativo))
        origen_lbl_path = os.path.join(os.path.dirname(origen_img_path).replace('images', 'labels'),
                                       nombre_base + '.txt')

        if not os.path.exists(origen_img_path) or not os.path.exists(origen_lbl_path):
            logging.warning(f"Archivo faltante para {img_path_relativo}, omitiendo.")
            continue

        nuevo_nombre_base = f"refuerzo_{fuente.lower()}_{nombre_base}"
        shutil.copy2(origen_img_path, os.path.join(destino_train_images, nuevo_nombre_base + ext))

        with open(origen_lbl_path, 'r', encoding='utf-8') as f_in, open(
                os.path.join(destino_train_labels, nuevo_nombre_base + '.txt'), 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split()
                id_original = int(parts[0])
                mapa_a_usar = mapa_ref_a_v2[fuente]
                if id_original in mapa_a_usar:
                    id_v2_nuevo = mapa_a_usar[id_original]
                    f_out.write(f"{id_v2_nuevo} {' '.join(parts[1:])}\n")

    # --- FASE 4: ANÁLISIS FINAL Y REPORTE ---
    logging.info("--- Análisis del Dataset v3 Final ---")
    conteo_v3, nombres_clases_v3 = analizar_contenido_dataset(dataset_v3_final_path)
    if conteo_v3:
        plt.figure(figsize=(14, 10))
        ax = sns.barplot(x=[conteo_v3.get(i, 0) for i in range(len(nombres_clases_v3))], y=nombres_clases_v3,
                         palette="viridis")
        ax.set_xscale('log')
        ax.set_title('Distribución de Clases en el Dataset v3 Final (Escala Log)')
        ax.set_xlabel('Número de Instancias (escala logarítmica)')
        ax.set_ylabel('Clases')
        plt.tight_layout()
        output_graph_path = os.path.join(dataset_v3_final_path, 'class_distribution_v3.png')
        plt.savefig(output_graph_path)
        logging.info(f"Gráfico de distribución de clases v3 guardado en '{output_graph_path}'")
        plt.close()

        logging.info("Distribución de clases ANTES (v2) y DESPUÉS (v3):")
        print(f"\n{'Clase':<20} | {'Instancias v2':<15} | {'Instancias v3':<15} | {'Cambio':<10}")
        print("-" * 65)
        for cid, count_v2 in sorted(conteo_v2.items()):
            count_v3 = conteo_v3.get(cid, count_v2)
            cambio = count_v3 - count_v2
            signo = '+' if cambio > 0 else ''
            print(f"{nombres_clases_v2[cid]:<20} | {count_v2:<15} | {count_v3:<15} | {signo}{cambio}")

    logging.info("--- Proceso de Generación de Dataset v3 Completado ---")


if __name__ == '__main__':
    generar_dataset_v3()
