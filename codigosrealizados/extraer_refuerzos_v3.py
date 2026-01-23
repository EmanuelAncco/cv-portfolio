# extraer_refuerzos_v3.py
import os
import shutil
import logging
from collections import defaultdict

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("extraccion_refuerzos.log", mode='w'),
        logging.StreamHandler()
    ]
)


def extraer_datos_de_refuerzo():
    """
    Este script lee el dataset original y extrae únicamente las imágenes que contienen
    las clases necesarias para reforzar nuestro Modelo v2, creando un nuevo dataset
    listo para ser subido a Roboflow.
    """
    # --- CONFIGURACIÓN ---
    base_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0'
    directorio_origen = os.path.join(base_path, 'dataset_original')
    directorio_destino = os.path.join(base_path, 'dataset_refuerzo_v3')

    # Clases del dataset original que queremos extraer para reforzar el Modelo v2.
    # IDs basados en el data.yaml original.
    clases_de_refuerzo = {
        1: 'Gloves',
        2: 'Goggles',  # Servirá para 'Shades' y 'No-Specs'
        5: 'Mask',
        6: 'NO-Gloves',
        7: 'NO-Goggles',
        9: 'NO-Mask',
    }

    logging.info("Iniciando extracción de datos de refuerzo para el Modelo v3...")

    # --- PROCESAMIENTO ---
    stats = defaultdict(int)

    for subdirectorio in ['train', 'valid', 'test']:
        logging.info(f"--- Procesando subdirectorio de origen: {subdirectorio} ---")

        origen_labels = os.path.join(directorio_origen, subdirectorio, 'labels')
        origen_images = os.path.join(directorio_origen, subdirectorio, 'images')

        # Para simplificar, guardaremos todo en una única carpeta de destino.
        # Roboflow se encargará de dividirlo de nuevo al subirlo.
        destino_labels = os.path.join(directorio_destino, 'labels')
        destino_images = os.path.join(directorio_destino, 'images')

        os.makedirs(destino_labels, exist_ok=True)
        os.makedirs(destino_images, exist_ok=True)

        if not os.path.isdir(origen_labels):
            logging.warning(f"Directorio de etiquetas de origen no existe: {origen_labels}. Saltando.")
            continue

        for filename in os.listdir(origen_labels):
            if not filename.endswith('.txt'):
                continue

            contiene_clase_refuerzo = False
            # Primero, revisamos si el archivo contiene alguna de las clases que nos interesan
            try:
                with open(os.path.join(origen_labels, filename), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id_original = int(parts[0])
                            if class_id_original in clases_de_refuerzo:
                                contiene_clase_refuerzo = True
                                break  # Encontramos una, no necesitamos seguir leyendo
            except Exception as e:
                logging.error(f"No se pudo leer el archivo {filename}: {e}")
                continue

            # Si el archivo contiene al menos una clase de refuerzo, lo copiamos
            if contiene_clase_refuerzo:
                imagen_origen_path = os.path.join(origen_images, filename.replace('.txt', '.jpg'))

                if not os.path.exists(imagen_origen_path):
                    logging.warning(
                        f"Etiqueta encontrada para '{filename}', pero la imagen correspondiente no existe. Omitiendo.")
                    stats['imagenes_faltantes'] += 1
                    continue

                try:
                    # Copiamos tanto la imagen como el archivo de etiquetas original
                    shutil.copy2(imagen_origen_path, destino_images)
                    shutil.copy2(os.path.join(origen_labels, filename), destino_labels)
                    stats['archivos_extraidos_ok'] += 1
                except Exception as e:
                    logging.error(f"Error al copiar {filename} o su imagen: {e}")
                    stats['errores_copia'] += 1

    logging.info("--- Proceso de Extracción Completado ---")
    for key, value in stats.items():
        logging.info(f"  - {key.replace('_', ' ').capitalize()}: {value}")
    logging.info(f"Tu nuevo dataset de refuerzo está listo en la carpeta: '{directorio_destino}'")
    logging.info(
        "El siguiente paso es subir el contenido de esta carpeta a un nuevo proyecto en Roboflow para fusionarlo con el dataset v2.")


if __name__ == '__main__':
    extraer_datos_de_refuerzo()
