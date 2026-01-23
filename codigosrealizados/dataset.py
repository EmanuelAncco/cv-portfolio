#!/usr/bin/env python3
"""
prepare_healthy_dataset_split.py – Script para Aumentar y Dividir Dataset de Hojas Healthy

Toma imágenes saludables originales, las aumenta para alcanzar un número objetivo,
y luego divide el conjunto resultante en carpetas de entrenamiento y validación.

Requiere:
- Python 3.10+
- TensorFlow 2.10+ (para ImageDataGenerator)
- Pillow (pip install Pillow)
- scipy (pip install scipy) # Añadido scipy a los requisitos
- shutil (librería estándar de Python)
"""

import os
import random
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image # Importar Image de Pillow
import time # Importar time para pausas pequeñas

# ─────────────────────────── Configuración de Rutas y Cantidades ────────────────────────────
# Ruta a la carpeta original que contiene las imágenes de hojas sanas de entrenamiento
# ¡ASEGÚRATE DE QUE ESTA RUTA APUNTE A TU CARPETA ORIGINAL 'Potato___healthy' DE ENTRENAMIENTO!
ORIGINAL_HEALTHY_TRAIN_DIR = r"D:\Python_proyectos_2025\Agricultura\archive\DATA\train\Potato___healthy" # ← Cambia esta ruta

# Ruta donde se guardarán las nuevas imágenes de hojas sanas para ENTRENAMIENTO
# Esto debería ser la carpeta 'Potato___healthy' dentro de tu carpeta de entrenamiento principal
NEW_HEALTHY_TRAIN_DIR = r"D:\Python_proyectos_2025\Agricultura\archive\DATA\train\Potato___healthy_new" # ← Define la nueva ruta de entrenamiento

# Ruta donde se guardarán las nuevas imágenes de hojas sanas para VALIDACIÓN
# Esto debería ser la carpeta 'Potato___healthy' dentro de tu carpeta de validación principal
NEW_HEALTHY_VAL_DIR = r"D:\Python_proyectos_2025\Agricultura\archive\DATA\val\Potato___healthy_new" # ← Define la nueva ruta de validación

# Número total de imágenes objetivo para la clase 'healthy' (entrenamiento + validación)
TARGET_TOTAL_COUNT = 1000

# Número de imágenes que se destinarán al conjunto de validación
VALIDATION_COUNT = 200

# Prefijo para los nombres de archivo de las imágenes generadas
GENERATED_IMAGE_PREFIX = "augmented_"

# ───────────────────────── Configuración de Aumento de Datos ─────────────────────────
# Define las transformaciones que se aplicarán.
# Ajusta estos parámetros según la cantidad y tipo de variación que necesites.
datagen = ImageDataGenerator(
    rotation_range=40,       # Rotar imágenes hasta 40 grados
    width_shift_range=0.2,   # Mover horizontalmente hasta el 20% del ancho
    height_shift_range=0.2,  # Mover verticalmente hasta el 20% de la altura
    shear_range=0.2,         # Aplicar cizallamiento
    zoom_range=0.2,          # Aplicar zoom
    horizontal_flip=True,    # Voltear horizontalmente
    vertical_flip=True,      # Voltear verticalmente (puede ser útil para hojas)
    brightness_range=[0.8, 1.2], # Ajustar brillo
    fill_mode='nearest'      # Estrategia para rellenar píxeles nuevos (ej. después de rotar)
)

# ─────────────────────────── Lógica de Preparación y División ────────────────────────────

def prepare_and_split_healthy_dataset(original_train_dir: str, new_train_dir: str, new_val_dir: str,
                                      target_total: int, val_count: int, prefix: str):
    """
    Prepara el dataset de imágenes saludables: aumenta si es necesario,
    y luego divide el conjunto total en carpetas de entrenamiento y validación.
    """
    if not os.path.exists(original_train_dir):
        print(f"Error: Directorio original de imágenes saludables de entrenamiento no encontrado: {original_train_dir}")
        return

    # Crear directorios de salida si no existen
    if not os.path.exists(new_train_dir):
        os.makedirs(new_train_dir)
        print(f"Directorio de nuevo entrenamiento creado: {new_train_dir}")
    if not os.path.exists(new_val_dir):
        os.makedirs(new_val_dir)
        print(f"Directorio de nueva validación creado: {new_val_dir}")

    # Contar imágenes originales
    original_images = [f for f in os.listdir(original_train_dir) if os.path.isfile(os.path.join(original_train_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    current_original_count = len(original_images)

    print(f"Imágenes saludables originales encontradas: {current_original_count}")

    # Lista para almacenar las rutas de todas las imágenes (originales + generadas)
    # Incluimos las rutas de las imágenes originales desde el principio
    all_images_to_split = [os.path.join(original_train_dir, img_name) for img_name in original_images]

    if current_original_count < target_total:
        images_to_generate = target_total - current_original_count
        print(f"Necesidad de generar imágenes adicionales: {images_to_generate}")

        # Guardamos temporalmente las imágenes generadas en un subdirectorio temporal
        temp_output_dir = os.path.join(original_train_dir, "temp_augmented")
        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)
            print(f"Directorio temporal para imágenes aumentadas creado: {temp_output_dir}")
        else:
             # Limpiar el directorio temporal si ya existe de una ejecución anterior
             print(f"Limpiando directorio temporal existente: {temp_output_dir}")
             for f in os.listdir(temp_output_dir):
                  os.remove(os.path.join(temp_output_dir, f))


        generated_count = 0
        # Generar imágenes aumentadas
        print("Generando imágenes aumentadas...")
        # Iterar para generar el número necesario de imágenes
        while generated_count < images_to_generate:
            # Seleccionar una imagen aleatoria original para transformar
            if not original_images:
                print("Advertencia: No hay imágenes originales para aumentar.")
                break

            img_name = random.choice(original_images)
            img_path = os.path.join(original_train_dir, img_name)

            try:
                # Cargar la imagen
                img = load_img(img_path)
                # Convertir a array numpy y añadir dimensión de batch (requerido por flow)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape) # Añade una dimensión: (1, altura, ancho, canales)

                # Usar el generador para crear un batch de imágenes transformadas
                # flow() genera imágenes indefinidamente, tomaremos solo una
                # Generar un nombre de archivo único para la imagen aumentada
                generated_img_name = f"{prefix}_{generated_count}_{int(time.time() * 1000)}.jpeg" # Nombre único basado en contador y timestamp
                generated_img_path = os.path.join(temp_output_dir, generated_img_name)

                # Usar flow con save_to_dir y save_prefix para guardar la imagen
                # Iterar una vez para generar y guardar una imagen
                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=temp_output_dir,
                                          save_prefix=prefix,
                                          save_format='jpeg'): # Puedes cambiar el formato si prefieres
                    # El archivo se guarda en save_to_dir con el nombre generado por flow
                    # No necesitamos el nombre generado por flow si ya construimos generated_img_path
                    # Sin embargo, flow *sí* guarda el archivo, solo necesitamos verificar su existencia
                    # y añadir la ruta generada a nuestra lista.

                    # all_images_to_split.append(generated_img_path) # No añadir aquí, flow guarda con otro nombre

                    # Esperar un momento y listar el directorio temporal para encontrar el archivo recién creado
                    time.sleep(0.01) # Pequeña pausa
                    temp_files_after_gen = os.listdir(temp_output_dir)
                    # Encontrar el archivo más reciente que coincida con el prefijo
                    recent_generated_files = [f for f in temp_files_after_gen if f.startswith(prefix)]
                    if recent_generated_files:
                         recent_generated_files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_output_dir, x)))
                         latest_generated_name = recent_generated_files[-1]
                         latest_generated_path = os.path.join(temp_output_dir, latest_generated_name)
                         # Añadir la ruta REAL del archivo generado por flow a la lista total
                         all_images_to_split.append(latest_generated_path)
                         generated_count += 1
                         print(f"Generada {generated_count}/{images_to_generate} imágenes adicionales...", end='\r')
                    else:
                         print("\nAdvertencia: No se encontró un archivo generado con el prefijo en el directorio temporal.")


                    break # Salir del bucle for después de generar 1 imagen

                if generated_count >= images_to_generate:
                     break # Salir del bucle while principal

            except Exception as e:
                print(f"\nError al procesar la imagen {img_path}: {e}")
                # Continuar con la siguiente imagen si hay un error

        print(f"\nGeneración de imágenes aumentadas completada. Total generadas: {generated_count}")

    else:
         print(f"El número de imágenes originales ({current_original_count}) ya alcanza o supera el total objetivo ({target_total}). No se necesita aumento.")


    print(f"Total de imágenes (originales + generadas) para la división: {len(all_images_to_split)}")

    # Asegurarse de que tenemos suficientes imágenes para la validación
    if len(all_images_to_split) < val_count:
        print(f"Error: El número total de imágenes ({len(all_images_to_split)}) es menor que el número requerido para validación ({val_count}).")
        print("Aumenta el TARGET_TOTAL_COUNT o reduce VALIDATION_COUNT.")
        # Limpiar el directorio temporal si se creó
        temp_output_dir = os.path.join(original_train_dir, "temp_augmented")
        if os.path.exists(temp_output_dir):
             shutil.rmtree(temp_output_dir)
        return

    # Mezclar aleatoriamente todas las imágenes
    random.shuffle(all_images_to_split)
    print("Imágenes mezcladas.")

    # Dividir en conjuntos de validación y entrenamiento
    val_images = all_images_to_split[:val_count]
    train_images = all_images_to_split[val_count:]

    print(f"Imágenes para validación: {len(val_images)}")
    print(f"Imágenes para entrenamiento: {len(train_images)}")

    # Mover imágenes a las nuevas carpetas
    print(f"Moviendo imágenes a {new_val_dir}...")
    for img_path in val_images:
        try:
            # Obtener solo el nombre del archivo
            img_name = os.path.basename(img_path)
            # Construir la ruta de destino
            dest_path = os.path.join(new_val_dir, img_name)
            # Mover (o copiar, si prefieres mantener los originales)
            shutil.move(img_path, dest_path)
        except Exception as e:
            print(f"\nError al mover la imagen {img_path} a validación: {e}")


    print(f"Moviendo imágenes a {new_train_dir}...")
    for img_path in train_images:
         try:
             # Obtener solo el nombre del archivo
             img_name = os.path.basename(img_path)
             # Construir la ruta de destino
             dest_path = os.path.join(new_train_dir, img_name)
             # Mover (o copiar)
             shutil.move(img_path, dest_path)
         except Exception as e:
            print(f"\nError al mover la imagen {img_path} a entrenamiento: {e}")

    # --- Limpiar el directorio temporal al final ---
    temp_output_dir = os.path.join(original_train_dir, "temp_augmented")
    if os.path.exists(temp_output_dir):
         print(f"Limpiando directorio temporal: {temp_output_dir}")
         shutil.rmtree(temp_output_dir)


    print("\nProceso de preparación y división completado.")
    # Contar los archivos en los directorios de destino para confirmar
    final_train_count = len([f for f in os.listdir(new_train_dir) if os.path.isfile(os.path.join(new_train_dir, f))])
    final_val_count = len([f for f in os.listdir(new_val_dir) if os.path.isfile(os.path.join(new_val_dir, f))])

    print(f"Total final en nueva carpeta de entrenamiento ({new_train_dir}): {final_train_count}")
    print(f"Total final en nueva carpeta de validación ({new_val_dir}): {final_val_count}")


# ─────────────────────────── Ejecución Principal ────────────────────────────
if __name__ == "__main__":
    prepare_and_split_healthy_dataset(
        ORIGINAL_HEALTHY_TRAIN_DIR,
        NEW_HEALTHY_TRAIN_DIR,
        NEW_HEALTHY_VAL_DIR,
        TARGET_TOTAL_COUNT,
        VALIDATION_COUNT,
        GENERATED_IMAGE_PREFIX
    )
