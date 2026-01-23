# consolidate_results.py (v2 - con lectura robusta)
import pandas as pd
import logging
import os

# --- Configuración ---
BASE_PROJECT_DIR = r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2"
ORIGINAL_FILE = os.path.join(BASE_PROJECT_DIR, 'data/fatalities_2015_to_2024.csv')
PROGRESS_FILE = os.path.join(BASE_PROJECT_DIR, 'data/augmented_progress.csv')
FINAL_OUTPUT_FILE = os.path.join(BASE_PROJECT_DIR, 'data/fatalities_augmented_FINAL.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

try:
    logging.info("Iniciando la consolidación final (v2)...")

    # --- LÓGICA DE LECTURA ROBUSTA (LA CORRECCIÓN) ---
    # Intenta leer el archivo original de la forma más robusta posible.
    df_original = None
    try:
        # Intenta leer con punto y coma, que es menos propenso a errores si hay comas en el texto.
        logging.info("Intentando leer el archivo original con delimitador ';'...")
        df_original = pd.read_csv(ORIGINAL_FILE, encoding='utf-8', delimiter=';')
        if len(df_original.columns) <= 1:
            raise ValueError("Delimitador incorrecto, solo se encontró una columna.")
        logging.info("Archivo original leído con éxito usando ';'.")
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError, FileNotFoundError):
        logging.warning("La lectura con ';' falló. Reintentando con delimitador ',' y encoding 'latin1'...")
        # Si falla, usa la coma como delimitador (el caso que causó el error).
        df_original = pd.read_csv(ORIGINAL_FILE, encoding='latin1', delimiter=',')
        logging.info("Archivo original leído con éxito usando ','.")

    logging.info(f"Cargadas {len(df_original)} filas del archivo original.")

    # 2. Cargar todas las filas aumentadas desde el archivo de progreso
    # Este archivo lo generamos nosotros, así que sabemos que está bien formateado con ';'.
    df_progress = pd.read_csv(PROGRESS_FILE, sep=';')
    if 'original_index' in df_progress.columns:
        df_progress.drop(columns=['original_index'], inplace=True)
    logging.info(f"Cargadas {len(df_progress)} filas del archivo de progreso.")

    # 3. Concatenar ambos dataframes
    df_final = pd.concat([df_original, df_progress], ignore_index=True)
    logging.info(f"Concatenación completa. Total de filas: {len(df_final)}.")

    # 4. Guardar en el archivo final
    df_final.to_csv(FINAL_OUTPUT_FILE, index=False, sep=';', encoding='utf-8-sig')
    logging.info(f"--- ¡CONSOLIDACIÓN COMPLETADA! ---")
    logging.info(f"Dataset final guardado en: {FINAL_OUTPUT_FILE}")

except Exception as e:
    logging.critical(f"Ha ocurrido un error inesperado: {e}", exc_info=True)
