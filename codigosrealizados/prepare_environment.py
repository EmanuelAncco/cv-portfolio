# -*- coding: utf-8 -*-
"""
Script de Preparación del Entorno para EMARC VISION - Motor de Riesgos

Objetivo:
Crear la estructura de directorios necesaria para el proyecto de entrenamiento
del modelo de predicción de riesgos en una ubicación base específica. Esto
asegura un entorno de trabajo limpio, organizado y reproducible.

Uso:
Ejecutar este script una vez al inicio del proyecto:
    python prepare_environment.py
"""

import os
import logging

# --- Configuración del Logging ---
# Usamos logging para informar al usuario sobre las acciones realizadas.
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# --- Definición de la Ruta Base y Estructura ---
# Se define la ruta raíz específica donde se creará toda la estructura del proyecto.
# Usamos un "raw string" (r"...") para evitar problemas con las barras invertidas en Windows.
BASE_DIRECTORY = r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2"
DIRECTORIES = ["data", "output", "logs"]


def create_project_structure():
    """
    Crea la estructura de carpetas definida en la lista DIRECTORIES dentro
    de la ruta BASE_DIRECTORY.
    """
    logging.info(f"Asegurando que el directorio base exista: '{BASE_DIRECTORY}'")
    try:
        # Primero, nos aseguramos de que el directorio base exista.
        os.makedirs(BASE_DIRECTORY, exist_ok=True)

        logging.info("Iniciando la creación de la estructura de subdirectorios del proyecto...")

        for directory in DIRECTORIES:
            # Construimos la ruta completa uniendo la base con el subdirectorio.
            # os.path.join es la forma correcta de hacerlo para que funcione en cualquier SO.
            full_path = os.path.join(BASE_DIRECTORY, directory)

            # os.makedirs con exist_ok=True es la forma idiomática y segura de crear
            # directorios. No lanzará un error si la carpeta ya existe.
            os.makedirs(full_path, exist_ok=True)
            logging.info(f"Directorio '{full_path}' asegurado.")

        # Crear archivos .gitkeep en las carpetas para que puedan ser rastreadas por Git.
        data_path = os.path.join(BASE_DIRECTORY, "data")
        logs_path = os.path.join(BASE_DIRECTORY, "logs")

        with open(os.path.join(data_path, ".gitkeep"), "w") as f:
            pass
        with open(os.path.join(logs_path, ".gitkeep"), "w") as f:
            pass

        logging.info("-" * 50)
        logging.info("¡Estructura creada con éxito!")
        logging.info(f"Por favor, coloca tu archivo 'fatalities_2015_to_2024.csv' dentro de la carpeta:")
        logging.info(f"'{os.path.join(BASE_DIRECTORY, 'data')}'")
        logging.info("-" * 50)

    except OSError as e:
        # Capturamos posibles errores del sistema operativo, como problemas de permisos.
        logging.error(f"Error al crear la estructura de directorios: {e}")
        logging.error(f"Por favor, verifica los permisos de escritura en '{BASE_DIRECTORY}'.")


if __name__ == "__main__":
    create_project_structure()
