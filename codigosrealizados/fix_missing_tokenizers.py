# -*- coding: utf-8 -*-
"""
fix_missing_tokenizers.py

Script de utilidad para EMAIRC VISIÓN.

Objetivo:
Reparar las carpetas de modelos entrenados con versiones antiguas del script
de entrenamiento que no guardaron el artefacto del tokenizer.

Este script encuentra las últimas ejecuciones para los modelos especificados,
verifica si les falta la carpeta 'tokenizer', y si es así, descarga el
tokenizer base ('distilbert-base-uncased') y lo guarda en la ubicación correcta.
"""

import logging
from pathlib import Path
from transformers import DistilBertTokenizer

# --- 1. CONFIGURACIÓN ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

# Apunta a la misma carpeta de 'output' que tu aplicación de inferencia.
OUTPUT_DIR = Path(r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2\output")
BASE_MODEL_NAME = 'distilbert-base-uncased'
MODELS_TO_CHECK = ["NatureTitle", "Part_of_Body_Title", "EventTitle"]


# --- 2. LÓGICA DEL SCRIPT ---

def find_latest_model_path(target_column: str) -> Path | None:
    """Encuentra la carpeta del entrenamiento más reciente para una columna objetivo."""
    target_dir = OUTPUT_DIR / target_column
    if not target_dir.exists():
        logging.warning(f"El directorio para '{target_column}' no existe. Omitiendo.")
        return None

    all_runs = sorted([d for d in target_dir.iterdir() if d.is_dir()], reverse=True)
    if not all_runs:
        logging.warning(f"No se encontraron ejecuciones de entrenamiento en '{target_dir}'. Omitiendo.")
        return None

    return all_runs[0]


def fix_tokenizer_if_missing(model_run_path: Path):
    """Verifica y guarda el tokenizer si no existe en la ruta de la ejecución."""
    tokenizer_path = model_run_path / "tokenizer"

    if tokenizer_path.exists():
        logging.info(f"✅ El tokenizer ya existe en: {model_run_path.relative_to(OUTPUT_DIR)}")
    else:
        logging.warning(
            f"🟡 Tokenizer no encontrado en: {model_run_path.relative_to(OUTPUT_DIR)}. Procediendo a crearlo...")
        try:
            # Descargar el tokenizer base que se usó para todos los entrenamientos
            tokenizer = DistilBertTokenizer.from_pretrained(BASE_MODEL_NAME)

            # Guardarlo en la carpeta que la aplicación de inferencia espera
            tokenizer.save_pretrained(tokenizer_path)
            logging.info(f"✅ Tokenizer guardado exitosamente en: {tokenizer_path}")
        except Exception as e:
            logging.error(f"❌ Falló la descarga o guardado del tokenizer para {model_run_path}. Error: {e}")


if __name__ == "__main__":
    logging.info("--- Iniciando script de reparación de tokenizers ---")
    if not OUTPUT_DIR.exists():
        logging.critical(f"El directorio base de output no existe: {OUTPUT_DIR}")
    else:
        for model_type in MODELS_TO_CHECK:
            logging.info(f"\n--- Verificando modelo: {model_type} ---")
            latest_run_path = find_latest_model_path(model_type)
            if latest_run_path:
                fix_tokenizer_if_missing(latest_run_path)

    logging.info("\n--- Proceso de reparación completado ---")

