# -*- coding: utf-8 -*-
"""
Script de Aumentación Híbrida v1.6 (Control de Tasa y Cliente Async)

Objetivo:
Automatizar la aumentación de datos de forma robusta, evitando errores de
rate limiting y usando un cliente HTTP asíncrono.

Mejoras sobre la v1.5:
- Se reemplaza la librería `requests` (síncrona) por `aiohttp` (asíncrona)
  para un correcto manejo dentro de un bucle `asyncio`.
- Se introduce una clase `RateLimiter` para controlar proactivamente el
  número de llamadas a la API por minuto, evitando el error `429`.
- Se ajusta la lógica de reintento para que funcione mejor con el control de tasa.
"""

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
import argparse
import logging
import os
from datetime import datetime
from tqdm import tqdm
import time
import json
import random
import asyncio
import aiohttp  # _# NUEVO_: Importamos la librería asíncrona para HTTP


# --- CONFIGURACIÓN ---
# (La función setup_parser y setup_logging se mantienen igual)
def setup_parser():
    """Configura el parser de argumentos para la línea de comandos."""
    BASE_PROJECT_DIR = r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2"

    parser = argparse.ArgumentParser(description="Script de aumentación de datos para clases minoritarias.")

    parser.add_argument(
        'target_column',
        type=str,
        choices=['NatureTitle', 'Part_of_Body_Title', 'EventTitle'],
        help="La columna objetivo que se analizará para la aumentación."
    )
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(BASE_PROJECT_DIR, 'data/fatalities_2015_to_2024.csv'),
                        help="Ruta al archivo CSV de datos original.")
    parser.add_argument('--output_file', type=str,
                        default=os.path.join(BASE_PROJECT_DIR, 'data/fatalities_augmented.csv'),
                        help="Ruta para guardar el nuevo archivo CSV aumentado.")
    parser.add_argument('--progress_file', type=str,
                        default=os.path.join(BASE_PROJECT_DIR, 'data/augmented_progress.csv'),
                        help="Ruta para guardar el progreso intermedio.")
    parser.add_argument('--log_dir', type=str, default=os.path.join(BASE_PROJECT_DIR, 'logs'),
                        help="Directorio para guardar los archivos de log.")
    parser.add_argument('--min_class_support', type=int, default=100,
                        help="Número mínimo de ejemplos que una clase debe tener para no ser aumentada.")
    parser.add_argument('--back_trans_count', type=int, default=2,
                        help="Número de variaciones a generar por Back Translation para cada muestra.")
    parser.add_argument('--llm_variations', type=int, default=3,
                        help="Número de variaciones a generar por el LLM para cada muestra.")
    # _# NUEVO_: Argumento para controlar la tasa de llamadas a la API
    parser.add_argument('--api_rpm', type=int, default=50,
                        help="Máximas solicitudes por minuto (RPM) a la API del LLM para evitar el rate limiting.")

    return parser


def setup_logging(log_dir, run_timestamp):
    """Configura el logging para guardar en archivo y mostrar en consola."""
    log_filename = os.path.join(log_dir, f"augmentation_{run_timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )


# --- LÓGICA DE AUMENTACIÓN ---

class BackTranslator:
    # (Esta clase se mantiene igual)
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.pivot_languages = {
            'es': {'en_to_piv': 'Helsinki-NLP/opus-mt-en-es', 'piv_to_en': 'Helsinki-NLP/opus-mt-es-en'},
            'de': {'en_to_piv': 'Helsinki-NLP/opus-mt-en-de', 'piv_to_en': 'Helsinki-NLP/opus-mt-de-en'}
        }
        self._load_models()

    def _load_models(self):
        logging.info("Cargando modelos de traducción...")
        for lang, paths in tqdm(self.pivot_languages.items(), desc="Cargando modelos"):
            try:
                self.models[lang] = {
                    'en_to_piv_tok': MarianTokenizer.from_pretrained(paths['en_to_piv']),
                    'en_to_piv_mod': MarianMTModel.from_pretrained(paths['en_to_piv'], use_safetensors=True).to(
                        self.device),
                    'piv_to_en_tok': MarianTokenizer.from_pretrained(paths['piv_to_en']),
                    'piv_to_en_mod': MarianMTModel.from_pretrained(paths['piv_to_en'], use_safetensors=True).to(
                        self.device)
                }
            except Exception as e:
                logging.error(f"No se pudo cargar el modelo para el idioma '{lang}'. Error: {e}")

    def translate(self, text, model, tokenizer):
        tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt').to(self.device)
        translated_tokens = model.generate(**tokenized_text)
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    def augment(self, text, num_variations):
        augmented_texts = []
        available_langs = list(self.models.keys())
        if not available_langs: return []
        for i in range(num_variations):
            try:
                lang = available_langs[i % len(available_langs)]
                lang_models = self.models[lang]
                pivot_text = self.translate(text, lang_models['en_to_piv_mod'], lang_models['en_to_piv_tok'])
                back_translated_text = self.translate(pivot_text, lang_models['piv_to_en_mod'],
                                                      lang_models['piv_to_en_tok'])
                augmented_texts.append(back_translated_text)
            except Exception as e:
                logging.error(f"Error durante back translation para el texto: '{text[:50]}...'. Error: {e}")
        return augmented_texts


# _# NUEVO_: Clase para controlar el ritmo de las llamadas a la API
class RateLimiter:
    """Gestiona el ritmo de las solicitudes para no exceder un límite de RPM."""

    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def wait(self):
        """Espera si es necesario para mantener la tasa de solicitudes."""
        while len(self.requests) >= self.requests_per_minute:
            # Elimina las marcas de tiempo que ya tienen más de un minuto
            now = time.monotonic()
            self.requests = [r for r in self.requests if r > now - 60]
            if len(self.requests) >= self.requests_per_minute:
                # Calcula cuánto esperar para que la solicitud más antigua "expire"
                wait_time = self.requests[0] - (now - 60)
                await asyncio.sleep(wait_time)

        self.requests.append(time.monotonic())


# _# MODIFICADO_: Función de parafraseo usando aiohttp y sesión
async def paraphrase_with_llm(session, text, num_variations, api_key, rate_limiter):
    if num_variations == 0 or not api_key: return []

    await rate_limiter.wait()  # _# NUEVO_: Espera a que el limitador de tasa nos dé luz verde

    prompt = (
        "Act as an expert in industrial safety and occupational risk prevention. "
        f"Rewrite the following accident narrative in {num_variations} distinct versions. "
        "Use technical terminology, vary the sentence structure, and maintain all key facts. "
        "Return ONLY the rewritten narratives, separated by the delimiter '|||'.\n\n"
        f"Original Narrative: \"{text}\""
    )
    chatHistory = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chatHistory}
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    max_retries, delay = 5, 2.0  # Aumentamos el delay inicial

    for attempt in range(max_retries):
        try:
            async with session.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'}) as response:
                response.raise_for_status()
                result = await response.json()
                if result.get("candidates"):
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    return [p.strip() for p in content.split('|||')]
                else:
                    logging.warning(f"Respuesta inesperada de la API: {result}")
                    return []
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                logging.warning(f"Límite de velocidad alcanzado (HTTP 429). El RateLimiter debería haberlo prevenido. "
                                f"Esperando {delay}s antes de reintentar...")
            else:
                logging.warning(f"Intento {attempt + 1} fallido (HTTP Error {e.status}): {e.message}. Reintentando...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60) + random.uniform(0, 1)  # Aumentamos el backoff máximo
        except Exception as e:
            logging.warning(f"Intento {attempt + 1} fallido: {e}. Reintentando...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60) + random.uniform(0, 1)

    logging.error(
        f"No se pudo obtener una paráfrasis de la API para: '{text[:50]}...' después de {max_retries} intentos.")
    return []


# --- SCRIPT PRINCIPAL ---
async def main():
    parser = setup_parser()
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir, run_timestamp)

    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        logging.warning("Variable de entorno 'GEMINI_API_KEY' no configurada. Se omitirá el parafraseo con LLM.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- INICIANDO SCRIPT DE AUMENTACIÓN HÍBRIDA (v1.6) ---")
    logging.info(f"Usando dispositivo: {device}")
    logging.info(f"Parámetros: {args}")

    try:
        try:
            logging.info("Intentando leer CSV con delimitador de punto y coma (;)...")
            df = pd.read_csv(args.input_file, encoding='utf-8', delimiter=';')
            if len(df.columns) <= 1: raise ValueError("Delimitador incorrecto")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
            logging.warning("La lectura con ';' falló. Reintentando con delimitador de coma (,)...")
            df = pd.read_csv(args.input_file, encoding='latin1', delimiter=',')

        logging.info(f"Dataset original cargado con {len(df)} filas.")
        value_counts = df[args.target_column].value_counts()
        minority_classes = value_counts[value_counts < args.min_class_support].index
        df_to_augment = df[df[args.target_column].isin(minority_classes)].copy()

        processed_indices = set()
        if os.path.exists(args.progress_file):
            logging.info(f"Archivo de progreso encontrado en '{args.progress_file}'. Cargando...")
            df_progress = pd.read_csv(args.progress_file, sep=';')
            if 'original_index' in df_progress.columns:
                processed_indices = set(df_progress['original_index'].unique())
                logging.info(f"{len(processed_indices)} narrativas ya han sido procesadas y serán omitidas.")
            else:
                logging.warning(
                    "El archivo de progreso no tiene la columna 'original_index'. Se procesará todo de nuevo.")

        df_to_augment['original_index'] = df_to_augment.index
        df_remaining = df_to_augment[~df_to_augment['original_index'].isin(processed_indices)]

        logging.info(f"Se han identificado {len(minority_classes)} clases minoritarias.")
        logging.info(f"Total a aumentar: {len(df_to_augment)}. Restantes por procesar: {len(df_remaining)}.")

        if len(df_remaining) == 0:
            logging.info("No hay nuevas narrativas que procesar. Finalizando y consolidando el archivo final.")
        else:
            translator = BackTranslator(device)
            # _# NUEVO_: Inicializamos el limitador de tasa y la sesión de aiohttp
            rate_limiter = RateLimiter(args.api_rpm)
            async with aiohttp.ClientSession() as session:
                with open(args.progress_file, 'a', newline='', encoding='utf-8') as f:
                    if not processed_indices:
                        header_df = pd.DataFrame(columns=list(df.columns) + ['original_index'])
                        header_df.to_csv(f, index=False, sep=';')

                    for index, row in tqdm(df_remaining.iterrows(), total=len(df_remaining),
                                           desc="Aumentando narrativas"):
                        original_narrative = row['FinalNarrative']
                        if not isinstance(original_narrative, str): continue

                        all_variations = []

                        bt_variations = translator.augment(original_narrative, args.back_trans_count)
                        all_variations.extend(bt_variations)

                        # _# MODIFICADO_: Pasamos la sesión y el limitador a la función
                        llm_variations = await paraphrase_with_llm(session, original_narrative, args.llm_variations,
                                                                   gemini_api_key, rate_limiter)
                        all_variations.extend(llm_variations)

                        if all_variations:
                            new_rows_df = pd.DataFrame([row.to_dict()] * len(all_variations))
                            new_rows_df['FinalNarrative'] = all_variations
                            new_rows_df['original_index'] = index
                            new_rows_df.to_csv(f, header=False, index=False, sep=';')

                        # Ya no necesitamos el time.sleep(1) aquí, el RateLimiter lo gestiona.

        logging.info("Consolidando el dataset final...")
        if os.path.exists(args.progress_file):
            df_progress_final = pd.read_csv(args.progress_file, sep=';')
            if 'original_index' in df_progress_final.columns:
                df_progress_final.drop(columns=['original_index'], inplace=True)
            df_final = pd.concat([df, df_progress_final], ignore_index=True)
        else:
            df_final = df

        df_final.to_csv(args.output_file, index=False, sep=';', encoding='utf-8')
        logging.info("--- PROCESO DE AUMENTACIÓN COMPLETADO ---")
        logging.info(f"Filas originales: {len(df)}")
        if os.path.exists(args.progress_file):
            logging.info(f"Total de filas aumentadas (desde progreso): {len(df_progress_final)}")
        logging.info(f"Total de filas en el nuevo dataset: {len(df_final)}")
        logging.info(f"Dataset aumentado guardado en: {args.output_file}")

    except FileNotFoundError:
        logging.error(f"¡Archivo de entrada no encontrado en '{args.input_file}'!")
    except Exception as e:
        logging.critical(f"Ha ocurrido un error fatal durante el proceso: {e}", exc_info=True)


if __name__ == "__main__":
    # La lógica para manejar el nest_asyncio se mantiene, es una buena práctica.
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot run loop while another loop is running" in str(e):
            import nest_asyncio

            nest_asyncio.apply()
            asyncio.run(main())
        else:
            raise