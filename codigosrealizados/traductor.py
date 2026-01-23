# -*- coding: utf-8 -*-

"""
SCRIPT DE PROCESAMIENTO DE VIDEO A SUBTÍTULOS (HERRAMIENTA DE LÍNEA DE COMANDOS)

Versión: 3.4
Autor: Tu Consultor de IA/ML

Descripción:
Esta versión implementa una arquitectura de IA 100% local y robusta.
- v3.4: Se añade el argumento `--initial_language` para forzar el idioma de
  origen en Whisper, mejorando la precisión en audios con ruido o música.
- v3.3: Se añade un sistema de checkpointing.
- v3.2: Se añade `use_safetensors=True` a la carga del modelo de traducción.

Principios de Diseño Aplicados:
- Resiliencia: El script es reiniciable y no repite trabajo.
- Precisión: Permite guiar al modelo de transcripción para mejores resultados.
- Reproducibilidad: El entorno de Python se define en `requirements.txt`.
- Seguridad: Se prioriza la carga de modelos en formato `safetensors`.

Requisitos:
- Python 3.8+
- FFmpeg: Instalado en el sistema y accesible desde el PATH.
- Librerías de Python: Instalar usando `pip install -r requirements.txt`

Uso desde la Terminal:
python traductor.py --video_path "C:\ruta\a\tu\video.mp4" --model large-v3 --lang es --initial_language ja
"""

import os
import subprocess
import logging
import datetime
from pathlib import Path
import sys
import time
import argparse
import torch
import json  # Necesario para guardar y cargar los checkpoints

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- IMPORTACIONES DE MODELOS DE IA ---
try:
    import whisper
    from transformers import MarianMTModel, MarianTokenizer
except ImportError as e:
    logging.critical(f"Error importando librerías de IA: {e}")
    logging.critical(
        "El entorno parece estar mal configurado. Asegúrate de haber ejecutado: pip install -r requirements.txt")
    sys.exit("Fallo crítico: Dependencias de IA no encontradas.")


def setup_file_logging(log_dir: Path):
    log_filename = log_dir / f"execution_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genera subtítulos traducidos para un archivo de video usando modelos locales.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--video_path", type=Path, required=True, help="Ruta completa al archivo de video.")
    parser.add_argument("--model", type=str, default="medium", choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                        help="Modelo de Whisper a utilizar. Para máxima precisión, usa 'large-v3'.")
    parser.add_argument("--lang", type=str, default="es",
                        help="Código de dos letras del idioma de destino (ej: 'es' para español).")
    # --- CAMBIO CLAVE (v3.4) ---
    parser.add_argument(
        "--initial_language",
        type=str,
        default=None,  # Por defecto, auto-detectar
        help="Código de dos letras del idioma del video (ej: 'ja' para japonés). Ayuda a mejorar la precisión inicial."
    )
    return parser.parse_args()


def create_output_directory(base_path: Path) -> Path:
    output_dir_name = f"{base_path.stem}_translation_output"
    output_dir = base_path.parent / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directorio de resultados en uso: {output_dir}")
    return output_dir


def extract_audio(video_path: Path, output_audio_path: Path) -> bool:
    command = ['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y',
               str(output_audio_path)]
    logging.info(f"Extrayendo audio de '{video_path.name}'...")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        logging.debug(f"FFmpeg stderr: {process.stderr}")
        logging.info("Extracción de audio completada.")
        return True
    except FileNotFoundError:
        logging.critical("Error: `ffmpeg` no encontrado. Asegúrate de que esté instalado y en el PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg falló al extraer el audio. Error: {e.stderr}")
        return False


def transcribe_audio(audio_path: Path, model_name: str, language: str) -> dict:
    logging.info(f"Cargando el modelo Whisper '{model_name}'...")
    try:
        model = whisper.load_model(model_name)
        logging.info("Modelo Whisper cargado. Iniciando transcripción...")
        # --- CAMBIO CLAVE (v3.4) ---
        # Pasamos el idioma inicial al modelo para guiarlo.
        result = model.transcribe(str(audio_path), verbose=True, fp16=torch.cuda.is_available(), language=language)
        logging.info("Transcripción completada.")
        return result
    except Exception as e:
        logging.error(f"Error durante la transcripción: {e}", exc_info=True)
        return None


def translate_segments_local(transcription_result: dict, target_lang: str, device) -> list:
    original_lang = transcription_result['language']
    model_name = f'Helsinki-NLP/opus-mt-{original_lang}-{target_lang}'

    logging.info(f"Cargando modelo de traducción local: {model_name}")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name, use_safetensors=True).to(device)
        logging.info(f"Modelo de traducción cargado en el dispositivo: {device}")
    except Exception as e:
        logging.error(f"No se pudo cargar el modelo de traducción '{model_name}'. Error: {e}", exc_info=True)
        return None

    segments = transcription_result['segments']
    translated_segments = []
    texts_to_translate = [segment['text'] for segment in segments]

    logging.info(f"Iniciando traducción local de {len(segments)} segmentos...")
    batch_size = 16
    for i in range(0, len(texts_to_translate), batch_size):
        batch_texts = texts_to_translate[i:i + batch_size]
        try:
            tokenized_text = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
            translated_tokens = model.generate(**tokenized_text)
            translated_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            for j, translated_text in enumerate(translated_batch):
                original_segment_index = i + j
                translated_segment = segments[original_segment_index].copy()
                translated_segment['text'] = translated_text
                translated_segments.append(translated_segment)

        except Exception as e:
            logging.warning(
                f"No se pudo traducir el lote que comienza en el segmento {i + 1}. Usando texto original. Error: {e}")
            for k in range(len(batch_texts)):
                translated_segments.append(segments[i + k])

        logging.info(
            f"Progreso de traducción: {min(i + batch_size, len(segments))}/{len(segments)} segmentos procesados.")

    logging.info("Traducción local finalizada.")
    return translated_segments


def format_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds_rem = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds_rem:02},{milliseconds:03}"


def generate_srt(segments: list, srt_path: Path):
    logging.info(f"Generando archivo de subtítulos en: {srt_path}")
    try:
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments):
                f.write(f"{i + 1}\n")
                f.write(f"{format_time(segment['start'])} --> {format_time(segment['end'])}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        logging.info("Archivo SRT generado con éxito.")
    except IOError as e:
        logging.error(f"No se pudo escribir el archivo SRT. Error: {e}")


def main():
    args = parse_arguments()
    if not args.video_path.exists():
        logging.critical(f"El archivo de video no existe: {args.video_path}")
        sys.exit("Ejecución abortada.")

    start_time = time.time()
    output_dir = create_output_directory(args.video_path)
    setup_file_logging(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("--- INICIO DE LA PIPELINE DE TRADUCCIÓN DE VIDEO (v3.4 - Resiliente) ---")
    logging.info(f"Dispositivo de cómputo detectado: {device.upper()}")

    audio_path = output_dir / f"{args.video_path.stem}.wav"
    transcription_checkpoint_path = output_dir / f"{args.video_path.stem}_transcription.json"

    if not audio_path.exists():
        if not extract_audio(args.video_path, audio_path):
            sys.exit("Pipeline detenida: fallo en extracción de audio.")
    else:
        logging.info("Archivo de audio ya existe, saltando extracción.")

    if not transcription_checkpoint_path.exists():
        logging.info("No se encontró checkpoint de transcripción. Iniciando transcripción...")
        # --- CAMBIO CLAVE (v3.4) ---
        # Pasamos el idioma inicial desde los argumentos al a función de transcripción.
        transcription_result = transcribe_audio(audio_path, args.model, args.initial_language)
        if not transcription_result:
            sys.exit("Pipeline detenida: fallo en transcripción.")

        logging.info(f"Guardando checkpoint de transcripción en: {transcription_checkpoint_path}")
        with open(transcription_checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(transcription_result, f, ensure_ascii=False, indent=4)
    else:
        logging.info(f"Checkpoint de transcripción encontrado. Cargando desde: {transcription_checkpoint_path}")
        with open(transcription_checkpoint_path, 'r', encoding='utf-8') as f:
            transcription_result = json.load(f)
        logging.info("Checkpoint cargado con éxito.")

    translated_segments = translate_segments_local(transcription_result, args.lang, device)
    if not translated_segments:
        sys.exit("Pipeline detenida: fallo en traducción.")

    srt_filename = output_dir / f"{args.video_path.stem}_{args.lang}.srt"
    generate_srt(translated_segments, srt_filename)

    total_time = time.time() - start_time
    logging.info("--- FIN DE LA PIPELINE ---")
    logging.info(f"Proceso completado en {total_time / 60:.2f} minutos.")
    logging.info(f"El archivo de subtítulos está listo en: {srt_filename}")


if __name__ == '__main__':
    main()
