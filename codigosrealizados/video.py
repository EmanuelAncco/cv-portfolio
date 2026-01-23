import cv2
import os
import glob
import logging
import re
import sys
import numpy as np
from PIL import Image as PILImage

# --- Configuración de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def natural_sort_key(s):
    """Ordenamiento natural para archivos (arita cfd.2 va antes que arita cfd.10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def imread_unicode(path):
    """
    Lee imágenes en Windows incluso si la ruta tiene tildes, espacios o caracteres especiales.
    """
    try:
        # Leemos el archivo como stream de bytes y lo decodificamos
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        logger.error(f"Error leyendo archivo: {path} | Causa: {e}")
        return None


def get_filtered_images(input_folder, extension, keyword_filter):
    """Busca y filtra imágenes."""
    # Verificación de existencia del directorio (Ingeniería Pesimista)
    if not os.path.exists(input_folder):
        logger.critical(f"EL DIRECTORIO NO EXISTE: {input_folder}")
        logger.critical("Verifica que la ruta sea correcta y esté escrita entre comillas simples o dobles.")
        sys.exit(1)

    search_pattern = os.path.join(input_folder, f"*.{extension}")
    all_images = glob.glob(search_pattern)

    if not all_images:
        logger.warning(f"No se encontraron archivos .{extension} en {input_folder}")
        return []

    # Filtrar imágenes
    if keyword_filter:
        images = [img for img in all_images if keyword_filter.lower() in os.path.basename(img).lower()]
        logger.info(f"Filtro '{keyword_filter}': {len(images)} imágenes encontradas de {len(all_images)} totales.")
    else:
        images = all_images

    images.sort(key=natural_sort_key)
    return images


def create_gif(input_folder, output_file, fps=10, extension="png", keyword_filter="", max_width=800):
    """Genera GIF optimizado."""
    logger.info(f"--- Iniciando generación de GIF ---")
    images = get_filtered_images(input_folder, extension, keyword_filter)

    if not images:
        return

    frames = []
    try:
        for idx, filename in enumerate(images):
            try:
                img = PILImage.open(filename)

                # Redimensionar para mantener el GIF ligero
                if img.width > max_width:
                    ratio = max_width / float(img.width)
                    new_height = int((float(img.height) * float(ratio)))
                    img = img.resize((max_width, new_height), PILImage.Resampling.LANCZOS)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                frames.append(img)

                if idx % 10 == 0:
                    print(f"Procesando frame {idx}/{len(images)}...", end='\r')
            except Exception as e:
                logger.warning(f"Error en frame {os.path.basename(filename)}: {e}")

        if frames:
            output_path = os.path.join(input_folder, output_file)
            duration_ms = int(1000 / fps)

            frames[0].save(
                output_path,
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=duration_ms,
                loop=0,
                optimize=True
            )
            logger.info(f"\n[ÉXITO] GIF guardado en: {output_path}")
        else:
            logger.error("No se procesaron frames válidos para el GIF.")

    except Exception as e:
        logger.critical(f"Fallo crítico al generar GIF: {e}")


def create_video(input_folder, output_file, fps=24, extension="png", keyword_filter=""):
    """Genera MP4."""
    logger.info(f"--- Iniciando generación de Video MP4 ---")
    images = get_filtered_images(input_folder, extension, keyword_filter)

    if not images:
        return

    first_frame = imread_unicode(images[0])
    if first_frame is None:
        logger.critical("No se pudo leer el primer frame para establecer dimensiones.")
        return

    height, width, _ = first_frame.shape
    size = (width, height)
    logger.info(f"Resolución de video: {width}x{height}")

    output_path = os.path.join(input_folder, output_file)

    try:
        # mp4v es seguro, avc1 es mejor si está disponible
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        if not out.isOpened():
            logger.error("No se pudo abrir el escritor de video. Verifica permisos o códecs.")
            return
    except Exception as e:
        logger.critical(f"Error inicializando codec: {e}")
        return

    processed = 0
    for filename in images:
        img = imread_unicode(filename)
        if img is None: continue

        if (img.shape[1], img.shape[0]) != size:
            img = cv2.resize(img, size)

        out.write(img)
        processed += 1
        print(f"Video: Frame {processed}/{len(images)}", end='\r')

    out.release()
    logger.info(f"\n[ÉXITO] Video guardado en: {output_path}")

    # Auto-reproducir video en Windows
    try:
        os.startfile(output_path)
    except Exception:
        pass


if __name__ == "__main__":
    # --- CONFIGURACIÓN CRÍTICA ---

    # 1. RUTA EXACTA DE TUS IMÁGENES
    # Usamos r"..." para que Windows no confunda las barras invertidas
    INPUT_FOLDER = r"C:\Users\Emanuel\Downloads\cfd"

    # 2. NOMBRE DEL ARCHIVO (FILTRO)
    # Basado en tu captura, tus archivos dicen "arita cfd". Ponemos "arita" para asegurar.
    FILTRO_NOMBRE = "arita"

    # 3. QUÉ GENERAR
    GENERAR_GIF = True  # Pon True si quieres GIF
    GENERAR_VIDEO = True  # Pon True si quieres MP4

    FPS = 10  # Velocidad (frames por segundo)

    # -----------------------------

    print(f"Trabajando en directorio: {INPUT_FOLDER}")

    if GENERAR_GIF:
        create_gif(
            INPUT_FOLDER,
            "animacion_cfd.gif",
            fps=FPS,
            keyword_filter=FILTRO_NOMBRE
        )

    if GENERAR_VIDEO:
        create_video(
            INPUT_FOLDER,
            "animacion_cfd.mp4",
            fps=FPS,
            keyword_filter=FILTRO_NOMBRE
        )

    # Abrir la carpeta al finalizar (Solo Windows)
    try:
        os.startfile(INPUT_FOLDER)
        print("\n📂 Carpeta abierta automáticamente.")
    except Exception:
        print(f"\n✅ Proceso terminado. Revisa la carpeta: {INPUT_FOLDER}")