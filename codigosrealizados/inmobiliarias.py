import httpx  # Usamos httpx por ser moderno y compatible con async (más rápido)
import re
import json
import logging
from http.cookies import SimpleCookie

# --- Configuración del Logger (Estándar de tu proyecto) ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("scraper_v5_regex.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- Configuración de la Misión ---
URL_PLANO = "https://sistema.casadoradainmobiliaria.com/ver_plano_publico.php?plano_id=14"
OUTPUT_FILE = "inventario_lotes_v5.json"

# --- CONFIGURACIÓN DE REGEX (TU TAREA) ---
#
# TU TAREA:
# 1. Ve a la URL en Chrome.
# 2. Haz clic derecho -> "Ver código fuente de la página" (o Ctrl+U).
# 3. Busca (Ctrl+F) por palabras clave como: "var", "lotes", "json", "L.geoJSON".
# 4. Encontrarás una variable de JavaScript que contiene un array [...].
#    Ejemplo: var mis_lotes = [{"id":...}, ...];
#
# 5. Pon el nombre de esa variable aquí:
REGEX_VARIABLE_NAME = "poligonos"  # CAMBIA ESTO. (Ej: "data_plano", "lotes_json", etc.)

# Esta RegEx buscará: var [TU_VARIABLE] = ([...]);
# Y extraerá el contenido del array/objeto.
# re.DOTALL hace que '.' coincida también con saltos de línea.
REGEX_PATTERN = rf"var {REGEX_VARIABLE_NAME} \s* = \s* (\[.*?\]);"


async def fetch_lot_data_v5_regex():
    """
    Inicia un cliente HTTP ligero (httpx) para descargar el HTML
    y extraer el JSON de datos incrustado usando Expresiones Regulares.
    """
    logger.info(f"Iniciando el proceso de extracción (v5 - RegEx Ligero)...")
    logger.info(f"Objetivo: {URL_PLANO}")

    # Usamos un 'try-except' para todo el proceso (Ingeniería Pesimista)
    try:
        async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
            # Añadimos headers para simular un navegador real
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "es-ES,es;q=0.9",
            }

            logger.info("Descargando contenido HTML...")
            response = await client.get(URL_PLANO, headers=headers)

            # Verificar si la descarga fue exitosa
            if response.status_code != 200:
                logger.error(f"Error al descargar la página. Código de estado: {response.status_code}")
                logger.error(f"Respuesta: {response.text[:200]}...")  # Muestra un pedazo de la respuesta
                return

            html_content = response.text
            logger.info("Contenido HTML descargado. Buscando patrón RegEx...")

            # Compilar y buscar la RegEx
            # re.DOTALL es CRUCIAL para que '.*?' capture a través de múltiples líneas.
            regex_compilada = re.compile(REGEX_PATTERN, re.DOTALL | re.IGNORECASE)
            match = regex_compilada.search(html_content)

            if not match:
                logger.error(f"¡Patrón RegEx NO encontrado!")
                logger.error(f"No se pudo encontrar la variable '{REGEX_VARIABLE_NAME}' en el HTML.")
                logger.error("--- INSTRUCCIONES ---")
                logger.error("1. Abre la URL en tu navegador.")
                logger.error("2. Mira el 'Código Fuente' (Ctrl+U).")
                logger.error("3. Busca la variable que contiene el JSON de los lotes (ej. 'var data_lotes = [...]').")
                logger.error(
                    "4. Actualiza la variable 'REGEX_VARIABLE_NAME' en este script con el nombre que encontraste.")
                return

            logger.info("¡Patrón RegEx encontrado! Extrayendo datos JSON...")

            # El Grupo 1 (match.group(1)) contiene el JSON (lo que estaba entre paréntesis en la RegEx)
            json_data_str = match.group(1)

            # Limpieza: A veces los JSON incrustados tienen comas al final (trailing commas)
            # o están envueltos en caracteres extra.
            json_data_str = json_data_str.strip()

            # Intento de parsear el JSON
            try:
                # Quitamos una posible coma al final del array
                if json_data_str.endswith(",]"):
                    json_data_str = json_data_str[:-2] + "]"

                json_data = json.loads(json_data_str)
                logger.info(f"¡JSON parseado exitosamente! Se encontraron {len(json_data)} elementos.")

                # Guardar datos en un archivo JSON
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
                logger.info(f"Datos guardados exitosamente en {OUTPUT_FILE}")
                logger.info("--- PROCESO COMPLETADO ---")

            except json.JSONDecodeError as e:
                logger.error(f"Error al parsear el string de JSON: {e}")
                logger.error("El string extraído podría estar malformado.")
                logger.error(f"--- Inicio del string extraído (primeros 200 caracteres) ---")
                logger.error(json_data_str[:200])
                logger.error(f"--- Fin del string extraído (últimos 200 caracteres) ---")
                logger.error(json_data_str[-200:])

    except httpx.RequestError as e:
        logger.error(f"Error de red al intentar conectar con {e.request.url}: {e}")
    except Exception as e:
        logger.error(f"Ha ocurrido un error inesperado en el script v5: {e}")


async def main():
    # Instalar httpx si no lo tienes: pip install httpx
    await fetch_lot_data_v5_regex()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())