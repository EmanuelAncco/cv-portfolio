import argparse
import csv
import logging
import os
import time
from datetime import datetime

# --- CONFIGURACIÓN DE LIBRERÍAS DE TERCEROS ---
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    # Manejo de error para la librería crítica.
    print("ERROR: La librería 'playwright' no está instalada o no está disponible.")
    print("Por favor, instale los paquetes necesarios: pip install playwright")
    print("Luego, instale los navegadores: playwright install")
    exit(1)

# --- CONFIGURACIÓN DEL LOGGING ---
# Define un formato de log detallado.
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = 'scraper_execution.log'

# 1. Configuración del archivo de log para auditoría.
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    filename=LOG_FILE,
    filemode='w'  # Sobrescribe el log en cada ejecución
)
# 2. Configuración de la salida a la consola para feedback en tiempo real.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(console_handler)

# --- VARIABLES Y CONSTANTES ---
OUTPUT_FIELDNAMES = ['Site_Name', 'Extracted_Address', 'Scrape_Status', 'Scrape_Timestamp', 'Error_Details']
TIMEOUT_MS = 30000  # 30 segundos de tiempo de espera para carga de página/selector.


def setup_args():
    """
    Configura los argumentos de línea de comandos para el script.

    El script espera dos argumentos:
    1. ruta_input_csv: El archivo CSV que contiene la lista de sitios a raspar.
    2. ruta_output_csv: El archivo CSV donde se guardarán los resultados.
    """
    parser = argparse.ArgumentParser(
        description="Scraper web resiliente para EMAIRC VISIÓN usando Playwright."
    )
    # Se añade la ruta del archivo de entrada que contiene las URLs a raspar.
    parser.add_argument('ruta_input_csv', type=str,
                        help='Ruta al archivo CSV de entrada (ej. lista_sitios.csv)')
    # Se añade la ruta del archivo de salida para los resultados del raspado.
    parser.add_argument('ruta_output_csv', type=str,
                        help='Ruta al archivo CSV de salida para los resultados del raspado.')
    return parser.parse_args()


def scrape_site(page, site_name, url):
    """
    Intenta navegar a una URL y extraer la dirección de contacto.

    Args:
        page: Instancia de la página de Playwright.
        site_name (str): Nombre amigable del sitio (para el log).
        url (str): La URL a raspar.

    Returns:
        dict: Diccionario con los datos extraídos o detalles del error.
    """
    logging.info(f"-> Intentando raspar el sitio: {site_name} ({url})")

    # Intenta navegar a la página.
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        logging.debug(f"Página cargada con éxito: {url}")
    except Exception as e:
        # Maneja fallos de conexión, DNS, o timeouts de carga inicial.
        return {
            'Site_Name': site_name,
            'Extracted_Address': 'N/A',
            'Scrape_Status': 'FAILURE',
            'Scrape_Timestamp': datetime.now().isoformat(),
            'Error_Details': f"Failed to load URL: {e}"
        }

    # --- LÓGICA DE EXTRACCIÓN PERSONALIZADA ---
    # La extracción de la dirección de contacto es el punto más frágil.
    # Usaremos varios selectores comunes (Ingeniería Pesimista).
    # Se asume que la dirección está en el footer, en la sección de contacto, o en una etiqueta de dirección común.

    # 1. Selectores CSS a intentar (ordenados por probabilidad de éxito en sitios empresariales).
    ADDRESS_SELECTORS = [
        '.footer-address',
        'address',
        'p:has-text("Dirección")',
        'div.contact-info:has-text("Calle")',
        'a[href*="maps"]'  # Enlace a Google Maps
    ]

    extracted_text = "Address Not Found"

    for selector in ADDRESS_SELECTORS:
        try:
            # Espera a que el selector esté presente antes de intentar extraer.
            element = page.wait_for_selector(selector, timeout=5000)

            # Si el elemento es un enlace de Google Maps, toma el atributo 'href'.
            if selector == 'a[href*="maps"]':
                extracted_text = element.get_attribute('href') or element.inner_text()
                if "maps" in extracted_text:
                    # A menudo el URL del mapa ya contiene la dirección codificada.
                    extracted_text = f"GMaps Link: {extracted_text}"
            else:
                # Si es un elemento de texto normal, toma el texto interno.
                extracted_text = element.inner_text().strip().replace('\n', ', ')

            logging.info(f"Dirección encontrada usando selector: {selector}")
            break  # Si se encuentra, sale del bucle de selectores.

        except Exception:
            logging.debug(f"Selector {selector} no encontrado para {site_name}.")
            continue  # Intenta el siguiente selector.

    # --- RESULTADO FINAL DEL RASPADOR ---
    if extracted_text != "Address Not Found":
        return {
            'Site_Name': site_name,
            'Extracted_Address': extracted_text,
            'Scrape_Status': 'SUCCESS',
            'Scrape_Timestamp': datetime.now().isoformat(),
            'Error_Details': ''
        }
    else:
        # Falla al encontrar cualquier selector después de intentarlos todos.
        return {
            'Site_Name': site_name,
            'Extracted_Address': 'N/A',
            'Scrape_Status': 'NO_DATA',
            'Scrape_Timestamp': datetime.now().isoformat(),
            'Error_Details': 'No se pudo extraer la dirección con los selectores predefinidos.'
        }


def process_batch(ruta_input_csv, ruta_output_csv):
    """
    Función principal que lee el input, ejecuta el scraping y guarda el output.
    Implementa el principio de "Continuar, no detener" para el procesamiento por lotes.
    """
    start_time = time.time()

    # Estadísticas para el resumen final.
    total_sites = 0
    success_count = 0
    failure_count = 0
    no_data_count = 0

    logging.info("--- INICIO DEL PROCESO DE RASPADO WEB ---")

    # 1. Leer el archivo de entrada.
    try:
        with open(ruta_input_csv, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            sites_to_scrape = list(reader)
            total_sites = len(sites_to_scrape)

        if not sites_to_scrape:
            logging.warning(f"Archivo de entrada vacío o con solo encabezados: {ruta_input_csv}")
            return

    except FileNotFoundError:
        # Error crítico: Detiene el programa si el archivo de entrada no existe.
        logging.error(f"ERROR CRÍTICO: Archivo de entrada no encontrado en la ruta: {ruta_input_csv}")
        return
    except Exception as e:
        # Error crítico: Fallo al leer o parsear el CSV.
        logging.error(f"ERROR CRÍTICO: No se pudo leer el archivo CSV de entrada: {e}")
        return

    logging.info(f"Total de {total_sites} sitios a procesar.")

    # 2. Inicializar Playwright y el archivo de salida.
    with sync_playwright() as p:
        # Usa Chromium como navegador principal.
        browser = p.chromium.launch(headless=True)  # headless=True es ideal para servidores (sin interfaz gráfica).
        context = browser.new_context()
        page = context.new_page()

        # Abre el archivo de salida para escribir los resultados.
        with open(ruta_output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()

            # 3. Procesamiento iterativo de cada sitio.
            for i, site in enumerate(sites_to_scrape):
                site_name = site.get('Site_Name', f'Site_{i + 1}')
                url = site.get('URL')  # Asume que el CSV de entrada tiene una columna 'URL'.

                if not url:
                    logging.warning(f"Sitio {site_name}: URL no encontrada en el registro. Omitiendo.")
                    no_data_count += 1
                    continue

                # Ejecuta el raspado del sitio e implementa el manejo de errores.
                result = scrape_site(page, site_name, url)

                # Actualiza las estadísticas.
                if result['Scrape_Status'] == 'SUCCESS':
                    success_count += 1
                elif result['Scrape_Status'] == 'NO_DATA':
                    no_data_count += 1
                    logging.warning(f"[{i + 1}/{total_sites}] NO DATA: {site_name}. Razón: {result['Error_Details']}")
                else:
                    failure_count += 1
                    logging.error(f"[{i + 1}/{total_sites}] FAILURE: {site_name}. Razón: {result['Error_Details']}")

                # Escribe el resultado en el archivo de salida para que n8n lo recoja.
                writer.writerow(result)

        # Cierra el navegador y el contexto.
        browser.close()

    end_time = time.time()

    # 4. Generar Resumen Estadístico de Ejecución.
    logging.info("\n--- RESUMEN ESTADÍSTICO DE EJECUCIÓN ---")
    logging.info(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos.")
    logging.info(f"Ítems Procesados: {total_sites}")
    logging.info(f"  - Exitosos (Dirección Encontrada): {success_count}")
    logging.info(f"  - Fallidos (Error de Conexión/Carga): {failure_count}")
    logging.info(f"  - Omitidos (No URL/Datos no encontrados): {no_data_count}")
    logging.info(f"Resultados guardados en: {ruta_output_csv}")
    logging.info("--- FIN DEL PROCESO DE RASPADO WEB ---")


if __name__ == '__main__':
    args = setup_args()
    process_batch(args.ruta_input_csv, args.ruta_output_csv)