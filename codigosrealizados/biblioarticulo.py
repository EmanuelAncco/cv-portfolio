import os
import re
import logging
from datetime import datetime
from PyPDF2 import PdfReader
import getpass  # Librería para obtener el nombre de usuario

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
# Obtenemos el nombre de usuario actual para construir las rutas
USERNAME = getpass.getuser()

# Define la ruta de entrada EXACTA que me proporcionaste.
ARTICLES_DIR = r'C:\Users\Emanuel\Downloads\descargas articulos graphos'

# Define una ruta de salida clara, por ejemplo, en el Escritorio del usuario.
RESULTS_DIR = fr'C:\Users\{USERNAME}\Desktop\Resultados_Bibliografia'
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_RESULTS_DIR = os.path.join(RESULTS_DIR, f"run_{TIMESTAMP}")

# --- CONFIGURACIÓN DEL LOGGING ---
LOG_FILE = os.path.join(RUN_RESULTS_DIR, 'execution.log')


def setup_logging():
    """
    Configura el logging para que escriba en la consola y en un archivo.
    El porqué: Separa la visibilidad en tiempo real (consola) de la
    persistencia para auditoría (archivo).
    """
    if not os.path.exists(RUN_RESULTS_DIR):
        os.makedirs(RUN_RESULTS_DIR)
        print(f"Directorio de resultados creado en: {RUN_RESULTS_DIR}")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def extract_metadata_from_pdf(pdf_path):
    """
    Extrae metadatos (título, autores, DOI) de un archivo PDF.
    El porqué: Se enfoca en las primeras páginas para eficiencia, ya que
    los metadatos clave suelen estar al inicio. Usa expresiones regulares
    (regex) por su alta precisión para encontrar patrones estandarizados como el DOI.
    Se envuelve en un bloque try...except para ser resiliente a PDFs corruptos.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        num_pages_to_scan = min(2, len(reader.pages))
        for i in range(num_pages_to_scan):
            page = reader.pages[i]
            text += page.extract_text() or ""  # Añadimos 'or ""' para manejar páginas vacías

        text = text.replace('\n', ' ').strip()

        doi_pattern = re.compile(r'doi[:\s]*/?10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
        doi_match = doi_pattern.search(text)
        doi = doi_match.group(0).replace('doi:', '').replace('doi', '').strip() if doi_match else "No encontrado"

        info = reader.metadata
        title = info.title if info.title else "Título no encontrado"
        author_str = info.author if info.author else "Autores no encontrados"

        if author_str and author_str != "Autores no encontrados":
            # Limpia posibles codificaciones extrañas
            author_str = re.sub(r'[^\x00-\x7F]+', ' ', author_str)
            first_author = author_str.split(',')[0].split(';')[0].strip()
            name_parts = [part for part in first_author.split() if part]
            if name_parts:
                last_name = name_parts[-1]
                # Formato Apellido, Iniciales.
                initials = " ".join([f"{p[0]}." for p in name_parts[:-1]])
                formatted_author = f"{last_name}, {initials} et al."
                sort_key = last_name.lower()
            else:
                formatted_author = "Autores no encontrados"
                sort_key = "zzz"
        else:
            formatted_author = "Autores no encontrados"
            sort_key = "zzz"

        return {
            'title': title.strip() if title else "Título no encontrado",
            'author': formatted_author,
            'doi': doi.strip(),
            'sort_key': sort_key,
            'source_file': os.path.basename(pdf_path)
        }

    except Exception as e:
        logging.error(f"No se pudo procesar el archivo '{os.path.basename(pdf_path)}'. Razón: {e}")
        return None


def main():
    """
    Función principal que orquesta todo el proceso.
    """
    setup_logging()
    logging.info("--- INICIO DEL PROCESO DE EXTRACCIÓN BIBLIOGRÁFICA ---")

    if not os.path.exists(ARTICLES_DIR):
        logging.critical(f"Error Crítico: El directorio de artículos '{ARTICLES_DIR}' no existe. Abortando.")
        return

    pdf_files = [f for f in os.listdir(ARTICLES_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logging.warning("No se encontraron archivos PDF en el directorio de artículos.")
        return

    logging.info(f"Se encontraron {len(pdf_files)} archivos PDF para procesar.")

    all_metadata = []
    success_count = 0
    fail_count = 0

    for filename in pdf_files:
        logging.info(f"Procesando: {filename}...")
        pdf_path = os.path.join(ARTICLES_DIR, filename)
        metadata = extract_metadata_from_pdf(pdf_path)
        if metadata:
            all_metadata.append(metadata)
            success_count += 1
        else:
            fail_count += 1

    logging.info("Procesamiento completado. Iniciando post-procesamiento...")

    unique_articles = {}
    omitted_duplicates = 0
    for data in all_metadata:
        key = data['doi']
        if key != "No encontrado" and key in unique_articles:
            omitted_duplicates += 1
            logging.warning(f"Duplicado omitido (DOI {key}): '{data['source_file']}'")
            continue

        title_key = data['title'].lower() if data['title'] else ""
        if key == "No encontrado" and title_key in unique_articles:
            omitted_duplicates += 1
            logging.warning(f"Duplicado omitido (Título): '{data['source_file']}'")
            continue

        unique_articles[key if key != "No encontrado" else title_key] = data

    final_list = list(unique_articles.values())
    final_list.sort(key=lambda x: x['sort_key'])

    output_file_path = os.path.join(RUN_RESULTS_DIR, 'bibliografia.txt')
    logging.info(f"Escribiendo bibliografía final en: {output_file_path}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("Bibliografía Generada Automáticamente\n")
        f.write(f"Fecha de Ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        for i, data in enumerate(final_list, 1):
            line = (
                f"[{i:<3}]    {data['author']}, "
                f"\"{data['title']}\", "
                f"doi: {data['doi']}.\n"
            )
            f.write(line)

    logging.info("Archivo de bibliografía generado con éxito.")

    logging.info("--- RESUMEN DE LA EJECUCIÓN ---")
    logging.info(f"Total de archivos PDF encontrados : {len(pdf_files)}")
    logging.info(f"Procesados con éxito            : {success_count}")
    logging.info(f"Fallos al procesar              : {fail_count} (ver detalles en el log)")
    logging.info(f"Duplicados omitidos             : {omitted_duplicates}")
    logging.info(f"Entradas únicas en bibliografía : {len(final_list)}")
    logging.info("--- PROCESO FINALIZADO ---")


if __name__ == "__main__":
    main()