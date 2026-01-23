import os
import logging
from datetime import datetime

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
# Definimos qué buscamos para no llenar el reporte de basura (ruido).
TARGET_EXTENSIONS = ['.cs', '.xml', '.xaml', '.js', '.html']

# Palabras clave que indican que el archivo es "JUGOSO" para nosotros.
# Si un archivo contiene esto, lo guardamos. Si es puro código genérico, lo ignoramos.
RELEVANT_KEYWORDS = [
    "IExternalCommand",
    "Transaction",
    "Rebar",
    "CreateFromCurves",
    "Structure",
    "Geometry",
    "XYZ",
    "FilteredElementCollector",
    "Estribo",
    "Columna",
    "Viga",
    "Math",
    "Stirrup"  # Agregado por si usan términos en inglés
]

# Directorios que NO nos importan (Ruido del compilador y binarios)
IGNORED_DIRS = ['obj', 'bin', 'Properties', '.vs', '.git', '.idea']

# Configuración de Logging (Auditoría)
LOG_FILENAME = f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def is_relevant(content):
    """
    Heurística simple: Determina si el contenido del archivo vale la pena ser analizado.
    """
    # Si el archivo es muy pequeño, probablemente no tenga lógica compleja.
    if len(content) < 50:
        return False

    # Verificar si contiene alguna palabra clave de nuestra lista
    for keyword in RELEVANT_KEYWORDS:
        if keyword in content:
            return True

    return False


def harvest_code(root_dir, output_file):
    """
    Recorre recursivamente el directorio, lee los archivos de código,
    filtra los relevantes y los concatena en un solo informe maestro.
    """
    # Validación de Ingeniería Pesimista: Verificar ruta antes de procesar
    if not os.path.exists(root_dir):
        logging.critical(f"ERROR FATAL: El directorio objetivo no existe: {root_dir}")
        return

    logging.info(f"Iniciando cosecha de código en: {root_dir}")
    logging.info(f"Reporte de salida: {os.path.abspath(output_file)}")

    files_processed = 0
    files_saved = 0
    files_skipped = 0

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(f"REPORTE DE AUDITORÍA DE CÓDIGO - EMAIRC VISIÓN\n")
            outfile.write(f"Fecha: {datetime.now()}\n")
            outfile.write(f"Directorio Origen: {root_dir}\n")
            outfile.write("=" * 80 + "\n\n")

            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Filtrar directorios ignorados en tiempo real (modificando la lista in-place)
                dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

                logging.info(f"Escaneando directorio: {dirpath}")

                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()

                    if ext in TARGET_EXTENSIONS:
                        filepath = os.path.join(dirpath, filename)
                        files_processed += 1

                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                                content = infile.read()

                                # APLICAR FILTRO DE RELEVANCIA
                                if is_relevant(content):
                                    # Escribir cabecera clara para separación visual
                                    outfile.write(f"\n{'=' * 30}\n")
                                    # Usamos relpath para que el reporte sea limpio, sin la ruta E:\ completa repetida
                                    relative_path = os.path.relpath(filepath, root_dir)
                                    outfile.write(f"ARCHIVO: {relative_path}\n")
                                    outfile.write(f"{'=' * 30}\n")
                                    outfile.write(content)
                                    outfile.write("\n\n")

                                    files_saved += 1
                                    logging.info(f"  [+] Guardado (Relevante): {filename}")
                                else:
                                    files_skipped += 1
                                    # logging.debug(f"  [-] Omitido (Irrelevante): {filename}")

                        except Exception as e:
                            logging.error(f"Error leyendo {filename}: {str(e)}")

        # Resumen Estadístico Final
        summary = (
            f"\n\n{'=' * 40}\n"
            f"RESUMEN DE EJECUCIÓN\n"
            f"Directorio analizado: {root_dir}\n"
            f"Archivos escaneados: {files_processed}\n"
            f"Archivos relevantes guardados: {files_saved}\n"
            f"Archivos omitidos (ruido): {files_skipped}\n"
            f"Reporte generado en: {os.path.abspath(output_file)}\n"
            f"{'=' * 40}\n"
        )
        logging.info(summary)
        print(summary)

    except Exception as e:
        logging.critical(f"Error fatal en el proceso: {str(e)}")


if __name__ == "__main__":
    # --- CONFIGURACIÓN ESPECÍFICA PARA EMAIRC (PYCHARM) ---
    # Usamos 'r' antes de las comillas para indicar Raw String y evitar problemas con backslashes en Windows
    PROJECT_DIR = r"E:\proyecto plugins\AcerosConjunto"

    # El archivo de salida se guardará en la carpeta donde esté este script de Python
    OUTPUT_REPORT = "EMAIRC_Analisis_Completo.txt"

    print("--- EMAIRC Code Harvester v1.1 (Windows Edition) ---")
    print(f"Objetivo: {PROJECT_DIR}")

    harvest_code(PROJECT_DIR, OUTPUT_REPORT)

    print(f"\n¡Listo! Busca el archivo '{OUTPUT_REPORT}' en tu proyecto de PyCharm y súbelo al chat.")