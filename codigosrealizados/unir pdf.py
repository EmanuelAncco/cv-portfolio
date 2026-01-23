import logging
import sys
from pathlib import Path
from pypdf import PdfMerger

CARPETA_BASE = Path(r'C:\Users\Emanuel\Downloads\expediente modificado')

# 3. Define los nombres de tus archivos.
#    ¡CRÍTICO! Estoy asumiendo que son 'parte1.pdf' y 'parte 2.pdf'.
#    Si la extensión es otra (o no tienen), debes escribirla tal cual.
ARCHIVOS_A_UNIR = [
    CARPETA_BASE / 'parte1.pdf',
    CARPETA_BASE / 'parte 2.pdf'  # He respetado el espacio en 'parte 2'
]

# 4. Define dónde guardar el archivo unificado.
#    Lo guardaremos en esa misma carpeta con un nuevo nombre.
ARCHIVO_SALIDA = CARPETA_BASE / 'expediente_completo_unificado.pdf'


def fusionar_pdfs(lista_pdfs: list[Path], ruta_salida: Path):
    """
    Función principal para validar y fusionar una lista de archivos PDF.
    """
    logging.info(f"Iniciando proceso de fusión para {len(lista_pdfs)} archivos.")
    logging.info(f"El archivo de salida será: {ruta_salida}")

    pdf_merger = PdfMerger()
    archivos_procesados = 0
    archivos_fallidos = 0

    # --- 1. Verificación y Adición (Ingeniería Pesimista) ---
    # No asumimos que los archivos existen. Verificamos cada uno.
    for pdf_path in lista_pdfs:
        logging.info(f"Verificando: {pdf_path}...")

        if not pdf_path.exists():
            logging.error(f"¡Archivo NO encontrado! Se omitirá: {pdf_path}")
            archivos_fallidos += 1
            continue  # Continuamos con el siguiente archivo, no detenemos todo.

        if pdf_path.suffix.lower() != '.pdf':
            logging.warning(f"El archivo no es PDF. Se omitirá: {pdf_path}")
            archivos_fallidos += 1
            continue

        try:
            # Añadimos el archivo al objeto 'merger'
            pdf_merger.append(str(pdf_path))
            logging.info(f"Añadido exitosamente: {pdf_path}")
            archivos_procesados += 1
        except Exception as e:
            logging.error(f"No se pudo procesar el archivo {pdf_path}. Error: {e}")
            archivos_fallidos += 1

    # --- 2. Escritura del Archivo Final ---
    # Solo escribimos si procesamos al menos un archivo con éxito.
    if archivos_procesados > 0:
        logging.info(f"Total de archivos añadidos: {archivos_procesados}. Escribiendo en {ruta_salida}...")
        try:
            pdf_merger.write(str(ruta_salida))
            logging.info(f"¡Éxito! Archivo unificado guardado en: {ruta_salida}")
        except Exception as e:
            logging.critical(f"Error CRÍTICO al escribir el archivo de salida. Error: {e}")
            archivos_fallidos += 1  # Contamos esto como un fallo
        finally:
            # Cerramos el objeto merger para liberar recursos.
            pdf_merger.close()
    else:
        logging.warning("No se procesó ningún archivo con éxito. No se generará ningún PDF.")

    # --- 3. Resumen de Ejecución ---
    logging.info("--- Resumen de Fusión ---")
    logging.info(f"Archivos procesados con éxito: {archivos_procesados}")
    logging.info(f"Archivos fallidos u omitidos: {archivos_fallidos}")
    logging.info(f"Total de archivos en la lista: {len(lista_pdfs)}")
    logging.info(f"Log completo guardado en: {log_file}")
    logging.info("--------------------------")


# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    try:
        fusionar_pdfs(ARCHIVOS_A_UNIR, ARCHIVO_SALIDA)
    except Exception as e:
        logging.critical(f"Ha ocurrido un error fatal e inesperado: {e}")
        sys.exit(1)