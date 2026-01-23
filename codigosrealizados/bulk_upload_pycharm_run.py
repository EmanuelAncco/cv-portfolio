import pandas as pd
import requests
import time
import logging
import sys
from requests.exceptions import RequestException

# --- CONFIGURACIÓN PRINCIPAL ---
# 1. Pega tu URL de PRODUCCIÓN de n8n (la que NO tiene "-test")
WEBHOOK_URL = "https://emairc.app.n8n.cloud/webhook/new-lead-webhook"  # <--- ¡PEGA TU URL AQUÍ!

# 2. Ruta a tu archivo de Excel ÚNICO
FILE_PATH = "LLAMADAS COSTAMAR (1).xlsx"  # <--- ¡EL NOMBRE DE TU ARCHIVO EXCEL!

# 3. Retraso entre envíos (en segundos). 0.5 es un valor seguro.
DELAY_BETWEEN_REQUESTS = 0.5

# --- ¡NUEVA ARQUITECTURA! (CON ÍNDICES CORREGIDOS) ---
# Define TODAS las hojas que quieres procesar en un solo lugar.
# El script las ejecutará TODAS, en este orden.
ALL_SHEETS_CONFIG = [
    {
        "sheet_name": "BASE 1",
        "header_index": 1,  # CORREGIDO: Fila 2 en Excel (basado en tu CSV)
        "name_col": "DATOS",
        "phone_col": "NRO CELULAR"
    },
    {
        "sheet_name": "POTENCIALES",
        "header_index": 3,  # CORREGIDO: Fila 4 en Excel (basado en tu CSV)
        "name_col": "NOMBRE",
        "phone_col": "NRO CELULAR"
    },
    {
        "sheet_name": "CLIENTES AGOSTO",
        "header_index": 0,  # CORREGIDO: Fila 1 en Excel (basado en tu CSV)
        "name_col": "DATOS",
        "phone_col": "NRO"
    },
    {
        "sheet_name": "OCTUBRE",
        "header_index": 0,  # CORREGIDO: Fila 1 en Excel (basado en tu CSV)
        "name_col": "DATOS",
        "phone_col": "NRO"
    }
]
# --- FIN DE LA CONFIGURACIÓN ---


# --- CONFIGURACIÓN DEL LOGGING ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename='master_excel_upload.log', filemode='w')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger().addHandler(console_handler)


def process_sheet(sheet_config):
    """
    Procesa una sola hoja de Excel basada en su configuración.
    Retorna (success_count, failure_count, skipped_count)
    """

    sheet_name = sheet_config["sheet_name"]
    header_index = sheet_config["header_index"]
    name_col = sheet_config["name_col"]
    phone_col = sheet_config["phone_col"]

    logging.info(f"--- INICIANDO PROCESAMIENTO DE HOJA: {sheet_name} ---")

    success_count = 0
    failure_count = 0
    skipped_count = 0

    try:
        # Usar pandas para leer la hoja de Excel específica
        df = pd.read_excel(
            io=FILE_PATH,
            sheet_name=sheet_name,
            header=header_index,  # Pandas usa la fila 0-indexada como cabecera
            engine='openpyxl'
        )

        logging.info(f"Hoja '{sheet_name}' cargada. Columnas encontradas: {df.columns.to_list()}")

        # Verificar que las columnas existan
        if name_col not in df.columns or phone_col not in df.columns:
            logging.error(
                f"Error Crítico: Las columnas '{name_col}' o '{phone_col}' no se encuentran en la hoja '{sheet_name}'.")
            logging.error(f"Columnas disponibles: {df.columns.to_list()}")
            return 0, 0, 0  # Omitir esta hoja

        logging.info("Columnas verificadas. Empezando el envío de leads...")

        # Iterar sobre las filas del DataFrame
        for index, row in df.iterrows():
            try:
                # Convertir a string por si pandas los lee como números
                name = str(row[name_col]).strip()
                phone = str(row[phone_col]).strip()

                # Validación: no enviar filas vacías o con valores "nan" (común en pandas)
                if not name or name == "nan" or not phone or phone == "nan":
                    logging.warning(
                        f"Fila {index + header_index + 1}: Datos incompletos (Nombre o Teléfono vacío). Omitiendo.")
                    skipped_count += 1
                    continue

                # Este es el payload que tu n8n SÍ entiende
                payload = {
                    "nombre": name,
                    "telefono": phone,
                    "Source": f"Excel_Bulk_Upload: {FILE_PATH} | {sheet_name}"
                }

                # Enviar a n8n
                response = requests.post(WEBHOOK_URL, json=payload, timeout=10)

                if response.status_code == 200:
                    logging.info(f"ÉXITO ({index + 1}): Lead '{name}' enviado.")
                    success_count += 1
                else:
                    logging.error(
                        f"FALLO ({index + 1}): Lead '{name}'. Status: {response.status_code}, Respuesta: {response.text}")
                    failure_count += 1

            except Exception as e_row:
                logging.error(
                    f"Fila {index + header_index + 1}: Error inesperado al procesar fila: {e_row}. Omitiendo.")
                failure_count += 1

            # Ser amable con el servidor de n8n
            time.sleep(DELAY_BETWEEN_REQUESTS)

    except FileNotFoundError:
        logging.error(f"ERROR CRÍTICO: Archivo no encontrado en: {FILE_PATH}")
        logging.warning("Asegúrate de que el script esté en la misma carpeta que tu archivo .xlsx.")
        return 0, 0, 0
    except ValueError as e:
        logging.error(f"ERROR CRÍTICO: ¿Pusiste mal el nombre de la hoja '{sheet_name}'? Error: {e}")
        return 0, 0, 0
    except Exception as e:
        logging.error(f"ERROR CRÍTICO INESPERADO procesando '{sheet_name}': {e}")
        return 0, 0, 0

    logging.info(f"--- RESUMEN DE HOJA {sheet_name} ---")
    logging.info(f"Éxitos: {success_count} | Fallos: {failure_count} | Omitidos: {skipped_count}")
    return success_count, failure_count, skipped_count


def main():
    """Función principal que orquesta el procesamiento de TODAS las hojas."""
    logging.info(f"--- INICIO DE CARGA MASIVA TOTAL (Modo Orquestador) ---")
    logging.info(f"Cargando archivo: {FILE_PATH}")

    total_success = 0
    total_failure = 0
    total_skipped = 0

    # Bucle maestro que itera sobre la configuración
    for config in ALL_SHEETS_CONFIG:
        s, f, k = process_sheet(config)
        total_success += s
        total_failure += f
        total_skipped += k

        # Pausa entre el procesamiento de cada HOJA
        logging.info("Pausa de 3 segundos antes de la siguiente hoja...")
        time.sleep(3)

    logging.info("--- FIN DE CARGA MASIVA TOTAL ---")
    logging.info("--- RESUMEN FINAL ---")
    logging.info(f"Total de leads enviados con éxito: {total_success}")
    logging.info(f"Total de fallos (ver log): {total_failure}")
    logging.info(f"Total de filas omitidas (vacías/nan): {total_skipped}")
    logging.info("Todas las hojas han sido procesadas.")


if __name__ == '__main__':
    main()