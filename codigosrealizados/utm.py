import pandas as pd
import pyproj
import logging
import os
import sys
from datetime import datetime

# --- CONFIGURACIÓN DE LOGGING ---
log_filename = f"gps_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def convert_gps_to_utm_csv(input_csv, output_csv):
    """
    Convierte un CSV de GPS (Lat/Long, separado por ;) a formato Civil 3D (UTM Norte/Este, separado por ,).
    Detecta automáticamente la zona UTM adecuada.
    """

    if not os.path.exists(input_csv):
        logger.critical(f"Archivo no encontrado: {input_csv}")
        return

    try:
        logger.info(f"Leyendo archivo GPS: {input_csv}")

        # 1. Lectura Robusta: Especificamos separadores latinos explícitamente
        # sep=';' -> Porque tu archivo usa punto y coma
        # decimal=',' -> Porque tu archivo usa comas para decimales
        df = pd.read_csv(input_csv, sep=';', decimal=',')

        # Limpieza de nombres de columnas (quitar espacios extra)
        df.columns = df.columns.str.strip()
        logger.info(f"Columnas detectadas: {list(df.columns)}")

        # 2. Validación de Columnas Críticas
        required_cols = ['LAT', 'LONG', 'ALT']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Faltan columnas críticas. Se requiere: {required_cols}")
            return

        # 3. Ingeniería de Proyección (Lat/Lon -> UTM)
        # Usamos pyproj para precisión topográfica.
        # Asumimos datum WGS84 (estándar GPS).

        # Detectar zona UTM basada en la longitud promedio
        avg_long = df['LONG'].mean()
        # Fórmula para zona UTM: floor((long + 180) / 6) + 1
        utm_zone = int((avg_long + 180) / 6) + 1
        # Determinar hemisferio (Sur para Perú)
        is_south = df['LAT'].mean() < 0
        hemisphere = "South" if is_south else "North"

        logger.info(f"Georreferenciación: Longitud media {avg_long:.2f} -> Zona UTM calculada: {utm_zone}S")

        # Definir transformador: WGS84 (Lat/Lon) -> WGS84 UTM Zona X
        # EPSG:4326 es Lat/Lon WGS84
        # Construimos el PROJ string para la zona destino
        crs_src = pyproj.CRS("EPSG:4326")
        crs_dst = pyproj.CRS(f"+proj=utm +zone={utm_zone} +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

        transformer = pyproj.Transformer.from_crs(crs_src, crs_dst, always_xy=True)

        logger.info("Iniciando reproyección de coordenadas...")

        # Transformación vectorizada (rápida)
        # Nota: pyproj espera (Longitud, Latitud) para coordenadas geográficas
        eastings, northings = transformer.transform(df['LONG'].values, df['LAT'].values)

        # 4. Construcción del formato para Civil 3D (PENZD)
        # P = Punto (Número)
        # E = Este (Calculado)
        # N = Norte (Calculado)
        # Z = Elevación (Directo del CSV)
        # D = Descripción

        output_df = pd.DataFrame({
            'Punto': range(1, len(df) + 1),
            'Este': eastings,
            'Norte': northings,
            'Elevacion': df['ALT'],
            'Descripcion': 'GPS_TRACK'  # Descripción genérica
        })

        # Redondear a 3 decimales para limpieza (estándar construcción)
        output_df['Este'] = output_df['Este'].round(3)
        output_df['Norte'] = output_df['Norte'].round(3)
        output_df['Elevacion'] = output_df['Elevacion'].round(3)

        # 5. Guardado compatible
        # index=False para no guardar el índice de pandas
        # header=False para que Civil 3D no lea "Este,Norte" como un punto fallido
        output_df.to_csv(output_csv, index=False, header=False)

        logger.info("--- Conversión Exitosa ---")
        logger.info(f"Archivo generado: {output_csv}")
        logger.info(f"Formato: P,E,N,Z,D (Comma Delimited). Sistema: WGS84 UTM Zona {utm_zone}S")
        logger.info("Instrucción Civil 3D: Import Points -> Format: PENZD (comma delimited)")

    except Exception as e:
        logger.critical(f"Error fatal en la conversión: {e}", exc_info=True)


if __name__ == "__main__":
    # Rutas de archivos
    INPUT_FILE = r"datos111.csv"  # Tu archivo descargado
    OUTPUT_FILE = r"puntos_civil3d_procesados.csv"

    convert_gps_to_utm_csv(INPUT_FILE, OUTPUT_FILE)