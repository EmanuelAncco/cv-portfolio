import os
import logging
import rasterio
import numpy as np

# --- 1. CONFIGURACIÓN DE LOGGING (Estándar EMAIRC) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sar_processing.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


def convert_sar_to_civil3d(input_tif, output_tif):
    """
    Convierte un GeoTIFF científico (SAR/DEM, float32/int16) a un formato
    visual estándar (uint8) compatible con AutoCAD Civil 3D.

    Aplica normalización lineal basada en percentiles para mejorar el contraste.
    """
    logger.info(f"--- Iniciando conversión para Civil 3D ---")
    logger.info(f"Archivo entrada: {input_tif}")

    # 1. Validación de existencia (Ingeniería Pesimista)
    if not os.path.exists(input_tif):
        logger.critical(f"El archivo no existe: {input_tif}")
        return

    try:
        with rasterio.open(input_tif) as src:
            logger.info(f"Metadatos originales: Driver={src.driver}, Dtype={src.dtypes[0]}, Bands={src.count}")

            # 2. Lectura de datos
            # Leemos solo la primera banda (generalmente HH o HV en PALSAR)
            data = src.read(1)

            # Manejar valores NoData (a veces vienen como -9999 o NaN)
            nodata_val = src.nodata
            if nodata_val is not None:
                mask = data != nodata_val
            else:
                mask = np.ones_like(data, dtype=bool)

            # 3. Normalización Estadística (Rescaling)
            # El radar tiene mucho "ruido" (speckle). Usamos percentiles 2% y 98%
            # para ignorar valores extremos y mejorar el contraste visual.
            valid_data = data[mask]

            if valid_data.size == 0:
                logger.error("El archivo parece estar vacío o solo contiene NoData.")
                return

            p2, p98 = np.percentile(valid_data, (2, 98))
            logger.info(f"Rango de datos detectado (2%-98%): {p2} a {p98}")

            # Clip de valores fuera del rango y escalado a 0-255
            data_clipped = np.clip(data, p2, p98)
            # Fórmula de normalización: (x - min) / (max - min) * 255
            data_normalized = ((data_clipped - p2) / (p98 - p2) * 255)

            # Convertir a enteros de 8 bits (Lo que Civil 3D necesita)
            data_uint8 = data_normalized.astype('uint8')

            # Limpiar zonas NoData (hacerlas 0 - negro)
            if nodata_val is not None:
                data_uint8[~mask] = 0

            # 4. Preparar perfil de salida
            profile = src.profile.copy()
            profile.update(
                dtype=rasterio.uint8,
                count=1,
                driver='GTiff',
                compress='lzw',  # LZW es seguro para Civil 3D
                nodata=0
            )

            # 5. Guardar
            os.makedirs(os.path.dirname(output_tif), exist_ok=True)
            with rasterio.open(output_tif, 'w', **profile) as dst:
                dst.write(data_uint8, 1)

            logger.info(f"¡Éxito! Archivo guardado en: {output_tif}")
            logger.info("Ahora puedes usar MAPIINSERT con este archivo nuevo.")

    except Exception as e:
        logger.critical(f"Error fatal procesando el TIF: {e}")
        raise e


# --- EJECUCIÓN ---
if __name__ == "__main__":
    # RUTA DE TU ARCHIVO (Cópiala tal cual la tienes)
    # Sugerencia: Mueve el archivo a una ruta más simple primero para evitar errores de ruta en Windows
    input_file = r"C:\Users\Emanuel\Downloads\tf hidra ARITA\terreno\terreno.tif"

    # Archivo arreglado
    output_file = r"C:\Users\Emanuel\Downloads\tf hidra ARITA\terreno\terreno_civil_fixed.tif"

    convert_sar_to_civil3d(input_file, output_file)