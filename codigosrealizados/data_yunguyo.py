import pandas as pd
import os
import logging
import sys
from datetime import datetime

# --- CORRECCIÓN DE CODIFICACIÓN PARA WINDOWS ---
# Justificación: Windows suele usar la codificación 'cp1252' por defecto en la consola.
# Esto causa que el programa colapse al intentar imprimir emojis (🚀, ✅).
# La 'Ingeniería Pesimista' dicta que debemos forzar la codificación a UTF-8 explícitamente
# para evitar depender de la configuración del entorno del usuario.
if sys.platform == 'win32':
    try:
        # Reconfiguramos la salida estándar para soportar caracteres unicode
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback para versiones muy antiguas de Python (aunque tienes 3.10, es buena práctica)
        pass

# --- CONFIGURACIÓN DEL LOGGING ---
log_filename = f"etl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Configuramos handlers específicos
file_handler = logging.FileHandler(log_filename, encoding='utf-8')  # Forzamos UTF-8 en el archivo
stream_handler = logging.StreamHandler(sys.stdout)  # Usará el stdout reconfigurado arriba

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)


class PowerBIPrepare:
    def __init__(self, directory, filename, sheet_name):
        self.directory = directory
        self.filename = filename
        self.sheet_name = sheet_name
        self.full_path = os.path.join(directory, filename)

    def validate_paths(self):
        """Verifica que el archivo exista antes de intentar cargarlo."""
        if not os.path.exists(self.full_path):
            logger.critical(f"❌ El archivo no existe en la ruta: {self.full_path}")
            raise FileNotFoundError(f"Archivo no encontrado: {self.full_path}")
        logger.info(f"✅ Archivo encontrado: {self.full_path}")

    def extract_data(self):
        """
        Extrae los datos de Excel manejando el encabezado desplazado.
        Justificación: Los reportes de Excel suelen tener títulos en las primeras filas.
        Usamos 'header=3' (Fila 4) basado en la estructura visualizada en tus CSVs.
        """
        try:
            logger.info(f"📂 Cargando hoja '{self.sheet_name}'...")

            # Leemos el excel. Asumimos que la tabla empieza en la fila 4 (índice 3)
            # basado en tus archivos CSV que muestran 3 líneas de metadatos antes del header.
            df = pd.read_excel(
                self.full_path,
                sheet_name=self.sheet_name,
                header=3
            )

            logger.info(f"📥 Datos crudos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df

        except PermissionError:
            logger.critical("🔒 ERROR: El archivo Excel está abierto. Ciérralo e intenta de nuevo.")
            sys.exit(1)
        except ValueError as e:
            logger.critical(f"❌ Error de valor (probablemente nombre de hoja incorrecto): {e}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"❌ Error inesperado al leer Excel: {e}")
            sys.exit(1)

    def transform_data(self, df):
        """
        Limpia y tipifica los datos para Power BI.
        Justificación: Power BI necesita consistencia. Eliminamos filas vacías y
        forzamos tipos numéricos para evitar errores en DAX.
        """
        logger.info("⚙️ Iniciando transformación y limpieza...")

        # 1. Eliminar filas que no sean datos reales (ej. filas totalmente vacías o pies de página)
        # Usamos la columna 'Item' o 'Codigo' como ancla. Si no tienen dato, la fila no sirve.
        df_clean = df.dropna(subset=['Codigo', 'Descripcion']).copy()

        # 2. Definir columnas numéricas críticas
        numeric_cols = [
            'Metrado_Orig', 'PU_Orig', 'Total_Orig',
            'Metrado_Mod', 'PU_Mod', 'Total_Mod',
            'Dif_Metrado', 'Dif_PU', 'Dif_Total',
            'Var_Metrado_%', 'Var_PU_%', 'Var_Total_%',
            'Magnitud', 'Impacto_Abs'
        ]

        # 3. Limpieza Numérica (Ingeniería Pesimista: asumir que hay texto o espacios en números)
        for col in numeric_cols:
            if col in df_clean.columns:
                # Convertir a numérico, forzando errores a NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Rellenar NaNs con 0 (Power BI prefiere 0 a null en matemáticas)
                df_clean[col] = df_clean[col].fillna(0)
            else:
                logger.warning(f"⚠️ Columna esperada no encontrada: {col}")

        # 4. Limpieza de Texto
        text_cols = ['Codigo', 'Descripcion', 'Unidad', 'Categoria', 'Tipo_Variacion']
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()

        # 5. Ordenamiento (Opcional, pero solicitado)
        # Ordenamos por Impacto Absoluto descendente para que Power BI muestre lo importante primero
        if 'Impacto_Abs' in df_clean.columns:
            df_clean = df_clean.sort_values(by='Impacto_Abs', ascending=False)
            logger.info("📊 Datos ordenados por Impacto Absoluto (Descendente)")

        # 6. Metadata de Auditoría
        df_clean['Data_Refreshed_At'] = datetime.now()

        logger.info(f"✨ Transformación completada. Filas limpias: {df_clean.shape[0]}")
        return df_clean

    def generate_animation_dataset(self, df):
        """
        Genera un dataset estructurado verticalmente (Unpivoted) para permitir
        animaciones de 'Play Axis' en Power BI.

        Estrategia:
        Creamos una 'línea de tiempo' artificial.
        - Tiempos 1: Estado Original
        - Tiempos 2: Estado Modificado
        Esto permite que Power BI interpole y anime el cambio entre ambos estados.
        """
        logger.info("🎬 Generando dataset optimizado para animaciones...")

        # Seleccionamos solo columnas críticas para la animación para no inflar el archivo
        base_cols = ['Codigo', 'Descripcion', 'Categoria', 'Unidad']

        # Crear DataFrame del estado ORIGINAL
        df_orig = df[base_cols + ['Metrado_Orig', 'PU_Orig', 'Total_Orig']].copy()
        df_orig.rename(columns={
            'Metrado_Orig': 'Metrado',
            'PU_Orig': 'Precio_Unitario',
            'Total_Orig': 'Total'
        }, inplace=True)
        df_orig['Estado'] = '1. Presupuesto Base'
        df_orig['Orden_Tiempo'] = 1  # Clave para el eje de reproducción

        # Crear DataFrame del estado MODIFICADO
        df_mod = df[base_cols + ['Metrado_Mod', 'PU_Mod', 'Total_Mod']].copy()
        df_mod.rename(columns={
            'Metrado_Mod': 'Metrado',
            'PU_Mod': 'Precio_Unitario',
            'Total_Mod': 'Total'
        }, inplace=True)
        df_mod['Estado'] = '2. Presupuesto Modificado'
        df_mod['Orden_Tiempo'] = 2

        # Unir ambos estados (Stacking)
        df_animation = pd.concat([df_orig, df_mod], ignore_index=True)

        # Calcular variación para colorear burbujas (Truco visual)
        # Unimos con el impacto original para tenerlo disponible en ambos estados
        impacto_map = df.set_index('Codigo')['Impacto_Abs'].to_dict()
        df_animation['Impacto_Global'] = df_animation['Codigo'].map(impacto_map).fillna(0)

        output_filename = f"PBI_Animation_Source.csv"
        output_path = os.path.join(self.directory, output_filename)

        try:
            df_animation.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"✅ DATASET DE ANIMACIÓN GENERADO: {output_path}")
        except Exception as e:
            logger.error(f"❌ Error al guardar dataset de animación: {e}")

    def load_to_csv(self, df):
        """
        Exporta a CSV optimizado para Power BI.
        Justificación: CSV es más rápido de leer para Power BI que xlsx y evita bloqueos de archivo.
        """
        output_filename = f"PBI_Source_{self.sheet_name}.csv"
        output_path = os.path.join(self.directory, output_filename)

        try:
            df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=',')
            logger.info(f"✅ EXPORTACIÓN EXITOSA: {output_path}")
            logger.info("👉 Ahora puedes importar este archivo en Power BI usando 'Obtener Datos > Texto/CSV'")
        except Exception as e:
            logger.error(f"❌ Error al guardar CSV: {e}")


def main():
    # --- PARÁMETROS DE ENTRADA ---
    # Ajusta estas rutas si mueves el script o cambias carpetas
    TARGET_DIR = r"C:\Users\Emanuel\Downloads"
    FILE_NAME = "EXCEL_INSUMOS_REALES_COMPLETO 24 nov.xlsx"
    SHEET_NAME = "ANALISIS_COMPLETO"

    logger.info("🚀 Iniciando Proceso ETL para EMAIRC VISIÓN - Power BI")

    etl = PowerBIPrepare(TARGET_DIR, FILE_NAME, SHEET_NAME)

    try:
        etl.validate_paths()
        df_raw = etl.extract_data()
        df_clean = etl.transform_data(df_raw)

        # 1. Exportar datos maestros (Tabla estándar)
        etl.load_to_csv(df_clean)

        # 2. Exportar datos para animación (Tabla transpuesta)
        etl.generate_animation_dataset(df_clean)

    except Exception as e:
        logger.critical(f"💀 El proceso falló fatalmente: {e}")


if __name__ == "__main__":
    main()