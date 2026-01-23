import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuración del Logging ---
# Este es un experimento científico reproducible, así que registramos todo.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analisis_distancias.log", mode='w'),
        logging.StreamHandler()
    ]
)

# --- 1. Definir rutas de archivos y carpetas ---
# NOTA: Se ha actualizado la ruta para que apunte a la ubicación especificada por el usuario.
# Esto hace que el script funcione para un caso de uso específico.
NOMBRE_ARCHIVO_PUNTOS = r'D:\2025 - 2 (ultimo ciclo)\Plano grupo 03_puntos.txt'

logging.info("Iniciando el script de análisis de distancias.")
logging.info(f"Archivo de entrada esperado: {NOMBRE_ARCHIVO_PUNTOS}")

# Creamos la ruta base para los resultados, que estará en el mismo directorio del archivo de entrada.
ruta_base = os.path.dirname(NOMBRE_ARCHIVO_PUNTOS)
CARPETA_RESULTADOS = os.path.join(ruta_base, f'resultados_distancias_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
NOMBRE_ARCHIVO_SALIDA = 'distancias_entre_puntos.xlsx'

# Crear la carpeta de resultados si no existe.
try:
    if not os.path.exists(CARPETA_RESULTADOS):
        os.makedirs(CARPETA_RESULTADOS)
        logging.info(f"Carpeta de resultados creada: {CARPETA_RESULTADOS}")
except OSError as e:
    logging.error(f"Error crítico: No se pudo crear la carpeta de resultados. Deteniendo la ejecución. Error: {e}")
    exit()

# --- 2. Cargar los datos ---
try:
    df_puntos = pd.read_csv(NOMBRE_ARCHIVO_PUNTOS)
    logging.info(f"Archivo de puntos '{NOMBRE_ARCHIVO_PUNTOS}' cargado correctamente.")
    logging.info(f"Se encontraron {len(df_puntos)} puntos en el archivo.")
    logging.info("Cabecera del DataFrame:\n" + str(df_puntos.head()))
except FileNotFoundError:
    logging.error(f"Error: El archivo '{NOMBRE_ARCHIVO_PUNTOS}' no se encontró.")
    logging.error("Asegúrate de que la ruta y el nombre del archivo sean correctos.")
    exit()
except pd.errors.ParserError as e:
    logging.error(f"Error de formato: El archivo '{NOMBRE_ARCHIVO_PUNTOS}' no es un CSV válido.")
    logging.error(f"Verifica que el archivo esté en el formato X,Y,Z. Error: {e}")
    exit()

# --- 3. Calcular las distancias entre puntos consecutivos ---
# Creamos un DataFrame vacío para almacenar los resultados.
df_distancias = pd.DataFrame(columns=['Punto_Inicial', 'Punto_Final', 'Distancia_2D_m', 'Distancia_3D_m'])

num_puntos = len(df_puntos)
puntos_procesados = 0

logging.info("Calculando las distancias entre puntos consecutivos...")

# Iteramos desde el segundo punto hasta el final para calcular la distancia con el punto anterior.
for i in range(1, num_puntos):
    try:
        # Extraemos las coordenadas del punto actual y el anterior.
        punto_anterior = df_puntos.iloc[i - 1][['X', 'Y', 'Z']].values
        punto_actual = df_puntos.iloc[i][['X', 'Y', 'Z']].values

        # Calculamos la distancia 2D (ignorando Z).
        distancia_2d = np.linalg.norm(punto_actual[:2] - punto_anterior[:2])

        # Calculamos la distancia 3D.
        distancia_3d = np.linalg.norm(punto_actual - punto_anterior)

        # Preparamos los datos para la nueva fila.
        nueva_fila = {
            'Punto_Inicial': i - 1,
            'Punto_Final': i,
            'Distancia_2D_m': round(distancia_2d, 4),  # Redondeamos a 4 decimales
            'Distancia_3D_m': round(distancia_3d, 4)  # Redondeamos a 4 decimales
        }

        # Agregamos la nueva fila al DataFrame.
        df_distancias = pd.concat([df_distancias, pd.DataFrame([nueva_fila])], ignore_index=True)
        puntos_procesados += 1

        # Opcional: mostrar progreso para archivos grandes.
        if i % 10 == 0:
            logging.info(f"-> Procesando punto {i} de {num_puntos}...")

    except Exception as e:
        logging.warning(f"Error al procesar el punto {i}. Saltando este cálculo. Error: {e}")
        continue

logging.info("Cálculo de distancias completado.")
logging.info(f"Total de distancias calculadas: {puntos_procesados}")

# --- 4. Guardar los resultados en un archivo de Excel ---
ruta_salida = os.path.join(CARPETA_RESULTADOS, NOMBRE_ARCHIVO_SALIDA)

try:
    with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
        df_distancias.to_excel(writer, index=False, sheet_name='Distancias')
    logging.info(f"Archivo de resultados guardado exitosamente en: {ruta_salida}")
except Exception as e:
    logging.error(f"Error al guardar el archivo de Excel. Error: {e}")
    logging.info("Aquí está la tabla de resultados para tu referencia:")
    logging.info("\n" + str(df_distancias))

logging.info("--- Resumen de la Ejecución ---")
logging.info(f"Puntos procesados exitosamente: {puntos_procesados}")
logging.info(f"Cálculos fallidos: {num_puntos - 1 - puntos_procesados}")
logging.info("Proceso finalizado.")
