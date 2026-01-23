import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import sys
import os
from datetime import datetime
import json

# --- CONFIGURACIÓN DEL ENTORNO ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"experiments/exp_{TIMESTAMP}"
LOG_FILE = os.path.join(OUTPUT_DIR, "execution.log")


# Configuración de Logging
def setup_logging():
    """Configura el sistema de logging para auditoría y debug."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Configuración robusta sin emojis para evitar errores de encoding en Windows (cp1252)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),  # Archivo soporta UTF-8
            logging.StreamHandler(sys.stdout)  # Consola depende del sistema
        ]
    )
    logging.info(f"Directorio de experimentos creado: {OUTPUT_DIR}")


# --- CLASE PRINCIPAL ---
class SalesTrendAnalyzer:
    """
    Clase encargada de cargar, procesar, modelar y visualizar tendencias de ventas.
    Diseño orientado a objetos para encapsular el estado y la lógica.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.model = None
        self.metrics = {}
        self.results = {}

    def load_data(self):
        """
        Carga el dataset manejando CSV o Excel de forma defensiva.
        """
        logging.info(f"Intentando cargar datos desde: {self.filepath}")

        try:
            if not os.path.exists(self.filepath):
                current_dir = os.getcwd()
                logging.error(f"[FALLO DE RUTA] El archivo no existe en: {current_dir}")

                logging.info("[DIAGNOSTICO] Escaneando directorio actual...")
                candidates = [f for f in os.listdir(current_dir) if f.endswith(('.csv', '.xlsx'))]

                if candidates:
                    logging.info(f"[CANDIDATOS] Archivos encontrados: {candidates}")
                    logging.info("[ACCION] Actualiza la variable 'INPUT_FILE' en el script.")
                else:
                    logging.warning("[ALERTA] No se encontraron archivos de datos compatibles.")

                raise FileNotFoundError(f"El archivo {self.filepath} no existe.")

            # --- MEJORA: DETECCIÓN DE FORMATO ---
            if self.filepath.lower().endswith('.xlsx'):
                logging.info("Detectado formato Excel (.xlsx). Usando motor 'openpyxl'...")
                self.df = pd.read_excel(self.filepath)
            else:
                logging.info("Detectado formato CSV. Usando lectura estandar...")
                self.df = pd.read_csv(self.filepath)

            # Normalización de nombres de columnas
            self.df.columns = [c.strip() for c in self.df.columns]

            logging.info(f"Datos cargados exitosamente. Dimensiones: {self.df.shape}")
            logging.info(f"Columnas detectadas: {list(self.df.columns)}")

            # Validación de esquema mínimo
            required_cols = ['10-Piece Set sold units', 'Time in month']
            missing = [c for c in required_cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"Faltan columnas criticas: {missing}")

        except Exception as e:
            logging.error(f"Error critico cargando datos: {e}")
            raise

    def train_linear_model(self):
        """
        Entrena un modelo de regresión lineal simple.
        X = Time in month
        y = Sold units
        """
        logging.info("Iniciando entrenamiento del modelo de Regresion Lineal...")

        try:
            # Preparación de datos (Scikit-Learn espera arrays 2D para X)
            X = self.df[['Time in month']].values
            y = self.df['10-Piece Set sold units'].values

            self.model = LinearRegression()
            self.model.fit(X, y)

            # Predicciones
            predictions = self.model.predict(X)
            self.df['prediction'] = predictions

            # Extracción de coeficientes
            slope = self.model.coef_[0]  # Pendiente (m)
            intercept = self.model.intercept_  # Intercepto (b)

            logging.info(f"Entrenamiento completado. Ecuacion: y = {slope:.4f}x + {intercept:.4f}")

            self.results = {
                'slope': slope,
                'intercept': intercept,
                'predictions': predictions
            }

        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {e}")
            raise

    def evaluate_model(self):
        """Calcula métricas de rendimiento y genera evidencia cuantitativa."""
        logging.info("Calculando metricas de rendimiento...")

        y_true = self.df['10-Piece Set sold units']
        y_pred = self.df['prediction']

        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        self.metrics = {
            'MSE (Error Cuadratico Medio)': mse,
            'R2 Score (Coeficiente de Determinacion)': r2,
            'Interpretacion R2': f"{r2 * 100:.2f}% de la varianza es explicada por el tiempo."
        }

        logging.info(f"Metricas calculadas: R2={r2:.4f}")

        # Guardar métricas en archivo de texto
        metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.txt")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            for k, v in self.metrics.items():
                f.write(f"{k}: {v}\n")
            f.write(
                f"\nEcuacion de la Tendencia: Sales = {self.results['slope']:.4f} * Time + {self.results['intercept']:.4f}\n")

    def generate_visual_evidence(self):
        """Genera gráficos para el reporte (Evidencia Cualitativa)."""
        logging.info("Generando graficos de evidencia...")

        try:
            plt.figure(figsize=(12, 6))

            # Datos Reales
            plt.scatter(self.df['Time in month'], self.df['10-Piece Set sold units'],
                        color='blue', alpha=0.6, label='Datos Reales (Ventas)')

            # Línea de Tendencia
            plt.plot(self.df['Time in month'], self.df['prediction'],
                     color='red', linewidth=2,
                     label=f"Tendencia (R2={self.metrics['R2 Score (Coeficiente de Determinacion)']:.2f})")

            plt.title('Analisis de Tendencia de Ventas: Real vs Modelo Lineal', fontsize=14)
            plt.xlabel('Tiempo (Mes consecutivo)', fontsize=12)
            plt.ylabel('Unidades Vendidas', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # Anotación de la ecuación
            eq_text = f"y = {self.results['slope']:.2f}x + {self.results['intercept']:.2f}"
            plt.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                         fontsize=12, bbox=dict(boxstyle="round", fc="white", ec="black"))

            # Guardar
            plot_path = os.path.join(OUTPUT_DIR, "trend_visualization.png")
            plt.savefig(plot_path)
            plt.close()
            logging.info(f"Grafico guardado en: {plot_path}")

        except Exception as e:
            logging.error(f"Error generando visualizaciones: {e}")

    def save_results_csv(self):
        """Guarda el DataFrame con las predicciones para auditoría."""
        csv_path = os.path.join(OUTPUT_DIR, "processed_results.csv")
        self.df.to_csv(csv_path, index=False)
        logging.info(f"Datos procesados guardados en: {csv_path}")


# --- EJECUCIÓN DEL PIPELINE ---
def main():
    setup_logging()

    # ---------------------------------------------------------
    # CONFIGURACIÓN DE ENTRADA
    # Nombre exacto detectado en tu log de errores:
    INPUT_FILE = "knXNF4icEemVeg5DpI4LqA_3c0cd64932044110aa7ba4bb5a2de17d_4.Trend-model-example.xlsx"
    # ---------------------------------------------------------

    logging.info("--- INICIO DEL EXPERIMENTO DE MODELADO DE TENDENCIA ---")

    analyzer = SalesTrendAnalyzer(INPUT_FILE)

    try:
        analyzer.load_data()
        analyzer.train_linear_model()
        analyzer.evaluate_model()
        analyzer.generate_visual_evidence()
        analyzer.save_results_csv()

        logging.info("--- EXPERIMENTO FINALIZADO CON EXITO ---")
        logging.info(f"Revisar carpeta '{OUTPUT_DIR}' para ver los resultados.")

    except Exception as e:
        logging.critical(f"El experimento fallo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()