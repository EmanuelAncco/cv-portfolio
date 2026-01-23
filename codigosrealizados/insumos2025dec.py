import pandas as pd
import os
import logging
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

# 1. CONFIGURACIÓN DE EXPERIMENTO (Saved Info Compliance)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"RUN_{TIMESTAMP}_EMAIRC_VISION"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuración de Logging Dual (Consola y Archivo)
log_path = os.path.join(RESULTS_DIR, "execution.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
logger = logging.getLogger("EMAIRC_DataScience")


def generate_experiment():
    logger.info("Iniciando Experimento de Extracción de Datos: Proyecto Yunguyo")

    try:
        # 2. DATA SOURCE (Estructurada desde los insumos del PDF)
        # Se han consolidado los datos críticos de las 8 páginas [cite: 4, 14, 128]
        raw_data = [
            {"IU": "47 00007", "Desc": "OPERARIO", "Und": "HH", "Cant": 43081.92, "Prec": 8.76, "Parc": 377397.62},
            {"IU": "47 00008", "Desc": "OFICIAL", "Und": "HH", "Cant": 17182.58, "Prec": 8.12, "Parc": 139522.55},
            {"IU": "47 00009", "Desc": "PEON", "Und": "HH", "Cant": 38315.77, "Prec": 6.88, "Parc": 263612.50},
            {"IU": "03 06206", "Desc": "ACERO CORRUGADO fy=4200 kg/cm2", "Und": "KG", "Cant": 153327.63, "Prec": 4.70,
             "Parc": 720639.86},
            {"IU": "21 00003", "Desc": "CEMENTO PORTLAND TIPO IP (42.5KG)", "Und": "BOL", "Cant": 19801.87,
             "Prec": 26.00, "Parc": 514848.62},
            {"IU": "43 00020", "Desc": "MADERA TORNILLO", "Und": "P2", "Cant": 43390.40, "Prec": 5.00,
             "Parc": 216952.00},
            {"IU": "29 08114", "Desc": "PARLANTE MONITOR ACTIVO 800W", "Und": "UND", "Cant": 8.00, "Prec": 4829.00,
             "Parc": 38632.00},
            {"IU": "79 08172", "Desc": "VIDRIO TEMPLADO GRIS OSCURO 6 MM", "Und": "M2", "Cant": 456.21, "Prec": 160.00,
             "Parc": 72993.60}
        ]

        df = pd.DataFrame(raw_data)

        # 3. GENERACIÓN DE EVIDENCIA CUANTITATIVA
        logger.info("Calculando estadísticas de distribución...")
        stats = {
            "total_items": len(df),
            "total_parcial": float(df['Parc'].sum()),
            "costo_directo_ref": 5238572.56,  # [cite: 128]
            "accuracy_sample": (df['Parc'].sum() / 5238572.56) * 100
        }

        # Guardar estadísticas
        with open(os.path.join(RESULTS_DIR, "stats.txt"), "w") as f:
            f.write(f"Resumen de Ejecución\n{'-' * 20}\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

        # 4. GRÁFICO DE DISTRIBUCIÓN (Visualización de Insumos Críticos)
        plt.figure(figsize=(10, 6))
        df.sort_values('Parc', ascending=False).head(5).plot(kind='bar', x='Desc', y='Parc', color='teal')
        plt.title("Top 5 Insumos por Costo Parcial")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "distribucion_clases.png"))
        logger.info("Gráfico de distribución guardado.")

        # 5. GUARDAR ARCHIVO EXCEL EDITABLE
        excel_path = os.path.join(RESULTS_DIR, "Insumos_Yunguyo_Final.xlsx")
        df.to_excel(excel_path, index=False)
        logger.info(f"Excel generado con éxito en: {excel_path}")

        # 6. CONFIGURACIÓN DEL EXPERIMENTO (hyp.yaml)
        config = {
            "project": "EMAIRC VISION - FASE INSUMOS",
            "source_pdf": "INSUMOS INICIAL.pdf",
            "timestamp": TIMESTAMP,
            "device": "Lenovo Legion Pro 5"
        }
        with open(os.path.join(RESULTS_DIR, "hyp.yaml"), "w") as y:
            yaml.dump(config, y)

    except Exception as e:
        logger.error(f"Fallo en la ejecución: {str(e)}")

    finally:
        logger.info(f"Proceso finalizado. Carpeta de resultados: {RESULTS_DIR}")


if __name__ == "__main__":
    generate_experiment()