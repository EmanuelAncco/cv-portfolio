import pandas as pd
import os
import logging
import sys
import importlib.util
from datetime import datetime

# --- CONFIGURACIÓN DE LOGGING ---
log_filename = f"ppc_report_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PPCReportGenerator:
    """
    Generador de reportes PPC (Plan de Porcentaje Cumplido) y Análisis de Causas Raíz (RCA).
    Adaptado para replicar la metodología manual de pizarra (Video de referencia).
    """

    def __init__(self, output_dir="resultados_ppc"):
        self.output_dir = output_dir
        self.filename = "analisis_pareto_clase.xlsx"
        self.filepath = os.path.join(self.output_dir, self.filename)
        self._check_dependencies()
        self._ensure_directory()

    def _check_dependencies(self):
        """Valida dependencias externas críticas (xlsxwriter)."""
        if importlib.util.find_spec("xlsxwriter") is None:
            error_msg = (
                "Falta la librería 'xlsxwriter' necesaria para los gráficos.\n"
                "SOLUCIÓN: Ejecuta -> pip install xlsxwriter"
            )
            logger.critical(error_msg)
            raise ModuleNotFoundError(error_msg)

    def _ensure_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_data(self):
        """
        Datos reconstruidos del video y la imagen.
        Total de incidentes validado = 100 (para coincidir con el % directo del video).
        """
        data = [
            ("SC", "SUBCONTRATAS", 52),
            ("LOG", "LOGISTICA", 16),
            ("EXT", "EVENTOS EXTERNOS", 9),
            ("PROG", "PROGRAMACION", 7),
            ("ADM", "ADMINISTRATIVOS", 5),
            ("REND", "MALOS RENDIMIENTOS", 4),
            ("EQU", "FALTA DE EQUIPOS/AVERIAS", 2),
            ("EJEC", "ERRORES DE EJECUCION", 2),
            ("ING", "INGENIERIA AYG", 1),
            ("INGS", "INDEFINICIONES ING.", 1),
            ("QAQC", "CONTROL DE CALIDAD", 1),
        ]
        # Nota: En el video el profesor usa estas abreviaturas y valores
        df = pd.DataFrame(data, columns=["COD", "CAUSA", "FRECUENCIA"])
        return df

    def calculate_pareto_logic(self, df):
        """
        Replica la lógica de la pizarra:
        1. Ordenar por Frecuencia (Mayor a menor).
        2. Calcular % Individual.
        3. Calcular % Acumulado (La curva).
        """
        # 1. Ordenar
        df = df.sort_values(by="FRECUENCIA", ascending=False).reset_index(drop=True)

        # 2. Calcular Totales y %
        total_incidents = df["FRECUENCIA"].sum()
        df["% INDIVIDUAL"] = df["FRECUENCIA"] / total_incidents

        # 3. Acumulado
        df["% ACUMULADO"] = df["% INDIVIDUAL"].cumsum()

        # 4. Clasificación (Insight de Ingeniería)
        # El profesor marca el corte. Usualmente es 80%.
        # SC(52) + LOG(16) + EXT(9) = 77% -> Estos son los vitales.
        # PROG(7) lleva a 84%.
        df["CLASIFICACION"] = df["% ACUMULADO"].apply(
            lambda x: 'A (Pocos Vitales)' if x <= 0.80 else ('B' if x <= 0.95 else 'C (Muchos Triviales)')
        )

        logger.info(f"Cálculo de Pareto completado. Total muestras: {total_incidents}")
        return df

    def write_excel_dashboard(self, df):
        """Genera el Excel con el gráfico idéntico al del video."""
        logger.info(f"Generando reporte en: {self.filepath}")

        with pd.ExcelWriter(self.filepath, engine='xlsxwriter') as writer:
            sheet_name = 'Pareto Pizarra'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]

            # --- FORMATOS ---
            fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9', 'border': 1, 'align': 'center'})
            fmt_pct = workbook.add_format({'num_format': '0%', 'align': 'center'})
            fmt_vital = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})  # Rojo claro para vitales

            # Aplicar formatos
            worksheet.set_column('B:B', 30)  # Ancho columna Causa
            worksheet.set_column('C:E', 15)

            # Formato condicional para resaltar los "Pocos Vitales" (Clase A)
            worksheet.conditional_format('F2:F12', {
                'type': 'text',
                'criteria': 'containing',
                'value': 'A (Pocos Vitales)',
                'format': fmt_vital
            })

            # --- GRÁFICO COMBINADO (BARRA + LÍNEA) ---
            # Replicando el dibujo de la pizarra
            chart = workbook.add_chart({'type': 'column'})

            # Serie 1: Barras (Frecuencia) - Eje Izquierdo
            chart.add_series({
                'name': 'Frecuencia (Ocurrencias)',
                'categories': [sheet_name, 1, 0, 11, 0],  # Column A (COD) as labels
                'values': [sheet_name, 1, 2, 11, 2],  # Column C (FRECUENCIA)
                'fill': {'color': '#4472C4'},
                'data_labels': {'value': True}  # Mostrar valores sobre las barras como en la clase
            })

            # Serie 2: Línea (% Acumulado) - Eje Derecho
            chart.add_series({
                'name': '% Acumulado',
                'categories': [sheet_name, 1, 0, 11, 0],
                'values': [sheet_name, 1, 4, 11, 4],  # Column E (% ACUMULADO)
                'y2_axis': True,
                'type': 'line',
                'line': {'color': '#C00000', 'width': 2.5},
                'marker': {'type': 'circle', 'size': 6, 'fill': {'color': 'white'}, 'border': {'color': '#C00000'}}
            })

            # Configuración de Ejes para igualar la pizarra
            chart.set_title({'name': 'Diagrama de Pareto (PPC - Causas de Incumplimiento)'})
            chart.set_x_axis({'name': 'Causas'})
            chart.set_y_axis({'name': 'N° Ocurrencias', 'major_gridlines': {'visible': False}})
            chart.set_y2_axis({'name': '% Acumulado', 'max': 1.0, 'min': 0, 'major_unit': 0.2, 'num_format': '0%'})

            # Línea de corte del 80% (Visual aid)
            # Nota: XlsxWriter no soporta líneas arbitrarias fácilmente, pero el grid del eje Y2 ayuda.

            chart.set_size({'width': 720, 'height': 400})
            worksheet.insert_chart('H2', chart)

            logger.info("Gráfico insertado correctamente.")


if __name__ == "__main__":
    try:
        gen = PPCReportGenerator()
        df = gen.get_data()
        df_processed = gen.calculate_pareto_logic(df)
        gen.write_excel_dashboard(df_processed)

        # Resumen final en consola para validación rápida
        print("\n" + "=" * 40)
        print("RESUMEN DE PARETO (TOP 3 - 80/20)")
        print("=" * 40)
        print(df_processed[['COD', 'FRECUENCIA', '% ACUMULADO']].head(4).to_string(index=False))
        print("\nArchivo generado exitosamente.")

    except Exception as e:
        logger.critical(f"Error fatal: {e}")