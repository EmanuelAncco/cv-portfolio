import pandas as pd
import xlsxwriter
import logging
import os
import sys
import io
from datetime import datetime

# --- Configuración de Logging ---
# Configuración robusta para auditoría y depuración
log_filename = f"carta_balance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CartaBalanceAnalyzer:
    """
    Clase para gestionar el análisis de Carta Balance y la generación de reportes.
    Procesa data cruda minuto a minuto para garantizar trazabilidad total.
    """

    def __init__(self, output_folder="resultados_carta_balance_g4"):
        self.output_folder = output_folder
        self.df_raw = None  # DataFrame para el registro minuto a minuto
        self.df_summary = None  # DataFrame para los totales calculados

        # Crear carpeta de resultados si no existe (Ingeniería Pesimista)
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            logger.info(f"Directorio de salida verificado: {self.output_folder}")
        except OSError as e:
            logger.critical(f"No se pudo crear el directorio de salida: {e}")
            sys.exit(1)

    def load_data(self):
        """
        Carga y procesa la data cruda del GRUPO 4.
        Incluye el registro completo de 60 minutos para cálculo automático.
        """
        logger.info("Cargando registro de campo completo (Minuto a Minuto)...")

        # Data cruda en formato CSV embebido para reproducibilidad
        csv_data = """Minuto,Juan Espinoza (Manguera),Alberto Requena (Vibrado),Jesús Pérez (Lampa 1),Oscar Salas (Lampa 2),Pedro Alfaro (Ayudante),Rafael Soto (Regla),José Pinto (Regla y Nivel)
1,TP,TNC,TP,TC,TP,TP,TNC
2,TP,TNC,TP,TC,TP,TP,TNC
3,TP,TNC,TP,TC,TP,TP,TNC
4,TNC,TP,TP,TC,TP,TC,TNC
5,TNC,TP,TP,TC,TP,TC,TNC
6,TNC,TNC,TP,TP,TP,TC,TNC
7,TNC,TNC,TP,TP,TP,TNC,TNC
8,TNC,TNC,TP,TP,TP,TNC,TNC
9,TNC,TNC,TP,TC,TC,TNC,TNC
10,TNC,TNC,TP,TC,TC,TNC,TNC
11,TNC,TNC,TC,TC,TC,TNC,TC
12,TC,TNC,TC,TC,TC,TNC,TC
13,TC,TNC,TC,TC,TC,TP,TC
14,TC,TNC,TC,TC,TC,TP,TC
15,TC,TNC,TC,TC,TC,TP,TC
16,TC,TNC,TNC,TC,TC,TP,TNC
17,TC,TNC,TNC,TC,TC,TP,TNC
18,TC,TNC,TNC,TC,TC,TP,TNC
19,TC,TNC,TNC,TC,TC,TP,TNC
20,TC,TNC,TNC,TC,TP,TP,TC
21,TC,TP,TNC,TC,TP,TP,TC
22,TC,TP,TNC,TC,TP,TP,TC
23,TC,TP,TNC,TC,TP,TP,TC
24,TC,TP,TNC,TC,TP,TP,TC
25,TC,TP,TNC,TC,TP,TP,TC
26,TC,TC,TNC,TC,TP,TNC,TP
27,TC,TC,TNC,TC,TP,TNC,TP
28,TP,TP,TP,TP,TP,TNC,TP
29,TP,TP,TP,TP,TP,TNC,TP
30,TP,TP,TP,TP,TP,TP,TP
31,TNC,TC,TC,TC,TP,TP,TP
32,TNC,TNC,TNC,TNC,TP,TP,TP
33,TP,TNC,TNC,TNC,TNC,TNC,TNC
34,TP,TNC,TNC,TNC,TNC,TNC,TNC
35,TP,TNC,TNC,TNC,TNC,TNC,TNC
36,TC,TC,TC,TC,TNC,TNC,TNC
37,TC,TC,TC,TC,TNC,TNC,TNC
38,TC,TC,TNC,TC,TC,TC,TNC
39,TNC,TC,TNC,TC,TC,TC,TNC
40,TNC,TC,TNC,TC,TC,TC,TNC
41,TP,TP,TNC,TC,TC,TC,TNC
42,TP,TP,TNC,TC,TC,TC,TNC
43,TP,TP,TNC,TC,TNC,TP,TNC
44,TP,TP,TNC,TC,TNC,TP,TNC
45,TP,TP,TP,TC,TNC,TP,TNC
46,TNC,TP,TP,TC,TNC,TP,TP
47,TC,TP,TP,TC,TNC,TP,TP
48,TC,TNC,TC,TC,TNC,TP,TP
49,TC,TNC,TC,TC,TNC,TP,TP
50,TC,TC,TC,TC,TC,TC,TC
51,TC,TNC,TP,TC,TP,TP,TC
52,TC,TNC,TNC,TC,TP,TP,TC
53,TNC,TNC,TNC,TP,TP,TP,TC
54,TNC,TNC,TNC,TNC,TNC,TP,TC
55,TNC,TNC,TNC,TNC,TNC,TNC,TNC
56,TNC,TP,TP,TP,TC,TNC,TNC
57,TNC,TP,TNC,TP,TP,TNC,TNC
58,TNC,TP,TC,TC,TC,TNC,TNC
59,TNC,TP,TC,TC,TC,TP,TNC
60,TNC,TP,TNC,TP,TNC,TP,TC"""

        try:
            # Leer CSV string a DataFrame
            self.df_raw = pd.read_csv(io.StringIO(csv_data))

            # Verificación de integridad: Deben ser exactamente 60 minutos
            if len(self.df_raw) != 60:
                raise ValueError(f"Error en datos: Se esperaban 60 minutos, se encontraron {len(self.df_raw)}")

            logger.info("Datos crudos cargados exitosamente (60 registros).")

        except Exception as e:
            logger.error(f"Error crítico al procesar datos crudos: {e}")
            sys.exit(1)

    def calculate_metrics(self):
        """Calcula los totales de TP, TC, TNC agregando la data cruda."""
        if self.df_raw is None:
            return

        logger.info("Calculando métricas agregadas...")

        summary_list = []

        # Iterar sobre las columnas de trabajadores (excluyendo 'Minuto')
        worker_columns = [col for col in self.df_raw.columns if col != 'Minuto']

        for worker in worker_columns:
            # Contar ocurrencias de cada categoría
            counts = self.df_raw[worker].value_counts()
            tp = counts.get('TP', 0)
            tc = counts.get('TC', 0)
            tnc = counts.get('TNC', 0)

            # Validación pesimista por trabajador
            total_min = tp + tc + tnc
            if total_min != 60:
                logger.warning(f"¡Alerta! {worker} suma {total_min} minutos, no 60.")

            # Separar Nombre y Rol (Asumiendo formato "Nombre (Rol)")
            if '(' in worker:
                nombre = worker.split('(')[0].strip()
                rol = worker.split('(')[1].replace(')', '').strip()
            else:
                nombre = worker
                rol = "N/A"

            summary_list.append({
                'Nombre_Completo': worker,  # Usado para títulos de gráficos
                'Nombre': nombre,
                'Rol': rol,
                'TP': tp,
                'TC': tc,
                'TNC': tnc,
                'Total_Min': total_min,
                '% TP': (tp / total_min),  # Guardar como decimal para formato Excel
                '% TC': (tc / total_min),
                '% TNC': (tnc / total_min)
            })

        self.df_summary = pd.DataFrame(summary_list)

        # Totales de la cuadrilla
        total_tp_global = self.df_summary['TP'].sum()
        total_tc_global = self.df_summary['TC'].sum()
        total_tnc_global = self.df_summary['TNC'].sum()
        total_global = total_tp_global + total_tc_global + total_tnc_global

        self.stats_cuadrilla = {
            'TP': total_tp_global,
            'TC': total_tc_global,
            'TNC': total_tnc_global,
            '% TP': total_tp_global / total_global,
            '% TC': total_tc_global / total_global,
            '% TNC': total_tnc_global / total_global
        }

        logger.info(f"Métricas calculadas. Eficiencia global TP: {self.stats_cuadrilla['% TP'] * 100:.2f}%")

    def generate_excel_dashboard(self):
        """Genera el Excel con Pestaña de Data Cruda + Pestaña de Gráficos."""
        filename = os.path.join(self.output_folder, "Carta_Balance_Completa_Grupo4.xlsx")
        logger.info(f"Generando reporte Excel en: {filename}...")

        try:
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            workbook = writer.book

            # --- Formatos ---
            header_fmt = workbook.add_format(
                {'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#4F81BD', 'font_color': 'white',
                 'border': 1})
            cell_fmt = workbook.add_format({'align': 'center', 'border': 1})
            pct_fmt = workbook.add_format({'num_format': '0.0%', 'align': 'center', 'border': 1})

            # --- HOJA 1: REGISTRO DE CAMPO (Data Cruda) ---
            sheet_raw = 'Registro_Campo'
            self.df_raw.to_excel(writer, sheet_name=sheet_raw, index=False, startrow=1, header=False)
            ws_raw = writer.sheets[sheet_raw]

            # Escribir headers manualmente
            for col_num, value in enumerate(self.df_raw.columns.values):
                ws_raw.write(0, col_num, value, header_fmt)

            # Formato condicional para visualizar patrones en la data cruda
            # Verde para TP, Amarillo para TC, Rojo para TNC
            last_row = len(self.df_raw) + 1
            last_col = len(self.df_raw.columns) - 1
            ws_raw.conditional_format(1, 1, last_row, last_col,
                                      {'type': 'text', 'criteria': 'containing', 'value': 'TP',
                                       'format': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})})
            ws_raw.conditional_format(1, 1, last_row, last_col,
                                      {'type': 'text', 'criteria': 'containing', 'value': 'TC',
                                       'format': workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C5700'})})
            ws_raw.conditional_format(1, 1, last_row, last_col,
                                      {'type': 'text', 'criteria': 'containing', 'value': 'TNC',
                                       'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})})
            ws_raw.set_column(0, last_col, 15)

            # --- HOJA 2: ANÁLISIS ESTADÍSTICO ---
            sheet_stats = 'Analisis_Grafico'
            # Preparar DF para visualización (Columnas seleccionadas)
            display_cols = ['Nombre', 'Rol', 'TP', 'TC', 'TNC', '% TP', '% TC', '% TNC']
            self.df_summary[display_cols].to_excel(writer, sheet_name=sheet_stats, index=False, startrow=1,
                                                   header=False)
            ws_stats = writer.sheets[sheet_stats]

            # Headers
            for col_num, value in enumerate(display_cols):
                ws_stats.write(0, col_num, value, header_fmt)

            # Aplicar formato de porcentaje a las columnas correspondientes
            for row in range(1, len(self.df_summary) + 1):
                ws_stats.write(row, 5, self.df_summary.loc[row - 1, '% TP'], pct_fmt)
                ws_stats.write(row, 6, self.df_summary.loc[row - 1, '% TC'], pct_fmt)
                ws_stats.write(row, 7, self.df_summary.loc[row - 1, '% TNC'], pct_fmt)

            # --- GRÁFICOS ---

            # 1. Gráfico Total Cuadrilla
            chart_total = workbook.add_chart({'type': 'pie'})
            row_total = len(self.df_summary) + 3

            # Escribir tabla auxiliar para el gráfico total
            ws_stats.write(row_total, 0, "TOTAL CUADRILLA", header_fmt)
            ws_stats.write(row_total, 1, "Minutos", header_fmt)
            cats = ['TP', 'TC', 'TNC']
            vals = [self.stats_cuadrilla['TP'], self.stats_cuadrilla['TC'], self.stats_cuadrilla['TNC']]
            colors = ['#00B050', '#FFC000', '#FF0000']  # Verde, Amarillo, Rojo

            for i, (c, v) in enumerate(zip(cats, vals)):
                ws_stats.write(row_total + 1 + i, 0, c, cell_fmt)
                ws_stats.write(row_total + 1 + i, 1, v, cell_fmt)

            chart_total.add_series({
                'name': 'Eficiencia Global',
                'categories': [sheet_stats, row_total + 1, 0, row_total + 3, 0],
                'values': [sheet_stats, row_total + 1, 1, row_total + 3, 1],
                'data_labels': {'percentage': True, 'font': {'bold': True}},
                'points': [{'fill': {'color': color}} for color in colors],
            })
            chart_total.set_title({'name': f"Eficiencia Total (TP: {self.stats_cuadrilla['% TP'] * 100:.0f}%)"})
            ws_stats.insert_chart('K2', chart_total)

            # 2. Gráficos Individuales
            for idx, row in self.df_summary.iterrows():
                row_data = idx + 1
                chart = workbook.add_chart({'type': 'pie'})
                chart.add_series({
                    'name': row['Nombre_Completo'],
                    'categories': [sheet_stats, 0, 2, 0, 4],  # TP, TC, TNC headers
                    'values': [sheet_stats, row_data, 2, row_data, 4],  # TP, TC, TNC values
                    'data_labels': {'percentage': True},
                    'points': [{'fill': {'color': color}} for color in colors],
                })
                chart.set_title({'name': row['Nombre']})
                chart.set_size({'width': 250, 'height': 200})
                ws_stats.insert_chart(f'K{18 + (idx * 11)}', chart)

            ws_stats.set_column('A:B', 25)
            ws_stats.set_column('C:H', 10)

            writer.close()
            logger.info("Archivo Excel generado exitosamente con data cruda y gráficos.")

        except Exception as e:
            logger.error(f"Error crítico al generar Excel: {e}")

    def run(self):
        logger.info("Iniciando análisis completo GRUPO 4...")
        self.load_data()
        self.calculate_metrics()
        self.generate_excel_dashboard()
        logger.info("Proceso finalizado.")


if __name__ == "__main__":
    analyzer = CartaBalanceAnalyzer()
    analyzer.run()