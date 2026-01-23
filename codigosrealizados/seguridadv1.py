# ==============================================================================
# SCRIPT CIENTÍFICO PARA EL ANÁLISIS DE SINIESTRALIDAD LABORAL EN PERÚ (v1.1)
# AUTOR: Emanuel Edgar Ancco Guaygua (Asistido por Gemini AI)
# FECHA: 10 de Agosto de 2025
# CORRECCIÓN: Se ajustó la creación del DataFrame para evitar el error 'Length mismatch'.
# ==============================================================================

# --- FASE 1: PREPARACIÓN DEL ENTORNO Y CARGA DE DATOS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
import warnings

# Configuración del Entorno
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Carga de Datos (Listas originales)
regional_data_2023_list = [
    {"category": "AMAZONAS", "values": [0, 1, 0, 0, 2, 1, 0, 1, 0, 0, 0, 1]}, {"category": "ÁNCASH", "values": [116, 125, 141, 62, 107, 81, 141, 87, 98, 173, 101, 77]},
    {"category": "APURÍMAC", "values": [1, 0, 3, 10, 1, 2, 5, 7, 20, 11, 6, 15]}, {"category": "AREQUIPA", "values": [298, 224, 214, 185, 198, 125, 321, 336, 246, 285, 233, 195]},
    {"category": "AYACUCHO", "values": [4, 2, 3, 5, 5, 4, 6, 6, 5, 2, 0, 3]}, {"category": "CAJAMARCA", "values": [21, 11, 95, 14, 6, 13, 27, 27, 15, 11, 11, 12]},
    {"category": "CALLAO", "values": [248, 229, 295, 169, 174, 346, 520, 262, 249, 247, 263, 169]}, {"category": "CUSCO", "values": [19, 12, 19, 22, 28, 13, 23, 16, 19, 19, 41, 14]},
    {"category": "HUANCAVELICA", "values": [4, 4, 4, 9, 9, 11, 10, 17, 22, 18, 8, 4]}, {"category": "HUÁNUCO", "values": [4, 50, 26, 2, 10, 4, 8, 2, 43, 20, 5, 0]},
    {"category": "ICA", "values": [25, 17, 16, 14, 15, 19, 54, 23, 15, 32, 18, 40]}, {"category": "JUNÍN", "values": [14, 12, 16, 20, 35, 67, 34, 27, 43, 28, 52, 23]},
    {"category": "LA LIBERTAD", "values": [25, 44, 21, 18, 25, 25, 38, 22, 20, 15, 22, 32]}, {"category": "LAMBAYEQUE", "values": [48, 38, 14, 25, 1, 6, 290, 43, 50, 44, 49, 117]},
    {"category": "LIMA", "values": [2400, 2039, 2076, 1780, 1762, 2358, 4374, 2041, 1983, 2250, 2426, 1896]}, {"category": "LORETO", "values": [3, 3, 18, 11, 12, 6, 31, 39, 63, 31, 40, 24]},
    {"category": "MADRE DE DIOS", "values": [0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]}, {"category": "MOQUEGUA", "values": [8, 39, 30, 7, 65, 28, 46, 32, 41, 49, 27, 55]},
    {"category": "PASCO", "values": [30, 23, 14, 32, 24, 48, 45, 29, 42, 37, 21, 36]}, {"category": "PIURA", "values": [121, 132, 43, 44, 24, 38, 37, 9, 11, 31, 23, 3]},
    {"category": "PUNO", "values": [2, 7, 1, 1, 4, 2, 2, 10, 23, 9, 7, 9]}, {"category": "SAN MARTÍN", "values": [1, 7, 1, 20, 0, 1, 0, 2, 0, 0, 1, 0]},
    {"category": "TACNA", "values": [9, 13, 11, 0, 17, 24, 47, 19, 26, 19, 28, 6]}, {"category": "TUMBES", "values": [4, 11, 4, 3, 4, 5, 2, 6, 6, 3, 2, 2]},
    {"category": "UCAYALI", "values": [1, 1, 0, 3, 0, 1, 0, 2, 42, 14, 5, 11]}
]
regional_data_2024_list = [
    {"category": "AMAZONAS", "values": [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 2]}, {"category": "ÁNCASH", "values": [95, 157, 177, 183, 146, 194, 137, 166, 158, 167, 165, 138]},
    {"category": "APURÍMAC", "values": [10, 0, 17, 9, 15, 14, 3, 58, 2, 22, 3, 37]}, {"category": "AREQUIPA", "values": [291, 289, 374, 329, 386, 237, 345, 239, 256, 273, 293, 223]},
    {"category": "AYACUCHO", "values": [7, 2, 3, 6, 4, 3, 2, 4, 7, 7, 5, 5]}, {"category": "CAJAMARCA", "values": [15, 12, 5, 14, 3, 3, 6, 12, 24, 76, 38, 13]},
    {"category": "CALLAO", "values": [206, 215, 205, 268, 195, 165, 206, 210, 223, 249, 225, 190]}, {"category": "CUSCO", "values": [16, 17, 4, 17, 11, 22, 18, 10, 11, 20, 12, 17]},
    {"category": "HUANCAVELICA", "values": [9, 17, 8, 8, 7, 22, 11, 22, 11, 27, 7, 11]}, {"category": "HUÁNUCO", "values": [4, 11, 13, 7, 7, 2, 7, 6, 4, 14, 5, 10]},
    {"category": "ICA", "values": [25, 20, 18, 29, 40, 14, 21, 45, 14, 22, 31, 27]}, {"category": "JUNÍN", "values": [18, 39, 40, 33, 30, 38, 31, 19, 37, 42, 28, 22]},
    {"category": "LA LIBERTAD", "values": [19, 14, 37, 30, 21, 51, 27, 32, 61, 31, 37, 40]}, {"category": "LAMBAYEQUE", "values": [22, 83, 55, 73, 50, 13, 57, 53, 62, 52, 0, 3]},
    {"category": "LIMA", "values": [2010, 2006, 1788, 2152, 2076, 1935, 2074, 2249, 2397, 2605, 2353, 2641]}, {"category": "LORETO", "values": [27, 31, 37, 27, 31, 36, 17, 32, 18, 30, 25, 27]},
    {"category": "MADRE DE DIOS", "values": [2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]}, {"category": "MOQUEGUA", "values": [13, 31, 12, 34, 27, 18, 26, 38, 32, 26, 20, 17]},
    {"category": "PASCO", "values": [22, 25, 34, 30, 30, 41, 25, 28, 33, 34, 32, 33]}, {"category": "PIURA", "values": [26, 52, 24, 154, 43, 26, 25, 15, 27, 35, 42, 34]},
    {"category": "PUNO", "values": [8, 16, 7, 7, 7, 8, 12, 13, 13, 11, 10, 20]}, {"category": "SAN MARTÍN", "values": [0, 1, 0, 1, 0, 2, 4, 0, 0, 0, 0, 1]},
    {"category": "TACNA", "values": [34, 14, 17, 14, 12, 27, 19, 11, 20, 18, 11, 26]}, {"category": "TUMBES", "values": [4, 5, 5, 0, 3, 1, 0, 5, 3, 3, 3, 0]},
    {"category": "UCAYALI", "values": [21, 13, 15, 26, 7, 7, 13, 12, 7, 33, 38, 31]}
]
months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# --- CORRECCIÓN APLICADA AQUÍ ---
# Se transforma la lista de diccionarios en un diccionario simple
data_dict_2023 = {item['category']: item['values'] for item in regional_data_2023_list}
data_dict_2024 = {item['category']: item['values'] for item in regional_data_2024_list}

# Se crea el DataFrame usando .from_dict, que maneja la estructura correctamente
df_2023 = pd.DataFrame.from_dict(data_dict_2023, orient='index', columns=months)
df_2024 = pd.DataFrame.from_dict(data_dict_2024, orient='index', columns=months)

print("Datos cargados y DataFrame construido correctamente.")
print("\n" + "="*50 + "\n")

# --- FASE 2: ANÁLISIS ESTADÍSTICO INFERENCIAL - PRUEBA T ---
print("--- Ejecutando Fase 2: Prueba T de Student ---")
total_monthly_2023 = df_2023.sum(axis=0)
total_monthly_2024 = df_2024.sum(axis=0)

t_stat, p_value = ttest_ind(total_monthly_2023, total_monthly_2024, equal_var=False)

plt.figure(figsize=(14, 7))
sns.kdeplot(total_monthly_2023, label='Distribución de Accidentes Mensuales 2023', fill=True, color='blue')
sns.kdeplot(total_monthly_2024, label='Distribución de Accidentes Mensuales 2024', fill=True, color='red')
plt.title('Comparación de la Distribución de Accidentes Totales Mensuales (2023 vs 2024)', fontsize=16)
plt.xlabel('Número de Accidentes Mensuales', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.text(0.95, 0.90, f'Estadístico T = {t_stat:.2f}\nP-valor = {p_value:.3f}',
         ha='right', va='top', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
plt.savefig('t_test_distribution.png', dpi=300, bbox_inches='tight')
print("Gráfico de la prueba T guardado como 't_test_distribution.png'")

print("Resultados de la Prueba T de Student:")
if p_value < 0.05:
    print("Conclusión: Existe una diferencia estadísticamente significativa (p < 0.05).")
else:
    print("Conclusión: No hay evidencia de una diferencia estadísticamente significativa (p >= 0.05).")
print("\n" + "="*50 + "\n")

# --- FASE 3: MODELADO PREDICTIVO DE SERIES DE TIEMPO (SARIMA) ---
print("--- Ejecutando Fase 3: Modelo SARIMA ---")
national_timeseries_data = pd.concat([total_monthly_2023, total_monthly_2024])
index = pd.date_range(start='2023-01-01', periods=24, freq='MS')
national_timeseries = pd.Series(national_timeseries_data.values, index=index)

decomposition = sm.tsa.seasonal_decompose(national_timeseries, model='additive')
fig = decomposition.plot()
plt.suptitle('Descomposición de la Serie Temporal de Accidentes Nacionales', y=1.02)
plt.savefig('series_decomposition.png', dpi=300, bbox_inches='tight')
print("Gráfico de descomposición guardado como 'series_decomposition.png'")

model = sm.tsa.SARIMAX(national_timeseries, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)
print("\nResumen del Modelo SARIMA:")
print(results.summary())

forecast_steps = 12
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=national_timeseries.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
confidence_intervals = forecast.conf_int()
confidence_intervals.index = forecast_index

plt.figure(figsize=(15, 8))
plt.plot(national_timeseries, label='Datos Históricos (Observados)')
plt.plot(forecast_series, label='Pronóstico', color='red')
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.5, label='Intervalo de Confianza (95%)')
plt.title('Pronóstico de Accidentes Nacionales para los Próximos 12 Meses', fontsize=16)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Número de Accidentes', fontsize=12)
plt.legend()
plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
print("Gráfico de pronóstico guardado como 'forecast_plot.png'")
print("\n" + "="*50 + "\n")
print("Análisis completo ejecutado.")