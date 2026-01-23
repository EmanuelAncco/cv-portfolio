# =================================================================
# SIMULACIÓN PROBABILÍSTICA DE GASODUCTOS (SMC)
# Autor: Emanuel Edgar Ancco Guaygua (Asistido por Gemini)
# Fecha: 28 de Octubre de 2025
#
# Descripción:
# Este script implementa la metodología de Simulación de Monte Carlo (SMC)
# descrita en el documento LaTeX 'tesis_metodologia_v3.tex'.
#
# Evoluciona el modelo determinista para cumplir los Objetivos 3, 4 y 5:
# 1. Caracteriza incertidumbres (Obj. 3) usando scipy.stats.
# 2. Ejecuta una SMC vectorizada (Obj. 4) para propagar incertidumbres.
# 3. Calcula la Probabilidad de Falla (Pf) y el Índice Beta (Obj. 5).
# =================================================================

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import datetime
from tqdm import tqdm  # Para una barra de progreso

# --- 1. CONFIGURACIÓN INICIAL (LOGGING Y RESULTADOS) ---

# Configuración de la carpeta de resultados única (Mejores prácticas de ML)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULTS_DIR = f"Resultados_SMC_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuración del logging
LOG_FILE = os.path.join(RESULTS_DIR, "simulacion.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logging.info("--- Iniciando Simulación de Monte Carlo ---")
logging.info(f"Resultados se guardarán en: {RESULTS_DIR}")

# --- 2. PARÁMETROS DEL MODELO (Basado en tesis_metodologia_v3.tex) ---

# --- Parámetros Deterministas (Constantes) ---
# Geometría de la Tubería (API 5L X65, 30in)
D = 0.762  # Diámetro exterior (m)
t = 0.0095  # Espesor de pared (m)

# Propiedades del Material (Acero)
E_p = 2.07e11  # Módulo de Young (Pa)
NU_ACERO = 0.3  # Coef. de Poisson
A_p = np.pi * (D - t) * t  # Área de la sección (m^2)
I_p = (np.pi / 64) * (D ** 4 - (D - 2 * t) ** 4)  # Momento de Inercia (m^4)

# Cargas Operacionales y Sísmicas (Constantes)
SIGMA_H = 150e6  # Esfuerzo tangencial por presión (Pa) - Supuesto
f = 1.0  # Frecuencia dominante del sismo (Hz)
NU_SUELO = 0.3  # Coef. de Poisson del suelo

logging.info("Parámetros deterministas cargados:")
logging.info(f"  Tubería: D={D} m, t={t} m")
logging.info(f"  Material: E_p={E_p:.2e} Pa, A_p={A_p:.4f} m^2")
logging.info(f"  Cargas: sigma_h={SIGMA_H / 1e6:.1f} MPa, f={f} Hz")

# --- Parámetros Probabilísticos (Variables Aleatorias) ---
N_SIMULATIONS = 1_000_000  # Número de iteraciones de Monte Carlo

# Definición de variables (Media 'mu' y Coef. de Variación 'COV')
# Estos son los datos de entrada para el "Caso Surco" (Obj. 3)
VAR_SPECS = {
    'pgv': {'dist': 'lognorm', 'mu': 0.5, 'cov': 0.30},
    'vs': {'dist': 'lognorm', 'mu': 300.0, 'cov': 0.25},
    'es': {'dist': 'lognorm', 'mu': 50.0e6, 'cov': 0.20},
    'kt': {'dist': 'lognorm', 'mu': 1.8, 'cov': 0.15},
    'sy': {'dist': 'lognorm', 'mu': 448.2e6, 'cov': 0.10},
    'theta': {'dist': 'uniform', 'min': 0, 'max': np.pi / 2}  # 0 a 90 grados
}

logging.info(f"Parámetros probabilísticos (N={N_SIMULATIONS}):")
for key, val in VAR_SPECS.items():
    logging.info(f"  {key}: {val}")


# --- 3. FUNCIONES AUXILIARES ---

def get_lognorm_params(mu, cov):
    """
    Convierte (media, cov) a los parámetros (shape, scale)
    requeridos por scipy.stats.lognorm.
    """
    sigma_ln_sq = np.log(cov ** 2 + 1)
    s = np.sqrt(sigma_ln_sq)  # s (shape)
    scale = mu / np.exp(sigma_ln_sq / 2)  # scale
    return s, scale


def get_uniform_params(min_val, max_val):
    """
    Convierte (min, max) a los parámetros (loc, scale)
    requeridos por scipy.stats.uniform.
    """
    loc = min_val
    scale = max_val - min_val
    return loc, scale


# --- 4. EJECUCIÓN DE LA SIMULACIÓN DE MONTE CARLO (OBJ. 4) ---

def run_simulation(n_samples, specs):
    """
    Ejecuta la SMC de forma vectorizada.
    """
    logging.info("Iniciando la generación de muestras aleatorias...")

    # --- Paso 1: Generar Muestras (Vectorizado) ---
    samples = {}

    # PGV
    s, scale = get_lognorm_params(specs['pgv']['mu'], specs['pgv']['cov'])
    samples['pgv'] = stats.lognorm.rvs(s=s, scale=scale, size=n_samples)

    # Vs
    s, scale = get_lognorm_params(specs['vs']['mu'], specs['vs']['cov'])
    samples['vs'] = stats.lognorm.rvs(s=s, scale=scale, size=n_samples)

    # Es
    s, scale = get_lognorm_params(specs['es']['mu'], specs['es']['cov'])
    samples['es'] = stats.lognorm.rvs(s=s, scale=scale, size=n_samples)

    # Kt
    s, scale = get_lognorm_params(specs['kt']['mu'], specs['kt']['cov'])
    samples['kt'] = stats.lognorm.rvs(s=s, scale=scale, size=n_samples)

    # Sy (Resistencia)
    s, scale = get_lognorm_params(specs['sy']['mu'], specs['sy']['cov'])
    samples['sy'] = stats.lognorm.rvs(s=s, scale=scale, size=n_samples)

    # Theta
    loc, scale = get_uniform_params(specs['theta']['min'], specs['theta']['max'])
    samples['theta'] = stats.uniform.rvs(loc=loc, scale=scale, size=n_samples)

    logging.info("Muestreo completado.")
    logging.info("Iniciando modelo determinista vectorizado...")

    # --- Paso 2: Modelo Determinista (Vectorizado) (Obj. 1 y 2) ---

    # Calcular propiedades del suelo (vectorizadas)
    G_s = samples['es'] / (2 * (1 + NU_SUELO))
    k_ax = np.pi * D * G_s  # Rigidez axial del suelo

    # Calcular solicitación sísmica (vectorizada)
    # Deformación del suelo (TGD)
    epsilon_g = (samples['pgv'] / samples['vs']) * (np.cos(samples['theta']) ** 2)

    # Número de onda (vectorizado, depende de Vs)
    k_w = (2 * np.pi * f) / samples['vs']

    # Calcular respuesta de la tubería (Ec. 14, vectorizada)
    # (Ignoramos flexión, demostrado despreciable)
    termino_axial = (E_p * A_p * k_w ** 2) / k_ax
    epsilon_p = epsilon_g * (1 / (1 + termino_axial))

    # Esfuerzo axial nominal
    sigma_ax = E_p * epsilon_p

    # --- Paso 3: Calcular Solicitación (S) (Vectorizado) (Obj. 2) ---

    # Esfuerzo longitudinal en la soldadura (con SCF)
    sigma_x_weld = samples['kt'] * sigma_ax

    # Esfuerzo de Von Mises en la soldadura (Ec. 19)
    # sigma_y_weld es la constante SIGMA_H
    S_solicitacion = np.sqrt(
        sigma_x_weld ** 2 + SIGMA_H ** 2 - (sigma_x_weld * SIGMA_H)
    )

    # --- Paso 4: Evaluar Falla (Obj. 5) ---

    R_resistencia = samples['sy']

    # Función de Estado Límite (LSF)
    g_lsf = R_resistencia - S_solicitacion

    # Contar fallas
    n_fallas = np.sum(g_lsf <= 0)

    logging.info("Cálculos vectorizados completados.")

    # Devolver resultados para análisis
    return n_fallas, S_solicitacion, R_resistencia, g_lsf


# --- 5. ANÁLISIS DE RESULTADOS Y VISUALIZACIÓN ---

def analizar_y_guardar(n_fallas, n_total, S, R, g):
    """
    Calcula Pf, Beta y guarda los artefactos de la simulación.
    """
    logging.info("--- Análisis de Resultados ---")

    # --- Cálculo de Pf y Beta (Obj. 5) ---
    if n_fallas == 0:
        pf = 1 / n_total  # Estimación conservadora si N=0
        logging.warning(f"No se observaron fallas en {n_total} simulaciones.")
        logging.warning(f"Usando Pf estimada = 1/{n_total} = {pf:.2e}")
    else:
        pf = n_fallas / n_total

    try:
        # ppf es la inversa de la CDF (Percent Point Function)
        beta = -stats.norm.ppf(pf)
        logging.info(f"Número de Fallas (Nf): {n_fallas}")
        logging.info(f"Probabilidad de Falla (Pf): {pf:.4e}")
        logging.info(f"Índice de Confiabilidad (Beta): {beta:.4f}")
    except Exception as e:
        logging.error(f"Error al calcular Beta (Pf={pf}): {e}")
        beta = np.nan

    # --- Guardar Datos Crudos (CSV) ---
    logging.info("Guardando resultados crudos en CSV...")
    try:
        df_results = pd.DataFrame({
            'Solicitacion_S_MPa': S / 1e6,
            'Resistencia_R_MPa': R / 1e6,
            'LSF_g_MPa': g / 1e6
        })
        csv_path = os.path.join(RESULTS_DIR, "resultados_completos.csv")
        df_results.to_csv(csv_path, index=False)
        logging.info(f"Resultados guardados en {csv_path}")
    except Exception as e:
        logging.error(f"No se pudo guardar el CSV: {e}")

    # --- Generar Gráfico de Distribución (Evidencia Clave) ---
    logging.info("Generando gráfico de distribución (R vs S)...")
    try:
        plt.figure(figsize=(12, 7))
        plt.hist(S / 1e6, bins=100, alpha=0.7, label="Solicitación (S)", density=True, color="red")
        plt.hist(R / 1e6, bins=100, alpha=0.7, label="Resistencia (R)", density=True, color="blue")
        plt.title("Distribución de Resistencia (R) vs. Solicitación (S)", fontsize=16)
        plt.xlabel("Esfuerzo (MPa)", fontsize=12)
        plt.ylabel("Densidad de Probabilidad", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Resumen en el gráfico
        stats_text = (
            f"Simulaciones: {n_total:,}\n"
            f"Fallos (S > R): {n_fallas:,}\n"
            f"$P_f = {pf:.2e}$\n"
            f"$\\beta = {beta:.3f}$\n\n"
            f"E[S] = {np.mean(S / 1e6):.2f} MPa\n"
            f"E[R] = {np.mean(R / 1e6):.2f} MPa"
        )
        plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=11, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

        plot_path = os.path.join(RESULTS_DIR, "distribucion_R_vs_S.png")
        plt.savefig(plot_path)
        logging.info(f"Gráfico guardado en {plot_path}")
        # plt.show() # Descomentar si se desea visualización interactiva

    except Exception as e:
        logging.error(f"No se pudo generar el gráfico: {e}")


# --- 6. PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    try:
        # Ejecutar la simulación
        # Se usa tqdm para envolver la función y mostrar progreso (no es necesario)
        # pero es útil saber que el cálculo está ocurriendo.
        logging.info("Iniciando cálculos... (Esto puede tardar)")

        # La función en sí misma es vectorizada, por lo que se ejecuta "de una vez"
        # tqdm aquí es más un indicador de que el paso está activo.
        n_f, S_vec, R_vec, g_vec = run_simulation(N_SIMULATIONS, VAR_SPECS)

        # Analizar y guardar los resultados
        analizar_y_guardar(n_f, N_SIMULATIONS, S_vec, R_vec, g_vec)

        logging.info("--- Simulación de Monte Carlo Completada Exitosamente ---")

    except Exception as e:
        logging.error("--- La Simulación ha Fallado ---")
        logging.error(f"Error: {e}", exc_info=True)
    