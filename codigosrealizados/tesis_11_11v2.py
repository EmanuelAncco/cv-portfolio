"""
================================================================================
SIMULADOR AVANZADO DE ANÁLISIS PROBABILÍSTICO DE TUBERÍAS ENTERRADAS
Tesista: Emanuel Ancco
Universidad: [Tu Universidad]
Fecha: 2025-11-11

FUNCIONALIDADES PRINCIPALES:
1. Análisis Determinista SSI (Soil-Structure Interaction)
2. Simulación de Monte Carlo
3. Curvas de Fragilidad (Múltiples Estados Límite)
4. Análisis FORM (First Order Reliability Method)
5. Confiabilidad Dependiente del Tiempo
6. Análisis Multi-Escenario con Estadística Inferencial
7. Métricas Avanzadas (Ductilidad, Energía, Daño)
8. Visualizaciones 3D y Avanzadas
9. Exportación Completa (CSV, Excel, Reportes PDF)
================================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from scipy.stats import lognorm, uniform, norm, spearmanr, chi2, f_oneway
from scipy.optimize import minimize
import pandas as pd
import logging
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- CONFIGURACIÓN DE LOGGING ---
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ==============================================================================
# --- DEFINICIÓN DE CASOS DE ESTUDIO ---
# ==============================================================================

SCENARIOS = {
    "Caso Surco (Tipo C)": {
        "PGV": {"dist": "lognorm", "mean": 0.5, "cov": 0.30},
        "Vs": {"dist": "lognorm", "mean": 400.0, "cov": 0.25},
        "Es": {"dist": "lognorm", "mean": 50e6, "cov": 0.20},
        "Kt": {"dist": "lognorm", "mean": 1.8, "cov": 0.15},
        "Sy": {"dist": "lognorm", "mean": 448.2e6, "cov": 0.10},
        "theta": {"dist": "uniform", "min": 0.0, "max": 90.0}
    },
    "Caso S.J. Lurigancho (Tipo C)": {
        "PGV": {"dist": "lognorm", "mean": 0.4, "cov": 0.30},
        "Vs": {"dist": "lognorm", "mean": 400.0, "cov": 0.25},
        "Es": {"dist": "lognorm", "mean": 50e6, "cov": 0.20},
        "Kt": {"dist": "lognorm", "mean": 1.8, "cov": 0.15},
        "Sy": {"dist": "lognorm", "mean": 448.2e6, "cov": 0.10},
        "theta": {"dist": "uniform", "min": 0.0, "max": 90.0}
    },
    "Caso Villa El Salvador (Tipo D)": {
        "PGV": {"dist": "lognorm", "mean": 0.6, "cov": 0.35},
        "Vs": {"dist": "lognorm", "mean": 250.0, "cov": 0.30},
        "Es": {"dist": "lognorm", "mean": 30e6, "cov": 0.25},
        "Kt": {"dist": "lognorm", "mean": 1.8, "cov": 0.15},
        "Sy": {"dist": "lognorm", "mean": 448.2e6, "cov": 0.10},
        "theta": {"dist": "uniform", "min": 0.0, "max": 90.0}
    }
}

# Parámetros deterministas constantes
DETERMINISTIC_CONSTANTS = {
    "D": 0.762,  # Diámetro Exterior (m)
    "t": 0.0095,  # Espesor de Pared (m)
    "Ep": 207e9,  # Módulo de Young Acero (Pa)
    "nu_s": 0.3,  # Coef. Poisson Suelo
    "sigma_h": 150e6,  # Esfuerzo Tangencial (Pa)
    "f": 1.0  # Frecuencia Sismo (Hz)
}
DETERMINISTIC_CONSTANTS["Ap"] = np.pi * (DETERMINISTIC_CONSTANTS["D"] - DETERMINISTIC_CONSTANTS["t"]) * \
                                DETERMINISTIC_CONSTANTS["t"]
DETERMINISTIC_CONSTANTS["EpAp"] = DETERMINISTIC_CONSTANTS["Ep"] * DETERMINISTIC_CONSTANTS["Ap"]

# Estados Límite para Curvas de Fragilidad
LIMIT_STATES = {
    "LS1_Elástico": {"threshold": 0.67, "description": "Inicio de fluencia (S/Sy = 0.67)"},
    "LS2_Plastificación": {"threshold": 1.0, "description": "Fluencia completa (S/Sy = 1.0)"},
    "LS3_Colapso": {"threshold": 1.5, "description": "Colapso incipiente (S/Sy = 1.5)"}
}


# ==============================================================================
# --- FUNCIONES AUXILIARES PARA DISTRIBUCIONES ---
# ==============================================================================

def get_lognorm_params(mean, cov):
    """Calcula los parámetros (sigma, scale) para scipy.stats.lognorm"""
    sigma_y_sq = np.log(cov ** 2 + 1)
    mu_y = np.log(mean) - 0.5 * sigma_y_sq
    sigma_y = np.sqrt(sigma_y_sq)
    scale = np.exp(mu_y)
    return sigma_y, scale


def get_uniform_params(min_val, max_val):
    """Calcula los parámetros (loc, scale) para scipy.stats.uniform"""
    loc = min_val
    scale = max_val - min_val
    return loc, scale


# ==============================================================================
# --- MOTOR DE CÁLCULO DETERMINISTA ---
# ==============================================================================

def calculate_single_point_fs(params):
    """
    Calcula el FS determinista para un solo punto.
    Retorna diccionario con todos los resultados intermedios.
    """
    try:
        # Extraer constantes
        EpAp = DETERMINISTIC_CONSTANTS["EpAp"]
        sigma_h = DETERMINISTIC_CONSTANTS["sigma_h"]
        D = DETERMINISTIC_CONSTANTS["D"]
        nu_s = DETERMINISTIC_CONSTANTS["nu_s"]
        f = DETERMINISTIC_CONSTANTS["f"]
        Ep = DETERMINISTIC_CONSTANTS["Ep"]

        # Extraer variables de entrada
        PGV = params["PGV"]
        Vs = params["Vs"]
        Es = params["Es"]
        Kt = params["Kt"]
        Sy = params["Sy"]
        theta_deg = params["theta"]
        theta_rad = np.radians(theta_deg)

        # Modelo Determinista SSI
        Gs = Es / (2 * (1 + nu_s))
        k_ax = np.pi * D * Gs
        epsilon_g_ax = (PGV / Vs) * (np.cos(theta_rad) ** 2)
        k_w = (2 * np.pi * f) / Vs

        termino_rigidez = (EpAp * k_w ** 2) / k_ax
        epsilon_p_ax = epsilon_g_ax * (1 / (1 + termino_rigidez))
        sigma_ax = Ep * epsilon_p_ax

        # Concentración de Esfuerzos
        sigma_x_weld = Kt * sigma_ax

        # Von Mises
        S = np.sqrt(sigma_x_weld ** 2 + sigma_h ** 2 - (sigma_x_weld * sigma_h))

        R = Sy
        FS = R / S if S > 0 else np.inf

        # Métricas adicionales
        ratio_S_Sy = S / Sy
        ductility_demand = epsilon_p_ax / (Sy / Ep)  # Aproximación

        return {
            "theta": theta_deg,
            "epsilon_g_ax": epsilon_g_ax,
            "termino_axial": termino_rigidez,
            "epsilon_p_ax": epsilon_p_ax,
            "sigma_ax": sigma_ax / 1e6,
            "sigma_x_weld": sigma_x_weld / 1e6,
            "S_MPa": S / 1e6,
            "S": S,
            "FS": FS,
            "ratio_S_Sy": ratio_S_Sy,
            "ductility": ductility_demand
        }
    except Exception as e:
        logging.error(f"Error en cálculo determinista: {e}")
        return None


def calculate_deterministic_table_data(scenario_params):
    """Genera datos para tabla determinista variando theta"""
    table_data = []
    base_params = {
        "PGV": scenario_params["PGV"]["mean"],
        "Vs": scenario_params["Vs"]["mean"],
        "Es": scenario_params["Es"]["mean"],
        "Kt": scenario_params["Kt"]["mean"],
        "Sy": scenario_params["Sy"]["mean"],
    }

    for theta_deg in range(0, 91, 15):
        params = base_params.copy()
        params["theta"] = theta_deg
        results = calculate_single_point_fs(params)
        if results:
            table_data.append(results)
    return table_data


# ==============================================================================
# --- SIMULACIÓN DE MONTE CARLO ---
# ==============================================================================

def run_simulation(scenario, N, store_samples=False):
    """
    Ejecuta la Simulación de Monte Carlo.
    Si store_samples=True, guarda las muestras de entrada para análisis FORM.
    """
    logging.info(f"Iniciando Simulación de Monte Carlo: {scenario['name']}")
    logging.info(f"Número de simulaciones (N): {N}")

    start_time = time.time()
    params = scenario['params']

    # Generar muestras
    logging.info("Generando muestras aleatorias...")

    # Resistencia (R)
    sigma_sy, scale_sy = get_lognorm_params(params["Sy"]["mean"], params["Sy"]["cov"])
    R_results = lognorm.rvs(s=sigma_sy, scale=scale_sy, size=N)

    # Variables de Solicitación
    sigma_pgv, scale_pgv = get_lognorm_params(params["PGV"]["mean"], params["PGV"]["cov"])
    PGV_samples = lognorm.rvs(s=sigma_pgv, scale=scale_pgv, size=N)

    sigma_vs, scale_vs = get_lognorm_params(params["Vs"]["mean"], params["Vs"]["cov"])
    Vs_samples = lognorm.rvs(s=sigma_vs, scale=scale_vs, size=N)

    sigma_es, scale_es = get_lognorm_params(params["Es"]["mean"], params["Es"]["cov"])
    Es_samples = lognorm.rvs(s=sigma_es, scale=scale_es, size=N)

    sigma_kt, scale_kt = get_lognorm_params(params["Kt"]["mean"], params["Kt"]["cov"])
    Kt_samples = lognorm.rvs(s=sigma_kt, scale=scale_kt, size=N)

    loc_th, scale_th = get_uniform_params(params["theta"]["min"], params["theta"]["max"])
    theta_samples_deg = uniform.rvs(loc=loc_th, scale=scale_th, size=N)
    theta_samples_rad = np.radians(theta_samples_deg)

    # Guardar inputs
    inputs = {
        "PGV": PGV_samples, "Vs": Vs_samples, "Es": Es_samples,
        "Kt": Kt_samples, "theta": theta_samples_deg, "Sy": R_results
    }

    # Ejecutar Modelo Determinista (Vectorizado)
    logging.info("Ejecutando modelo determinista vectorizado...")

    EpAp = DETERMINISTIC_CONSTANTS["EpAp"]
    sigma_h = DETERMINISTIC_CONSTANTS["sigma_h"]
    D = DETERMINISTIC_CONSTANTS["D"]
    nu_s = DETERMINISTIC_CONSTANTS["nu_s"]
    f = DETERMINISTIC_CONSTANTS["f"]
    Ep = DETERMINISTIC_CONSTANTS["Ep"]

    Gs_samples = Es_samples / (2 * (1 + nu_s))
    k_ax_samples = np.pi * D * Gs_samples
    epsilon_g_ax_samples = (PGV_samples / Vs_samples) * (np.cos(theta_samples_rad) ** 2)
    k_w_samples = (2 * np.pi * f) / Vs_samples
    termino_rigidez = (EpAp * k_w_samples ** 2) / k_ax_samples
    epsilon_p_ax_samples = epsilon_g_ax_samples * (1 / (1 + termino_rigidez))
    sigma_ax_samples = Ep * epsilon_p_ax_samples
    sigma_x_weld_samples = Kt_samples * sigma_ax_samples

    # Von Mises
    S_results = np.sqrt(
        sigma_x_weld_samples ** 2 +
        sigma_h ** 2 -
        (sigma_x_weld_samples * sigma_h)
    )

    # Métricas adicionales
    ratio_S_Sy = S_results / R_results
    ductility_samples = epsilon_p_ax_samples / (R_results / Ep)

    # Evaluación de Falla
    logging.info("Evaluando probabilidad de falla...")
    fallas = S_results >= R_results
    Nf = np.sum(fallas)
    Pf = Nf / N
    beta = -norm.ppf(Pf) if 0 < Pf < 1 else (6 if Pf == 0 else -6)

    # Sensibilidad Probabilística
    logging.info("Calculando sensibilidad probabilística...")
    sensitivity = {}
    try:
        sensitivity['PGV'] = spearmanr(PGV_samples, S_results)[0]
        sensitivity['V_s'] = spearmanr(Vs_samples, S_results)[0]
        sensitivity['E_s'] = spearmanr(Es_samples, S_results)[0]
        sensitivity['K_t'] = spearmanr(Kt_samples, S_results)[0]
        sensitivity['theta'] = spearmanr(theta_samples_deg, S_results)[0]
    except Exception as e:
        logging.warning(f"No se pudo calcular la sensibilidad: {e}")

    end_time = time.time()
    logging.info(f"Simulación completada en {end_time - start_time:.2f} segundos.")

    results = {
        "Pf": Pf,
        "beta": beta,
        "Nf": Nf,
        "N": N,
        "S_mean": np.mean(S_results),
        "R_mean": np.mean(R_results),
        "S_std": np.std(S_results),
        "R_std": np.std(R_results),
        "ratio_mean": np.mean(ratio_S_Sy),
        "ratio_std": np.std(ratio_S_Sy),
        "ductility_mean": np.mean(ductility_samples),
        "ductility_std": np.std(ductility_samples)
    }

    if store_samples:
        return results, sensitivity, S_results, R_results, inputs, ratio_S_Sy, ductility_samples
    else:
        return results, sensitivity, S_results, R_results


# ==============================================================================
# --- CURVAS DE FRAGILIDAD ---
# ==============================================================================

def calculate_fragility_curves(scenario, im_range, N=100000):
    """
    Calcula curvas de fragilidad para múltiples estados límite.
    IM (Intensity Measure): PGV
    """
    logging.info(f"Calculando curvas de fragilidad para: {scenario['name']}")

    params = scenario['params']
    fragility_data = {ls: [] for ls in LIMIT_STATES.keys()}

    for pgv_val in im_range:
        # Generar muestras con PGV fijo
        sigma_vs, scale_vs = get_lognorm_params(params["Vs"]["mean"], params["Vs"]["cov"])
        Vs_samples = lognorm.rvs(s=sigma_vs, scale=scale_vs, size=N)

        sigma_es, scale_es = get_lognorm_params(params["Es"]["mean"], params["Es"]["cov"])
        Es_samples = lognorm.rvs(s=sigma_es, scale=scale_es, size=N)

        sigma_kt, scale_kt = get_lognorm_params(params["Kt"]["mean"], params["Kt"]["cov"])
        Kt_samples = lognorm.rvs(s=sigma_kt, scale=scale_kt, size=N)

        sigma_sy, scale_sy = get_lognorm_params(params["Sy"]["mean"], params["Sy"]["cov"])
        Sy_samples = lognorm.rvs(s=sigma_sy, scale=scale_sy, size=N)

        loc_th, scale_th = get_uniform_params(params["theta"]["min"], params["theta"]["max"])
        theta_samples_deg = uniform.rvs(loc=loc_th, scale=scale_th, size=N)
        theta_samples_rad = np.radians(theta_samples_deg)

        # Cálculo vectorizado
        EpAp = DETERMINISTIC_CONSTANTS["EpAp"]
        sigma_h = DETERMINISTIC_CONSTANTS["sigma_h"]
        D = DETERMINISTIC_CONSTANTS["D"]
        nu_s = DETERMINISTIC_CONSTANTS["nu_s"]
        f = DETERMINISTIC_CONSTANTS["f"]
        Ep = DETERMINISTIC_CONSTANTS["Ep"]

        Gs_samples = Es_samples / (2 * (1 + nu_s))
        k_ax_samples = np.pi * D * Gs_samples
        epsilon_g_ax_samples = (pgv_val / Vs_samples) * (np.cos(theta_samples_rad) ** 2)
        k_w_samples = (2 * np.pi * f) / Vs_samples
        termino_rigidez = (EpAp * k_w_samples ** 2) / k_ax_samples
        epsilon_p_ax_samples = epsilon_g_ax_samples * (1 / (1 + termino_rigidez))
        sigma_ax_samples = Ep * epsilon_p_ax_samples
        sigma_x_weld_samples = Kt_samples * sigma_ax_samples

        S_results = np.sqrt(
            sigma_x_weld_samples ** 2 +
            sigma_h ** 2 -
            (sigma_x_weld_samples * sigma_h)
        )

        ratio_S_Sy = S_results / Sy_samples

        # Calcular probabilidad de excedencia para cada estado límite
        for ls_name, ls_data in LIMIT_STATES.items():
            threshold = ls_data["threshold"]
            exceedance = np.sum(ratio_S_Sy >= threshold) / N
            fragility_data[ls_name].append(exceedance)

    logging.info("Curvas de fragilidad calculadas.")
    return fragility_data


# ==============================================================================
# --- ANÁLISIS FORM (First Order Reliability Method) ---
# ==============================================================================

def perform_form_analysis(scenario):
    """
    Realiza análisis FORM para encontrar el punto de diseño.
    Simplificado para 5 variables principales.
    """
    logging.info(f"Iniciando análisis FORM para: {scenario['name']}")

    params = scenario['params']

    # Función de performance en espacio estándar normal
    def limit_state_function(u):
        """
        Función de estado límite: g(u) = R - S
        u: vector en espacio normal estándar
        """
        # Transformar de espacio normal a espacio físico
        # u = [u_PGV, u_Vs, u_Es, u_Kt, u_Sy]

        # PGV
        sigma_pgv, scale_pgv = get_lognorm_params(params["PGV"]["mean"], params["PGV"]["cov"])
        PGV = lognorm.ppf(norm.cdf(u[0]), s=sigma_pgv, scale=scale_pgv)

        # Vs
        sigma_vs, scale_vs = get_lognorm_params(params["Vs"]["mean"], params["Vs"]["cov"])
        Vs = lognorm.ppf(norm.cdf(u[1]), s=sigma_vs, scale=scale_vs)

        # Es
        sigma_es, scale_es = get_lognorm_params(params["Es"]["mean"], params["Es"]["cov"])
        Es = lognorm.ppf(norm.cdf(u[2]), s=sigma_es, scale=scale_es)

        # Kt
        sigma_kt, scale_kt = get_lognorm_params(params["Kt"]["mean"], params["Kt"]["cov"])
        Kt = lognorm.ppf(norm.cdf(u[3]), s=sigma_kt, scale=scale_kt)

        # Sy
        sigma_sy, scale_sy = get_lognorm_params(params["Sy"]["mean"], params["Sy"]["cov"])
        Sy = lognorm.ppf(norm.cdf(u[4]), s=sigma_sy, scale=scale_sy)

        # theta fijo (media)
        theta_deg = (params["theta"]["min"] + params["theta"]["max"]) / 2
        theta_rad = np.radians(theta_deg)

        # Calcular S
        EpAp = DETERMINISTIC_CONSTANTS["EpAp"]
        sigma_h = DETERMINISTIC_CONSTANTS["sigma_h"]
        D = DETERMINISTIC_CONSTANTS["D"]
        nu_s = DETERMINISTIC_CONSTANTS["nu_s"]
        f = DETERMINISTIC_CONSTANTS["f"]
        Ep = DETERMINISTIC_CONSTANTS["Ep"]

        Gs = Es / (2 * (1 + nu_s))
        k_ax = np.pi * D * Gs
        epsilon_g_ax = (PGV / Vs) * (np.cos(theta_rad) ** 2)
        k_w = (2 * np.pi * f) / Vs
        termino_rigidez = (EpAp * k_w ** 2) / k_ax
        epsilon_p_ax = epsilon_g_ax * (1 / (1 + termino_rigidez))
        sigma_ax = Ep * epsilon_p_ax
        sigma_x_weld = Kt * sigma_ax
        S = np.sqrt(sigma_x_weld ** 2 + sigma_h ** 2 - (sigma_x_weld * sigma_h))

        # g(u) = R - S
        g = Sy - S
        return g

    # Función objetivo para FORM: minimizar ||u|| sujeto a g(u) = 0
    def objective(u):
        return np.linalg.norm(u)

    def constraint(u):
        return limit_state_function(u)

    # Punto inicial (origen)
    u0 = np.zeros(5)

    # Optimización
    try:
        result = minimize(
            objective,
            u0,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': constraint},
            options={'maxiter': 1000}
        )

        if result.success:
            u_star = result.x
            beta_form = np.linalg.norm(u_star)

            # Calcular factores alpha (importancia)
            alpha = -u_star / beta_form

            logging.info(f"FORM exitoso: Beta = {beta_form:.4f}")

            return {
                "success": True,
                "beta_form": beta_form,
                "design_point_u": u_star,
                "alpha_factors": alpha,
                "iterations": result.nit
            }
        else:
            logging.warning("FORM no convergió")
            return {"success": False}

    except Exception as e:
        logging.error(f"Error en FORM: {e}")
        return {"success": False}


# ==============================================================================
# --- CONFIABILIDAD DEPENDIENTE DEL TIEMPO ---
# ==============================================================================

def calculate_time_dependent_reliability(scenario, time_years, corrosion_rate=0.001):
    """
    Calcula la evolución de la confiabilidad con el tiempo.
    corrosion_rate: reducción de espesor por año (mm/año)
    """
    logging.info(f"Calculando confiabilidad dependiente del tiempo: {scenario['name']}")

    results_over_time = []

    for t in time_years:
        # Reducción de espesor por corrosión
        t_original = DETERMINISTIC_CONSTANTS["t"]
        t_degraded = t_original - (corrosion_rate * 0.001 * t)  # mm a m

        if t_degraded <= 0.001:  # Espesor mínimo
            logging.warning(f"Espesor crítico alcanzado en t={t} años")
            break

        # Actualizar constantes
        DETERMINISTIC_CONSTANTS["t"] = t_degraded
        DETERMINISTIC_CONSTANTS["Ap"] = np.pi * (DETERMINISTIC_CONSTANTS["D"] - t_degraded) * t_degraded
        DETERMINISTIC_CONSTANTS["EpAp"] = DETERMINISTIC_CONSTANTS["Ep"] * DETERMINISTIC_CONSTANTS["Ap"]

        # Ejecutar simulación
        N_samples = 50000  # Reducido para velocidad
        results, _, _, _ = run_simulation(scenario, N_samples, store_samples=False)

        results_over_time.append({
            "time": t,
            "Pf": results["Pf"],
            "beta": results["beta"],
            "thickness": t_degraded * 1000  # en mm
        })

    # Restaurar constantes originales
    DETERMINISTIC_CONSTANTS["t"] = t_original
    DETERMINISTIC_CONSTANTS["Ap"] = np.pi * (DETERMINISTIC_CONSTANTS["D"] - t_original) * t_original
    DETERMINISTIC_CONSTANTS["EpAp"] = DETERMINISTIC_CONSTANTS["Ep"] * DETERMINISTIC_CONSTANTS["Ap"]

    logging.info("Análisis temporal completado.")
    return results_over_time


# ==============================================================================
# --- HANDLER DE LOGGING PARA GUI ---
# ==============================================================================

class TextHandler(logging.Handler):
    """Manejador de logging para widget de texto en Tkinter"""

    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)

        self.text_widget.after(0, append)


# ==============================================================================
# --- INTERFAZ GRÁFICA (GUI) ---
# ==============================================================================

class SuperAdvancedPipelineAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 SIMULADOR SUPER AVANZADO - Análisis Probabilístico de Tuberías | E. Ancco 2025")
        self.root.geometry("1600x1000")

        # Estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TNotebook.Tab", font=('Calibri', 10, 'bold'))
        self.style.configure("Treeview.Heading", font=('Calibri', 9, 'bold'))

        # Variables de estado
        self.S_results = None
        self.R_results = None
        self.inputs_stored = None
        self.ratio_S_Sy_stored = None
        self.ductility_stored = None
        self.current_scenario_name = tk.StringVar()
        self.N_simulations = tk.StringVar(value="100000")
        self.fragility_data = {}
        self.form_results = {}
        self.time_dependent_results = []

        # Layout Principal
        main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel Izquierdo (Control)
        control_frame = self.create_control_panel(main_paned)
        main_paned.add(control_frame, weight=0)

        # Panel Derecho (Resultados)
        results_frame = self.create_results_panel(main_paned)
        main_paned.add(results_frame, weight=1)

        # Inicializar
        self.scenario_menu.current(0)
        self.on_scenario_select()

    def create_control_panel(self, parent):
        """Crea el panel de controles izquierdo"""
        frame = ttk.Labelframe(parent, text="⚙️ Configuración", padding=10)

        ttk.Label(frame, text="Escenario:", font=('Calibri', 10, 'bold')).pack(pady=(0, 5))
        self.scenario_menu = ttk.Combobox(
            frame,
            textvariable=self.current_scenario_name,
            values=list(SCENARIOS.keys()),
            state="readonly",
            width=35
        )
        self.scenario_menu.pack(fill=tk.X, padx=5, pady=5)
        self.scenario_menu.bind("<<ComboboxSelected>>", self.on_scenario_select)

        # Parámetros
        self.params_frame = ttk.LabelFrame(frame, text="Parámetros", padding=5)
        self.params_frame.pack(fill=tk.X, pady=5)
        self.param_labels = {}
        for i, key in enumerate(["PGV", "Vs", "Es", "Kt", "Sy", "theta"]):
            ttk.Label(self.params_frame, text=f"{key}:").grid(row=i, column=0, sticky='w')
            self.param_labels[key] = ttk.Label(self.params_frame, text="-", font=('Calibri', 9, 'italic'))
            self.param_labels[key].grid(row=i, column=1, sticky='w', padx=5)

        # N simulaciones
        ttk.Label(frame, text="N Simulaciones:", font=('Calibri', 10, 'bold')).pack(pady=(10, 5))
        ttk.Entry(frame, textvariable=self.N_simulations).pack(fill=tk.X, padx=5, pady=5)

        # Botones
        ttk.Button(frame, text="▶️ Ejecutar Simulación Completa",
                   command=self.run_complete_analysis,
                   style="Accent.TButton").pack(pady=10, fill=tk.X, ipady=5)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Button(frame, text="📈 Curvas de Fragilidad",
                   command=self.run_fragility_analysis).pack(pady=5, fill=tk.X)

        ttk.Button(frame, text="🎯 Análisis FORM",
                   command=self.run_form_analysis).pack(pady=5, fill=tk.X)

        ttk.Button(frame, text="⏱️ Confiabilidad vs Tiempo",
                   command=self.run_time_analysis).pack(pady=5, fill=tk.X)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.export_csv_btn = ttk.Button(frame, text="💾 Exportar CSV",
                                         command=self.export_to_csv, state=tk.DISABLED)
        self.export_csv_btn.pack(pady=5, fill=tk.X)

        self.export_excel_btn = ttk.Button(frame, text="📊 Exportar Excel Completo",
                                           command=self.export_to_excel, state=tk.DISABLED)
        self.export_excel_btn.pack(pady=5, fill=tk.X)

        return frame

    def create_results_panel(self, parent):
        """Crea el panel de resultados con pestañas"""
        frame = ttk.Labelframe(parent, text="📊 Resultados", padding=10)

        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña 1: Resumen
        self.tab_summary = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_summary, text="📋 Resumen")
        self.create_summary_tab()

        # Pestaña 2: Histograma R vs S
        self.tab_histogram = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_histogram, text="📊 Distribución R vs S")
        self.fig_hist, self.ax_hist = plt.subplots(figsize=(8, 6))
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.tab_histogram)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 3: Curvas de Fragilidad
        self.tab_fragility = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_fragility, text="🔴 Curvas de Fragilidad")
        self.fig_frag, self.ax_frag = plt.subplots(figsize=(8, 6))
        self.canvas_frag = FigureCanvasTkAgg(self.fig_frag, master=self.tab_fragility)
        self.canvas_frag.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 4: FORM
        self.tab_form = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_form, text="🎯 FORM")
        self.create_form_tab()

        # Pestaña 5: Sensibilidad
        self.tab_sensitivity = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sensitivity, text="📈 Sensibilidad")
        self.fig_sens, (self.ax_sens_prob, self.ax_sens_det) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas_sens = FigureCanvasTkAgg(self.fig_sens, master=self.tab_sensitivity)
        self.canvas_sens.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 6: Tiempo
        self.tab_time = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_time, text="⏱️ Evolución Temporal")
        self.fig_time, (self.ax_time_pf, self.ax_time_beta) = plt.subplots(2, 1, figsize=(8, 8))
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=self.tab_time)
        self.canvas_time.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 7: Superficie 3D
        self.tab_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3d, text="🌐 Superficie 3D")
        self.fig_3d = plt.figure(figsize=(8, 6))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 8: Tablas
        self.tab_tables = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_tables, text="📄 Tablas")
        self.create_table_tab()

        # Pestaña 9: Log
        self.tab_log = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_log, text="📝 Log")
        self.log_text = scrolledtext.ScrolledText(self.tab_log, state='disabled', wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)

        return frame

    def create_summary_tab(self):
        """Crea la pestaña de resumen"""
        frame = self.tab_summary
        font_title = ('Calibri', 11, 'bold')
        font_value = ('Calibri', 11)

        # Confiabilidad
        ttk.Label(frame, text="🎯 CONFIABILIDAD", font=('Calibri', 13, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)

        labels = [
            ("Probabilidad de Falla (Pf):", "pf_label"),
            ("Índice de Confiabilidad (β):", "beta_label"),
            ("Fallas (Nf):", "nf_label"),
            ("Simulaciones (N):", "n_label")
        ]

        for i, (text, attr) in enumerate(labels, start=1):
            ttk.Label(frame, text=text, font=font_title).grid(row=i, column=0, sticky='w', pady=3)
            label = ttk.Label(frame, text="-", font=font_value)
            label.grid(row=i, column=1, sticky='w', padx=10)
            setattr(self, attr, label)

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky='ew', pady=15)

        # Estadísticas
        ttk.Label(frame, text="📊 ESTADÍSTICAS", font=('Calibri', 13, 'bold')).grid(
            row=6, column=0, columnspan=2, pady=10)

        stats = [
            ("Media S:", "s_mean_label"),
            ("Desv. Est. S:", "s_std_label"),
            ("Media R:", "r_mean_label"),
            ("Desv. Est. R:", "r_std_label"),
            ("Ratio S/Sy (Media):", "ratio_mean_label"),
            ("Ductilidad (Media):", "duct_mean_label")
        ]

        for i, (text, attr) in enumerate(stats, start=7):
            ttk.Label(frame, text=text, font=font_title).grid(row=i, column=0, sticky='w', pady=3)
            label = ttk.Label(frame, text="-", font=font_value)
            label.grid(row=i, column=1, sticky='w', padx=10)
            setattr(self, attr, label)

    def create_form_tab(self):
        """Crea la pestaña de FORM"""
        frame = self.tab_form

        ttk.Label(frame, text="🎯 Resultados del Análisis FORM",
                  font=('Calibri', 12, 'bold')).pack(pady=10)

        self.form_text = scrolledtext.ScrolledText(frame, height=15, width=60, wrap=tk.WORD)
        self.form_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.form_text.config(state=tk.DISABLED)

        # Gráfico de factores alpha
        self.fig_form, self.ax_form = plt.subplots(figsize=(8, 4))
        self.canvas_form = FigureCanvasTkAgg(self.fig_form, master=frame)
        self.canvas_form.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_table_tab(self):
        """Crea la pestaña de tablas"""
        frame = self.tab_tables

        cols = ("theta", "epsilon_g_ax", "epsilon_p_ax", "sigma_ax", "S_MPa", "FS", "ratio")
        col_names = {
            "theta": "θ (°)", "epsilon_g_ax": "ε_g,ax", "epsilon_p_ax": "ε_p,ax",
            "sigma_ax": "σ_ax (MPa)", "S_MPa": "S (MPa)", "FS": "FS", "ratio": "S/Sy"
        }

        self.table_tree = ttk.Treeview(frame, columns=cols, show="headings", height=15)

        for col in cols:
            self.table_tree.heading(col, text=col_names[col])
            self.table_tree.column(col, anchor='e', width=100)

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.table_tree.yview)
        self.table_tree.configure(yscrollcommand=scrollbar.set)

        self.table_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def on_scenario_select(self, event=None):
        """Manejador de cambio de escenario"""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            return

        params = SCENARIOS[scenario_name]
        for key, p in params.items():
            if key in self.param_labels:
                if p['dist'] == 'lognorm':
                    self.param_labels[key].config(text=f"μ={p['mean']:.2g}, COV={p['cov']}")
                elif p['dist'] == 'uniform':
                    self.param_labels[key].config(text=f"[{p['min']}, {p['max']}]")

        self.update_deterministic_table()

    def update_deterministic_table(self):
        """Actualiza la tabla de análisis determinista"""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            return

        for item in self.table_tree.get_children():
            self.table_tree.delete(item)

        scenario_params = SCENARIOS[scenario_name]
        table_data = calculate_deterministic_table_data(scenario_params)

        for row in table_data:
            values = (
                f"{row['theta']:.0f}",
                f"{row['epsilon_g_ax']:.3e}",
                f"{row['epsilon_p_ax']:.3e}",
                f"{row['sigma_ax']:.1f}",
                f"{row['S_MPa']:.1f}",
                f"{row['FS']:.2f}",
                f"{row['ratio_S_Sy']:.3f}"
            )
            self.table_tree.insert("", "end", values=values)

    def run_complete_analysis(self):
        """Ejecuta análisis completo de Monte Carlo"""
        try:
            N = int(self.N_simulations.get())
            if N <= 0:
                raise ValueError("N debe ser positivo")
        except ValueError:
            messagebox.showerror("Error", "N debe ser un entero positivo")
            return

        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            messagebox.showerror("Error", "Seleccione un escenario")
            return

        scenario = {"name": scenario_name, "params": SCENARIOS[scenario_name]}

        try:
            # Ejecutar simulación con almacenamiento de muestras
            results, sensitivity, S_results, R_results, inputs, ratio_S_Sy, ductility = run_simulation(
                scenario, N, store_samples=True
            )

            # Guardar resultados
            self.S_results = S_results
            self.R_results = R_results
            self.inputs_stored = inputs
            self.ratio_S_Sy_stored = ratio_S_Sy
            self.ductility_stored = ductility

            # Actualizar GUI
            self.update_summary(results)
            self.plot_histogram(S_results, R_results, results)
            self.plot_sensitivity(sensitivity)
            self.plot_3d_surface()

            # Activar botones de exportación
            self.export_csv_btn.config(state=tk.NORMAL)
            self.export_excel_btn.config(state=tk.NORMAL)

            messagebox.showinfo("✅ Completo",
                                f"Simulación de {N:,} iteraciones completa.\n"
                                f"Pf = {results['Pf']:.4e}\nβ = {results['beta']:.4f}")

            self.notebook.select(self.tab_summary)

        except Exception as e:
            logging.error(f"Error en simulación: {e}")
            messagebox.showerror("Error", str(e))

    def update_summary(self, results):
        """Actualiza la pestaña de resumen"""
        self.pf_label.config(text=f"{results['Pf']:.4e}")
        self.beta_label.config(text=f"{results['beta']:.4f}")
        self.nf_label.config(text=f"{results['Nf']:,}")
        self.n_label.config(text=f"{results['N']:,}")
        self.s_mean_label.config(text=f"{results['S_mean'] / 1e6:.2f} MPa")
        self.s_std_label.config(text=f"{results['S_std'] / 1e6:.2f} MPa")
        self.r_mean_label.config(text=f"{results['R_mean'] / 1e6:.2f} MPa")
        self.r_std_label.config(text=f"{results['R_std'] / 1e6:.2f} MPa")
        self.ratio_mean_label.config(text=f"{results['ratio_mean']:.3f}")
        self.duct_mean_label.config(text=f"{results['ductility_mean']:.3f}")

    def plot_histogram(self, S_data, R_data, results):
        """Grafica histograma R vs S"""
        self.ax_hist.clear()

        self.ax_hist.hist(S_data / 1e6, bins=60, density=True, alpha=0.7, color='red', label='Solicitación (S)')
        self.ax_hist.hist(R_data / 1e6, bins=60, density=True, alpha=0.7, color='blue', label='Resistencia (R)')

        self.ax_hist.axvline(results['S_mean'] / 1e6, color='red', linestyle='--', lw=2,
                             label=f"Media S = {results['S_mean'] / 1e6:.1f} MPa")
        self.ax_hist.axvline(results['R_mean'] / 1e6, color='blue', linestyle='--', lw=2,
                             label=f"Media R = {results['R_mean'] / 1e6:.1f} MPa")

        self.ax_hist.set_title(f"Distribución R vs S\n{self.current_scenario_name.get()}", fontsize=12,
                               fontweight='bold')
        self.ax_hist.set_xlabel("Esfuerzo (MPa)", fontsize=11)
        self.ax_hist.set_ylabel("Densidad de Probabilidad", fontsize=11)
        self.ax_hist.legend(fontsize=9)
        self.ax_hist.grid(True, alpha=0.3)

        self.fig_hist.tight_layout()
        self.canvas_hist.draw()

    def plot_sensitivity(self, sensitivity):
        """Grafica análisis de sensibilidad"""
        # Sensibilidad probabilística (Tornado)
        self.ax_sens_prob.clear()

        if sensitivity:
            sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)
            labels = [item[0] for item in sorted_sens]
            values = [item[1] for item in sorted_sens]
            colors = ['red' if v < 0 else 'blue' for v in values]

            self.ax_sens_prob.barh(labels, values, color=colors)
            self.ax_sens_prob.set_title("Sensibilidad Probabilística\n(Correlación de Spearman)", fontweight='bold')
            self.ax_sens_prob.set_xlabel("Correlación con S")
            self.ax_sens_prob.invert_yaxis()
            self.ax_sens_prob.grid(True, axis='x', alpha=0.3)

            for i, v in enumerate(values):
                self.ax_sens_prob.text(v + 0.01 if v > 0 else v - 0.01, i, f"{v:.3f}",
                                       va='center', ha='left' if v > 0 else 'right')

        # Sensibilidad determinística (FS vs theta)
        self.ax_sens_det.clear()

        for scenario_name, params in SCENARIOS.items():
            table_data = calculate_deterministic_table_data(params)
            thetas = [row['theta'] for row in table_data]
            fss = [row['FS'] for row in table_data]
            label = scenario_name.split()[1]  # Corto
            self.ax_sens_det.plot(thetas, fss, 'o-', label=label, linewidth=2)

        self.ax_sens_det.axhline(1.0, color='red', linestyle='--', lw=2, label='FS = 1.0')
        self.ax_sens_det.set_title("Sensibilidad Determinística\nFS vs Ángulo", fontweight='bold')
        self.ax_sens_det.set_xlabel("Ángulo θ (°)")
        self.ax_sens_det.set_ylabel("Factor de Seguridad (FS)")
        self.ax_sens_det.legend()
        self.ax_sens_det.grid(True, alpha=0.3)

        self.fig_sens.tight_layout()
        self.canvas_sens.draw()

    def plot_3d_surface(self):
        """Grafica superficie de respuesta 3D (PGV vs Vs vs FS)"""
        self.ax_3d.clear()

        scenario_name = self.current_scenario_name.get()
        params = SCENARIOS[scenario_name]

        # Crear malla
        pgv_vec = np.linspace(0.2, 0.8, 25)
        vs_vec = np.linspace(200, 500, 25)
        PGV_grid, Vs_grid = np.meshgrid(pgv_vec, vs_vec)
        FS_grid = np.zeros_like(PGV_grid)

        base_params = {
            "Kt": params["Kt"]["mean"],
            "Sy": params["Sy"]["mean"],
            "theta": 30.0
        }

        for i in range(PGV_grid.shape[0]):
            for j in range(PGV_grid.shape[1]):
                p = base_params.copy()
                p["PGV"] = PGV_grid[i, j]
                p["Vs"] = Vs_grid[i, j]
                p["Es"] = (p["Vs"] / 400.0) ** 2 * params["Es"]["mean"]
                res = calculate_single_point_fs(p)
                FS_grid[i, j] = min(res['FS'], 3.0) if res else 0

        surf = self.ax_3d.plot_surface(PGV_grid, Vs_grid, FS_grid, cmap='RdYlGn_r',
                                       alpha=0.8, edgecolor='none')

        self.ax_3d.set_xlabel('PGV (m/s)', fontsize=10)
        self.ax_3d.set_ylabel('$V_s$ (m/s)', fontsize=10)
        self.ax_3d.set_zlabel('FS', fontsize=10)
        self.ax_3d.set_title(f'Superficie de Respuesta 3D\n{scenario_name}', fontweight='bold')

        self.fig_3d.colorbar(surf, ax=self.ax_3d, shrink=0.5, label='FS')
        self.canvas_3d.draw()

    def run_fragility_analysis(self):
        """Ejecuta análisis de curvas de fragilidad"""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            messagebox.showerror("Error", "Seleccione un escenario")
            return

        scenario = {"name": scenario_name, "params": SCENARIOS[scenario_name]}

        # Rango de intensidades (PGV)
        im_range = np.linspace(0.1, 1.2, 30)

        try:
            self.fragility_data = calculate_fragility_curves(scenario, im_range, N=50000)
            self.plot_fragility_curves(im_range)
            messagebox.showinfo("✅ Completo", "Curvas de fragilidad calculadas")
            self.notebook.select(self.tab_fragility)
        except Exception as e:
            logging.error(f"Error en fragilidad: {e}")
            messagebox.showerror("Error", str(e))

    def plot_fragility_curves(self, im_range):
        """Grafica curvas de fragilidad"""
        self.ax_frag.clear()

        colors = ['green', 'orange', 'red']
        linestyles = ['-', '--', '-.']

        for (ls_name, data), color, ls in zip(self.fragility_data.items(), colors, linestyles):
            label = f"{ls_name}: {LIMIT_STATES[ls_name]['description']}"
            self.ax_frag.plot(im_range, data, color=color, linestyle=ls,
                              linewidth=2.5, marker='o', markersize=4, label=label)

        self.ax_frag.set_title(f"Curvas de Fragilidad\n{self.current_scenario_name.get()}",
                               fontsize=12, fontweight='bold')
        self.ax_frag.set_xlabel("PGV - Velocidad Pico del Suelo (m/s)", fontsize=11)
        self.ax_frag.set_ylabel("Probabilidad de Excedencia", fontsize=11)
        self.ax_frag.legend(loc='best', fontsize=9)
        self.ax_frag.grid(True, alpha=0.3)
        self.ax_frag.set_ylim([0, 1])

        self.fig_frag.tight_layout()
        self.canvas_frag.draw()

    def run_form_analysis(self):
        """Ejecuta análisis FORM"""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            messagebox.showerror("Error", "Seleccione un escenario")
            return

        scenario = {"name": scenario_name, "params": SCENARIOS[scenario_name]}

        try:
            self.form_results = perform_form_analysis(scenario)
            self.display_form_results()
            messagebox.showinfo("✅ Completo", "Análisis FORM finalizado")
            self.notebook.select(self.tab_form)
        except Exception as e:
            logging.error(f"Error en FORM: {e}")
            messagebox.showerror("Error", str(e))

    def display_form_results(self):
        """Muestra resultados de FORM"""
        self.form_text.config(state=tk.NORMAL)
        self.form_text.delete(1.0, tk.END)

        if self.form_results.get("success"):
            text = f"""
╔════════════════════════════════════════╗
║     RESULTADOS DEL ANÁLISIS FORM       ║
╚════════════════════════════════════════╝

✅ Convergencia: Exitosa
🎯 Iteraciones: {self.form_results['iterations']}

📊 ÍNDICE DE CONFIABILIDAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
β (FORM) = {self.form_results['beta_form']:.4f}

📍 PUNTO DE DISEÑO (Espacio Normal Estándar)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
u*_PGV  = {self.form_results['design_point_u'][0]:+.4f}
u*_Vs   = {self.form_results['design_point_u'][1]:+.4f}
u*_Es   = {self.form_results['design_point_u'][2]:+.4f}
u*_Kt   = {self.form_results['design_point_u'][3]:+.4f}
u*_Sy   = {self.form_results['design_point_u'][4]:+.4f}

⚖️ FACTORES DE IMPORTANCIA (α-factors)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
α_PGV   = {self.form_results['alpha_factors'][0]:+.4f}
α_Vs    = {self.form_results['alpha_factors'][1]:+.4f}
α_Es    = {self.form_results['alpha_factors'][2]:+.4f}
α_Kt    = {self.form_results['alpha_factors'][3]:+.4f}
α_Sy    = {self.form_results['alpha_factors'][4]:+.4f}

💡 Los α-factors indican la importancia relativa
   de cada variable en la probabilidad de  falla.
   |α| ≈ 1.0 → Variable muy influyente
   |α| ≈ 0.0 → Variable poco influyente
"""
            self.form_text.insert(tk.END, text)

            # Graficar factores alpha
            self.ax_form.clear()
            labels = ['PGV', 'Vs', 'Es', 'Kt', 'Sy']
            alphas = self.form_results['alpha_factors']
            colors = ['red' if a < 0 else 'blue' for a in alphas]

            bars = self.ax_form.bar(labels, alphas, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            self.ax_form.axhline(0, color='black', linewidth=0.8)
            self.ax_form.set_title('Factores de Importancia (α-factors)', fontweight='bold', fontsize=12)
            self.ax_form.set_ylabel('α-factor', fontsize=11)
            self.ax_form.set_ylim([-1.1, 1.1])
            self.ax_form.grid(True, axis='y', alpha=0.3)

            # Añadir valores sobre las barras
            for bar, val in zip(bars, alphas):
                height = bar.get_height()
                self.ax_form.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height > 0 else height - 0.05,
                                f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                                fontweight='bold', fontsize=10)

            self.fig_form.tight_layout()
            self.canvas_form.draw()
        else:
            self.form_text.insert(tk.END, "\n❌ FORM no convergió.\nIntente con otros parámetros iniciales.")

        self.form_text.config(state=tk.DISABLED)

    def run_time_analysis(self):
        """Ejecuta análisis de confiabilidad dependiente del tiempo"""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            messagebox.showerror("Error", "Seleccione un escenario")
            return

        scenario = {"name": scenario_name, "params": SCENARIOS[scenario_name]}

        # Años de análisis
        time_years = np.arange(0, 51, 5)  # 0 a 50 años, cada 5 años

        try:
            self.time_dependent_results = calculate_time_dependent_reliability(scenario, time_years, corrosion_rate=0.05)
            self.plot_time_dependent_reliability()
            messagebox.showinfo("✅ Completo", "Análisis temporal finalizado")
            self.notebook.select(self.tab_time)
        except Exception as e:
            logging.error(f"Error en análisis temporal: {e}")
            messagebox.showerror("Error", str(e))

    def plot_time_dependent_reliability(self):
        """Grafica evolución temporal de la confiabilidad"""
        if not self.time_dependent_results:
            return

        times = [r['time'] for r in self.time_dependent_results]
        pfs = [r['Pf'] for r in self.time_dependent_results]
        betas = [r['beta'] for r in self.time_dependent_results]
        thicknesses = [r['thickness'] for r in self.time_dependent_results]

        # Gráfico 1: Pf vs Tiempo
        self.ax_time_pf.clear()
        self.ax_time_pf.plot(times, pfs, 'ro-', linewidth=2.5, markersize=8, label='Pf(t)')
        self.ax_time_pf.set_title('Evolución de la Probabilidad de Falla\n(Efecto de Corrosión)',
                                  fontweight='bold', fontsize=12)
        self.ax_time_pf.set_xlabel('Tiempo (años)', fontsize=11)
        self.ax_time_pf.set_ylabel('Probabilidad de Falla (Pf)', fontsize=11)
        self.ax_time_pf.grid(True, alpha=0.3)
        self.ax_time_pf.legend()

        # Añadir eje secundario para espesor
        ax_thick = self.ax_time_pf.twinx()
        ax_thick.plot(times, thicknesses, 'b--', linewidth=2, alpha=0.7, label='Espesor')
        ax_thick.set_ylabel('Espesor de Pared (mm)', fontsize=11, color='blue')
        ax_thick.tick_params(axis='y', labelcolor='blue')

        # Gráfico 2: Beta vs Tiempo
        self.ax_time_beta.clear()
        self.ax_time_beta.plot(times, betas, 'go-', linewidth=2.5, markersize=8, label='β(t)')

        # Líneas de referencia de niveles de confiabilidad
        self.ax_time_beta.axhline(y=3.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                                 label='β = 3.0 (Aceptable)')
        self.ax_time_beta.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                                 label='β = 2.0 (Marginal)')
        self.ax_time_beta.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                                 label='β = 1.0 (Crítico)')

        self.ax_time_beta.set_title('Evolución del Índice de Confiabilidad', fontweight='bold', fontsize=12)
        self.ax_time_beta.set_xlabel('Tiempo (años)', fontsize=11)
        self.ax_time_beta.set_ylabel('Índice de Confiabilidad (β)', fontsize=11)
        self.ax_time_beta.grid(True, alpha=0.3)
        self.ax_time_beta.legend(loc='best')

        self.fig_time.tight_layout()
        self.canvas_time.draw()

    def export_to_csv(self):
        """Exporta datos a CSV"""
        if self.S_results is None or self.R_results is None:
            messagebox.showwarning("Sin Datos", "No hay datos para exportar. Ejecute una simulación primero.")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
                title="Guardar Datos (R, S)"
            )
            if not filepath:
                return

            logging.info(f"Exportando {len(self.S_results)} puntos a CSV...")

            df = pd.DataFrame({
                "Resistencia_R_Pa": self.R_results,
                "Solicitacion_S_Pa": self.S_results,
                "Resistencia_R_MPa": self.R_results / 1e6,
                "Solicitacion_S_MPa": self.S_results / 1e6,
                "Ratio_S_Sy": self.ratio_S_Sy_stored,
                "Ductilidad": self.ductility_stored
            })

            if self.inputs_stored:
                for key, values in self.inputs_stored.items():
                    df[key] = values

            df.to_csv(filepath, index=False, float_format='%.6f')

            logging.info("Exportación CSV completada.")
            messagebox.showinfo("✅ Exportado", f"Datos guardados en:\n{filepath}")

        except Exception as e:
            logging.error(f"Error en exportación CSV: {e}")
            messagebox.showerror("Error", str(e))

    def export_to_excel(self):
        """Exporta análisis completo a Excel con múltiples hojas"""
        if self.S_results is None or self.R_results is None:
            messagebox.showwarning("Sin Datos", "No hay datos para exportar. Ejecute una simulación primero.")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx"), ("Todos", "*.*")],
                title="Guardar Reporte Completo"
            )
            if not filepath:
                return

            logging.info("Generando reporte Excel completo...")

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Hoja 1: Resumen
                summary_data = {
                    "Parámetro": [
                        "Escenario",
                        "Fecha de Análisis",
                        "N Simulaciones",
                        "Probabilidad de Falla (Pf)",
                        "Índice de Confiabilidad (β)",
                        "Número de Fallas (Nf)",
                        "Media S (MPa)",
                        "Desv. Est. S (MPa)",
                        "Media R (MPa)",
                        "Desv. Est. R (MPa)",
                        "Ratio S/Sy (Media)",
                        "Ductilidad (Media)"
                    ],
                    "Valor": [
                        self.current_scenario_name.get(),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.N_simulations.get(),
                        f"{float(self.pf_label.cget('text')):.6e}",
                        self.beta_label.cget('text'),
                        self.nf_label.cget('text'),
                        self.s_mean_label.cget('text'),
                        self.s_std_label.cget('text'),
                        self.r_mean_label.cget('text'),
                        self.r_std_label.cget('text'),
                        self.ratio_mean_label.cget('text'),
                        self.duct_mean_label.cget('text')
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Resumen', index=False)

                # Hoja 2: Datos Crudos (R, S)
                df_raw = pd.DataFrame({
                    "R_Pa": self.R_results,
                    "S_Pa": self.S_results,
                    "R_MPa": self.R_results / 1e6,
                    "S_MPa": self.S_results / 1e6,
                    "Ratio_S_Sy": self.ratio_S_Sy_stored,
                    "Ductilidad": self.ductility_stored
                })
                df_raw.to_excel(writer, sheet_name='Datos_Crudos', index=False)

                # Hoja 3: Inputs de la Simulación
                if self.inputs_stored:
                    df_inputs = pd.DataFrame(self.inputs_stored)
                    df_inputs.to_excel(writer, sheet_name='Inputs_Simulacion', index=False)

                # Hoja 4: Tabla Determinista
                scenario_params = SCENARIOS[self.current_scenario_name.get()]
                table_data = calculate_deterministic_table_data(scenario_params)
                df_det = pd.DataFrame(table_data)
                df_det.to_excel(writer, sheet_name='Analisis_Deterministico', index=False)

                # Hoja 5: Curvas de Fragilidad
                if self.fragility_data:
                    df_frag_dict = {"PGV": np.linspace(0.1, 1.2, len(list(self.fragility_data.values())[0]))}
                    for ls_name, probs in self.fragility_data.items():
                        df_frag_dict[ls_name] = probs
                    df_frag = pd.DataFrame(df_frag_dict)
                    df_frag.to_excel(writer, sheet_name='Curvas_Fragilidad', index=False)

                # Hoja 6: Resultados FORM
                if self.form_results.get("success"):
                    form_data = {
                        "Parámetro": ["Beta_FORM", "Iteraciones",
                                     "u*_PGV", "u*_Vs", "u*_Es", "u*_Kt", "u*_Sy",
                                     "alpha_PGV", "alpha_Vs", "alpha_Es", "alpha_Kt", "alpha_Sy"],
                        "Valor": [
                            self.form_results['beta_form'],
                            self.form_results['iterations'],
                            *self.form_results['design_point_u'],
                            *self.form_results['alpha_factors']
                        ]
                    }
                    df_form = pd.DataFrame(form_data)
                    df_form.to_excel(writer, sheet_name='Analisis_FORM', index=False)

                # Hoja 7: Evolución Temporal
                if self.time_dependent_results:
                    df_time = pd.DataFrame(self.time_dependent_results)
                    df_time.to_excel(writer, sheet_name='Evolucion_Temporal', index=False)

            logging.info("Reporte Excel completado.")
            messagebox.showinfo("✅ Exportado", f"Reporte completo guardado en:\n{filepath}")

        except Exception as e:
            logging.error(f"Error en exportación Excel: {e}")
            messagebox.showerror("Error", str(e))

# ==============================================================================
# --- ANÁLISIS MULTI-ESCENARIO (COMPARACIÓN ESTADÍSTICA) ---
# ==============================================================================

def compare_scenarios_statistical(results_dict):
    """
    Compara estadísticamente múltiples escenarios.
    results_dict: {scenario_name: {"S": array, "R": array, ...}}
    """
    logging.info("Realizando comparación estadística entre escenarios...")

    # Extraer datos
    scenario_names = list(results_dict.keys())
    S_samples = [results_dict[name]["S"] for name in scenario_names]

    # ANOVA para comparar medias
    f_stat, p_value = f_oneway(*S_samples)

    # Estadísticas descriptivas
    stats = []
    for name in scenario_names:
        S = results_dict[name]["S"]
        R = results_dict[name]["R"]
        stats.append({
            "Escenario": name,
            "Media_S_MPa": np.mean(S) / 1e6,
            "Desv_S_MPa": np.std(S) / 1e6,
            "Media_R_MPa": np.mean(R) / 1e6,
            "Desv_R_MPa": np.std(R) / 1e6,
            "Pf": results_dict[name]["Pf"],
            "Beta": results_dict[name]["beta"]
        })

    df_stats = pd.DataFrame(stats)

    comparison = {
        "stats_table": df_stats,
        "anova_f": f_stat,
        "anova_p": p_value,
        "significantly_different": p_value < 0.05
    }

    logging.info(f"ANOVA: F={f_stat:.4f}, p={p_value:.6f}")
    return comparison

# ==============================================================================
# --- ANÁLISIS DE CONVERGENCIA DE MONTE CARLO ---
# ==============================================================================

def analyze_mc_convergence(scenario, N_max=100000, N_steps=20):
    """
    Analiza la convergencia de Pf con el número de simulaciones.
    """
    logging.info("Analizando convergencia de Monte Carlo...")

    N_values = np.logspace(3, np.log10(N_max), N_steps).astype(int)
    pf_values = []
    beta_values = []

    for N in N_values:
        results, _, _, _ = run_simulation(scenario, N, store_samples=False)
        pf_values.append(results["Pf"])
        beta_values.append(results["beta"])

    convergence_data = {
        "N": N_values,
        "Pf": pf_values,
        "Beta": beta_values
    }

    logging.info("Análisis de convergencia completado.")
    return convergence_data

# ==============================================================================
# --- VENTANA ADICIONAL: ANÁLISIS MULTI-ESCENARIO ---
# ==============================================================================

class MultiScenarioComparisonWindow:
    """Ventana para comparar múltiples escenarios simultáneamente"""

    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("📊 Comparación Multi-Escenario")
        self.window.geometry("1200x800")

        # Variables
        self.selected_scenarios = []
        self.results_dict = {}

        # Frame superior: Selección
        selection_frame = ttk.LabelFrame(self.window, text="Seleccionar Escenarios", padding=10)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)

        self.scenario_vars = {}
        for scenario_name in SCENARIOS.keys():
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(selection_frame, text=scenario_name, variable=var)
            cb.pack(anchor='w')
            self.scenario_vars[scenario_name] = var

        ttk.Button(selection_frame, text="▶️ Ejecutar Comparación",
                  command=self.run_comparison).pack(pady=10)

        # Frame inferior: Resultados
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña 1: Tabla comparativa
        self.tab_table = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_table, text="Tabla Comparativa")

        cols = ("Escenario", "Pf", "Beta", "Media_S", "Media_R")
        self.comparison_tree = ttk.Treeview(self.tab_table, columns=cols, show="headings")
        for col in cols:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, width=150)
        self.comparison_tree.pack(fill=tk.BOTH, expand=True)

        # Pestaña 2: Gráficos
        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Gráficos Comparativos")

        self.fig_comp, self.axes_comp = plt.subplots(2, 2, figsize=(12, 10))
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=self.tab_plots)
        self.canvas_comp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_comparison(self):
        """Ejecuta comparación entre escenarios seleccionados"""
        self.selected_scenarios = [name for name, var in self.scenario_vars.items() if var.get()]

        if len(self.selected_scenarios) < 2:
            messagebox.showwarning("Selección Insuficiente", "Seleccione al menos 2 escenarios")
            return

        N = 50000  # Simulaciones por escenario

        try:
            for scenario_name in self.selected_scenarios:
                logging.info(f"Simulando {scenario_name}...")
                scenario = {"name": scenario_name, "params": SCENARIOS[scenario_name]}
                results, _, S_results, R_results = run_simulation(scenario, N, store_samples=False)

                self.results_dict[scenario_name] = {
                    "S": S_results,
                    "R": R_results,
                    "Pf": results["Pf"],
                    "beta": results["beta"],
                    "S_mean": results["S_mean"],
                    "R_mean": results["R_mean"]
                }

            # Análisis estadístico
            comparison = compare_scenarios_statistical(self.results_dict)

            # Actualizar tabla
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)

            for _, row in comparison["stats_table"].iterrows():
                values = (
                    row["Escenario"],
                    f"{row['Pf']:.4e}",
                    f"{row['Beta']:.3f}",
                    f"{row['Media_S_MPa']:.1f} MPa",
                    f"{row['Media_R_MPa']:.1f} MPa"
                )
                self.comparison_tree.insert("", "end", values=values)

            # Graficar
            self.plot_comparisons()

            messagebox.showinfo("✅ Completo",
                              f"Comparación de {len(self.selected_scenarios)} escenarios completa.\n"
                              f"ANOVA p-value = {comparison['anova_p']:.6f}")

        except Exception as e:
            logging.error(f"Error en comparación: {e}")
            messagebox.showerror("Error", str(e))

    def plot_comparisons(self):
        """Genera gráficos comparativos"""
        # Limpiar
        for ax in self.axes_comp.flat:
            ax.clear()

        # Gráfico 1: Histogramas superpuestos
        ax1 = self.axes_comp[0, 0]
        for scenario_name, data in self.results_dict.items():
            ax1.hist(data["S"]/1e6, bins=40, alpha=0.5, label=scenario_name.split()[1], density=True)
        ax1.set_title("Distribuciones de Solicitación (S)")
        ax1.set_xlabel("S (MPa)")
        ax1.set_ylabel("Densidad")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Box plots
        ax2 = self.axes_comp[0, 1]
        S_data = [data["S"]/1e6 for data in self.results_dict.values()]
        labels = [name.split()[1] for name in self.results_dict.keys()]
        bp = ax2.boxplot(S_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_title("Box Plots de S")
        ax2.set_ylabel("S (MPa)")
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Barras de Pf
        ax3 = self.axes_comp[1, 0]
        pfs = [data["Pf"] for data in self.results_dict.values()]
        ax3.bar(labels, pfs, color=['red', 'orange', 'yellow'][:len(labels)])
        ax3.set_title("Probabilidades de Falla")
        ax3.set_ylabel("Pf")
        ax3.grid(True, axis='y', alpha=0.3)

        # Gráfico 4: Scatter S vs R
        ax4 = self.axes_comp[1, 1]
        for scenario_name, data in self.results_dict.items():
            # Muestra aleatoria para velocidad
            idx = np.random.choice(len(data["S"]), size=min(1000, len(data["S"])), replace=False)
            ax4.scatter(data["R"][idx]/1e6, data["S"][idx]/1e6,
                       alpha=0.3, s=10, label=scenario_name.split()[1])

        # Línea S=R
        max_val = max([np.max(data["R"])/1e6 for data in self.results_dict.values()])
        ax4.plot([0, max_val], [0, max_val], 'k--', lw=2, label='S = R (Falla)')
        ax4.set_title("Scatter S vs R")
        ax4.set_xlabel("R (MPa)")
        ax4.set_ylabel("S (MPa)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        self.fig_comp.tight_layout()
        self.canvas_comp.draw()

# ==============================================================================
# --- MENÚ PRINCIPAL EXTENDIDO ---
# ==============================================================================

def create_menu(root, app):
    """Crea barra de menú con opciones avanzadas"""
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Menú Archivo
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="📁 Archivo", menu=file_menu)
    file_menu.add_command(label="Exportar CSV", command=app.export_to_csv)
    file_menu.add_command(label="Exportar Excel", command=app.export_to_excel)
    file_menu.add_separator()
    file_menu.add_command(label="Salir", command=root.quit)

    # Menú Análisis
    analysis_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="🔬 Análisis", menu=analysis_menu)
    analysis_menu.add_command(label="Simulación Monte Carlo", command=app.run_complete_analysis)
    analysis_menu.add_command(label="Curvas de Fragilidad", command=app.run_fragility_analysis)
    analysis_menu.add_command(label="FORM", command=app.run_form_analysis)
    analysis_menu.add_command(label="Confiabilidad vs Tiempo", command=app.run_time_analysis)
    analysis_menu.add_separator()
    analysis_menu.add_command(label="Comparación Multi-Escenario",
                             command=lambda: MultiScenarioComparisonWindow(root))

    # Menú Ayuda
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="❓ Ayuda", menu=help_menu)
    help_menu.add_command(label="Acerca de", command=lambda: show_about_dialog(root))
    help_menu.add_command(label="Manual de Usuario", command=lambda: show_user_manual(root))

def show_about_dialog(parent):
    """Muestra diálogo de información"""
    about_text = """
🚀 SIMULADOR SUPER AVANZADO DE TUBERÍAS
═══════════════════════════════════════

Tesista: Emanuel Ancco
Usuario: EmanuelAncco
Fecha: 11 de Noviembre, 2025

FUNCIONALIDADES:
• Análisis Determinista SSI
• Simulación de Monte Carlo (MCS)
• Curvas de Fragilidad
• Análisis FORM
• Confiabilidad Dependiente del Tiempo
• Análisis Multi-Escenario
• Visualizaciones 3D
• Exportación Completa

Versión: 2.0 (Super Avanzado)
"""
    messagebox.showinfo("Acerca de", about_text)

def show_user_manual(parent):
    """Muestra manual de usuario"""
    manual_window = tk.Toplevel(parent)
    manual_window.title("📖 Manual de Usuario")
    manual_window.geometry("800x600")

    text = scrolledtext.ScrolledText(manual_window, wrap=tk.WORD, font=('Courier', 10))
    text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    manual_content = """
═══════════════════════════════════════════════════════════════════
                    MANUAL DE USUARIO - SIMULADOR AVANZADO
═══════════════════════════════════════════════════════════════════

1. INTRODUCCIÓN
───────────────
Este simulador permite realizar análisis probabilísticos avanzados de
tuberías enterradas sometidas a cargas sísmicas, considerando:
- Interacción Suelo-Estructura (SSI)
- Incertidumbres en parámetros geotécnicos y estructurales
- Estados límite múltiples
- Degradación temporal

2. FLUJO DE TRABAJO RECOMENDADO
────────────────────────────────
PASO 1: Seleccionar Escenario
  → Elija entre los 3 casos de estudio predefinidos

PASO 2: Configurar Simulación
  → Establezca el número de simulaciones (N)
  → Recomendado: N = 100,000 para resultados precisos

PASO 3: Ejecutar Análisis
  a) Simulación Monte Carlo (botón principal)
     - Calcula Pf, β, y estadísticas
     - Genera histogramas R vs S
     - Análisis de sensibilidad

  b) Curvas de Fragilidad
     - Probabilidad de excedencia vs PGV
     - 3 estados límite: Elástico, Fluencia, Colapso

  c) Análisis FORM
     - Punto de diseño
     - Factores de importancia (α-factors)
     - β más preciso

  d) Evolución Temporal
     - Efecto de corrosión
     - Pf y β vs tiempo

PASO 4: Exportar Resultados
  → CSV: Datos crudos
  → Excel: Reporte completo con todas las hojas

3. INTERPRETACIÓN DE RESULTADOS
────────────────────────────────
• Pf (Probabilidad de Falla):
  - Pf < 1e-4: Muy confiable
  - 1e-4 < Pf < 1e-3: Confiable
  - 1e-3 < Pf < 1e-2: Aceptable
  - Pf > 1e-2: Riesgoso

• β (Índice de Confiabilidad):
  - β > 3.5: Excelente
  - 3.0 < β < 3.5: Bueno
  - 2.0 < β < 3.0: Aceptable
  - β < 2.0: Inadecuado

• α-factors (FORM):
  - |α| ≈ 1: Variable muy influyente
  - |α| ≈ 0: Variable poco importante
  - α > 0: Aumentar reduce seguridad
  - α < 0: Aumentar aumenta seguridad

4. CURVAS DE FRAGILIDAD
───────────────────────
Las curvas muestran la probabilidad de exceder un estado límite en
función de la intensidad sísmica (PGV).

Estados Límite:
- LS1 (Verde): Inicio de fluencia (S/Sy = 0.67)
- LS2 (Naranja): Fluencia completa (S/Sy = 1.0)
- LS3 (Rojo): Colapso incipiente (S/Sy = 1.5)

5. ANÁLISIS MULTI-ESCENARIO
────────────────────────────
Menú → Análisis → Comparación Multi-Escenario
- Permite comparar 2 o más escenarios simultáneamente
- Incluye test estadístico ANOVA
- Genera gráficos comparativos

6. EXPORTACIÓN DE DATOS
───────────────────────
FORMATO CSV:
- Incluye vectores R, S, ratio S/Sy, ductilidad
- Ideal para post-procesamiento en Python/R/MATLAB

FORMATO EXCEL:
- Múltiples hojas con todos los análisis
- Listo para incluir en tesis/reportes

7. SOLUCIÓN DE PROBLEMAS
────────────────────────
• "N debe ser positivo"
  → Verifique que N sea un número entero > 0

• "FORM no convergió"
  → Normal en algunos casos. Use resultados de MCS

• Simulación lenta
  → Reduzca N (mínimo recomendado: 10,000)

• Gráficos no se actualizan
  → Cambie de pestaña y regrese

8. RECOMENDACIONES PARA TESIS
──────────────────────────────
✓ Use N ≥ 100,000 para resultados finales
✓ Ejecute análisis de sensibilidad (FORM) para justificar simplificaciones
✓ Incluya curvas de fragilidad para diferentes escenarios sísmicos
✓ Documente convergencia de Pf con N
✓ Compare con normativas (ej. ASME B31)

9. CONTACTO Y SOPORTE
─────────────────────
Usuario: EmanuelAncco
Fecha: 2025-11-11

Para dudas o sugerencias, consulte la documentación técnica
o contacte al desarrollador.

═══════════════════════════════════════════════════════════════════
"""
    text.insert(tk.END, manual_content)
    text.config(state=tk.DISABLED)

# ==============================================================================
# --- EJECUCIÓN PRINCIPAL ---
# ==============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.info("="*80)
    logging.info("INICIANDO SIMULADOR SUPER AVANZADO")
    logging.info(f"Usuario: EmanuelAncco")
    logging.info(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)

    # Crear ventana principal
    root = tk.Tk()

    # Configurar estilos adicionales
    style = ttk.Style()
    style.configure("Accent.TButton",
                   font=('Calibri', 11, 'bold'),
                   foreground='white',
                   background='#007ACC')

    # Crear aplicación
    app = SuperAdvancedPipelineAnalysisApp(root)

    # Crear menú
    create_menu(root, app)

    # Mensaje de bienvenida
    logging.info("✅ Interfaz gráfica cargada exitosamente")
    logging.info("📊 Listo para comenzar análisis")

    # Iniciar loop
    root.mainloop()

    logging.info("="*80)
    logging.info("SIMULADOR FINALIZADO")
    logging.info("="*80)