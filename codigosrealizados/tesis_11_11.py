import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import lognorm, uniform, norm, spearmanr
import pandas as pd
import logging
import time
import io

# ==============================================================================
# --- DEFINICIÓN DE CASOS DE ESTUDIO (OBJETIVO 3) ---
# ==============================================================================

# Parámetros probabilísticos para los 3 escenarios
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

# Parámetros deterministas constantes para todos los casos
DETERMINISTIC_CONSTANTS = {
    "D": 0.762,  # Diámetro Exterior (m)
    "t": 0.0095,  # Espesor de Pared (m)
    "Ep": 207e9,  # Módulo de Young Acero (Pa)
    "nu_s": 0.3,  # Coef. Poisson Suelo
    "sigma_h": 150e6,  # Esfuerzo Tangencial (Pa)
    "f": 1.0  # Frecuencia Sismo (Hz)
}
# Propiedades calculadas
DETERMINISTIC_CONSTANTS["Ap"] = np.pi * (DETERMINISTIC_CONSTANTS["D"] - DETERMINISTIC_CONSTANTS["t"]) * \
                                DETERMINISTIC_CONSTANTS["t"]
DETERMINISTIC_CONSTANTS["EpAp"] = DETERMINISTIC_CONSTANTS["Ep"] * DETERMINISTIC_CONSTANTS["Ap"]


# ==============================================================================
# --- MOTOR DE CÁLCULO (OBJETIVOS 1, 2, 4) ---
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


def calculate_single_point_fs(params):
    """
    Calcula el FS determinista para un solo punto (un set de parámetros medios).
    Esta es la implementación central de las Ecuaciones 1-15.
    """
    try:
        # Extraer constantes
        EpAp = DETERMINISTIC_CONSTANTS["EpAp"]
        sigma_h = DETERMINISTIC_CONSTANTS["sigma_h"]
        D = DETERMINISTIC_CONSTANTS["D"]
        nu_s = DETERMINISTIC_CONSTANTS["nu_s"]
        f = DETERMINISTIC_CONSTANTS["f"]

        # Extraer variables de entrada
        PGV = params["PGV"]
        Vs = params["Vs"]
        Es = params["Es"]
        Kt = params["Kt"]
        Sy = params["Sy"]
        theta_deg = params["theta"]
        theta_rad = np.radians(theta_deg)

        # --- Objetivo 1: Modelo Determinista SSI (Ecs. 1-10) ---
        # Este modelo se enfoca en el resorte axial (longitudinal) 't-x'
        # (Ver Tesis González Peña, 2016, Cap 5)
        Gs = Es / (2 * (1 + nu_s))
        k_ax = np.pi * D * Gs
        epsilon_g_ax = (PGV / Vs) * (np.cos(theta_rad) ** 2)  # Ec. 5
        k_w = (2 * np.pi * f) / Vs

        termino_rigidez = (EpAp * k_w ** 2) / k_ax
        epsilon_p_ax = epsilon_g_ax * (1 / (1 + termino_rigidez))  # Ec. 11

        sigma_ax = DETERMINISTIC_CONSTANTS["Ep"] * epsilon_p_ax

        # --- Objetivo 2: Concentración de Esfuerzos (Ecs. 12-15) ---
        sigma_x_weld = Kt * sigma_ax  # Ec. 13

        # Ec. 15 (Von Mises)
        S = np.sqrt(sigma_x_weld ** 2 + sigma_h ** 2 - (sigma_x_weld * sigma_h))

        R = Sy
        FS = R / S if S > 0 else np.inf

        return {
            "theta": theta_deg,
            "cos2_theta": np.cos(theta_rad) ** 2,
            "epsilon_g_ax": epsilon_g_ax,
            "k_w": k_w,
            "termino_axial": termino_rigidez,
            "epsilon_p_ax": epsilon_p_ax,
            "sigma_ax": sigma_ax / 1e6,  # en MPa
            "sigma_x_weld": sigma_x_weld / 1e6,  # en MPa
            "S_MPa": S / 1e6,  # en MPa
            "FS": FS
        }
    except Exception as e:
        logging.error(f"Error en cálculo determinista: {e}")
        return None


def calculate_deterministic_table_data(scenario_params):
    """
    Genera los datos para las Tablas 9, 10, 11, variando el ángulo theta.
    """
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


def run_simulation(scenario, N):
    """
    Ejecuta la Simulación de Monte Carlo (Objetivo 4).
    """
    logging.info(f"Iniciando Simulación de Monte Carlo para: {scenario['name']}")
    logging.info(f"Número de simulaciones (N): {N}")

    start_time = time.time()

    params = scenario['params']

    # 1. Generar N muestras para cada Variable Aleatoria (RV)
    logging.info("Generando muestras aleatorias (Obj. 3)...")

    # Resistencia (R)
    sigma_sy, scale_sy = get_lognorm_params(params["Sy"]["mean"], params["Sy"]["cov"])
    R_results = lognorm.rvs(s=sigma_sy, scale=scale_sy, size=N)

    # Variables de Solicitación (S)
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

    # Guardar inputs para análisis de sensibilidad
    inputs = {
        "PGV": PGV_samples, "Vs": Vs_samples, "Es": Es_samples,
        "Kt": Kt_samples, "theta": theta_samples_deg
    }

    # 2. Ejecutar Modelo Determinista (Vectorizado) (Obj. 1 & 2)
    logging.info("Ejecutando modelo determinista vectorizado (Obj. 1 y 2)...")

    # Extraer constantes
    EpAp = DETERMINISTIC_CONSTANTS["EpAp"]
    sigma_h = DETERMINISTIC_CONSTANTS["sigma_h"]
    D = DETERMINISTIC_CONSTANTS["D"]
    nu_s = DETERMINISTIC_CONSTANTS["nu_s"]
    f = DETERMINISTIC_CONSTANTS["f"]
    Ep = DETERMINISTIC_CONSTANTS["Ep"]

    # --- Cálculo vectorizado ---
    Gs_samples = Es_samples / (2 * (1 + nu_s))
    k_ax_samples = np.pi * D * Gs_samples

    epsilon_g_ax_samples = (PGV_samples / Vs_samples) * (np.cos(theta_samples_rad) ** 2)
    k_w_samples = (2 * np.pi * f) / Vs_samples

    termino_rigidez = (EpAp * k_w_samples ** 2) / k_ax_samples
    epsilon_p_ax_samples = epsilon_g_ax_samples * (1 / (1 + termino_rigidez))

    sigma_ax_samples = Ep * epsilon_p_ax_samples

    sigma_x_weld_samples = Kt_samples * sigma_ax_samples

    # Ec. 15 (Von Mises) - Vectorizada
    S_results = np.sqrt(
        sigma_x_weld_samples ** 2 +
        sigma_h ** 2 -
        (sigma_x_weld_samples * sigma_h)
    )
    # --- Fin del cálculo vectorizado ---

    # 3. Evaluar Falla (Obj. 5)
    logging.info("Evaluando probabilidad de falla (Obj. 5)...")

    fallas = S_results >= R_results
    Nf = np.sum(fallas)
    Pf = Nf / N

    # Índice de Confiabilidad (Beta) - Ec. 26
    beta = -norm.ppf(Pf) if 0 < Pf < 1 else (6 if Pf == 0 else -6)

    # 4. Calcular Sensibilidad Probabilística
    logging.info("Calculando sensibilidad probabilística...")
    sensitivity = {}
    try:
        # Usamos Spearman (correlación de rango) porque es no paramétrico
        # y funciona bien con relaciones no lineales.
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
        "R_mean": np.mean(R_results)
    }

    return results, sensitivity, S_results, R_results


# ==============================================================================
# --- Handler para Logging en GUI ---
# ==============================================================================

class TextHandler(logging.Handler):
    """Manejador de logging personalizado para enviar logs a un widget de texto de Tkinter."""

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

class AdvancedPipelineAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super Simulador de Tesis - E. Ancco (TGD Probabilístico y Determinista)")
        self.root.geometry("1400x900")  # Tamaño aumentado para más gráficos

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TNotebook.Tab", font=('Calibri', 10, 'bold'))
        self.style.configure("Treeview.Heading", font=('Calibri', 9, 'bold'))

        # --- Variables de estado ---
        self.S_results = None
        self.R_results = None
        self.current_scenario_name = tk.StringVar()
        self.N_simulations = tk.StringVar(value="1000000")  # 1 millón por defecto

        # --- Layout Principal (Panel Izquierdo: Control, Panel Derecho: Resultados) ---
        main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Panel Izquierdo (Controles) ---
        control_frame = ttk.Labelframe(main_paned_window, text="Configuración de Simulación", padding=10)
        control_frame.pack(fill="y", side="left")  # No expandir horizontalmente
        main_paned_window.add(control_frame, weight=0)  # Peso 0 para tamaño fijo

        ttk.Label(control_frame, text="Seleccionar Escenario:", font=('Calibri', 10, 'bold')).pack(pady=(0, 5))
        self.scenario_menu = ttk.Combobox(
            control_frame,
            textvariable=self.current_scenario_name,
            values=list(SCENARIOS.keys()),
            state="readonly",
            width=35  # Ancho aumentado
        )
        self.scenario_menu.pack(fill=tk.X, padx=5, pady=5)
        self.scenario_menu.bind("<<ComboboxSelected>>", self.on_scenario_select)

        # Frame para mostrar parámetros
        self.params_frame = ttk.Frame(control_frame, padding=5)
        self.params_frame.pack(fill=tk.X, pady=5)
        self.param_labels = {}
        param_keys_to_display = ["PGV", "Vs", "Es", "Kt", "Sy", "theta"]
        for i, key in enumerate(param_keys_to_display):
            ttk.Label(self.params_frame, text=f"{key}:").grid(row=i, column=0, sticky='w')
            self.param_labels[key] = ttk.Label(self.params_frame, text="-", font=('Calibri', 9, 'italic'))
            self.param_labels[key].grid(row=i, column=1, sticky='w', padx=5)

        ttk.Label(control_frame, text="Número de Simulaciones (N):", font=('Calibri', 10, 'bold')).pack(pady=(10, 5))
        ttk.Entry(control_frame, textvariable=self.N_simulations).pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(control_frame, text="Ejecutar Simulación", command=self.run_simulation_and_plot).pack(pady=20,
                                                                                                         fill=tk.X,
                                                                                                         ipady=5)

        self.export_button = ttk.Button(control_frame, text="Exportar Datos (R, S) a CSV", command=self.export_to_csv,
                                        state=tk.DISABLED)
        self.export_button.pack(pady=10, fill=tk.X)

        # --- Panel Derecho (Resultados con Pestañas) ---
        results_frame = ttk.Labelframe(main_paned_window, text="Resultados del Análisis", padding=10)
        results_frame.pack(fill="both", expand=True, side="right")
        main_paned_window.add(results_frame, weight=3)  # Peso 3 para expandir

        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña 1: Resumen de Confiabilidad
        self.tab_summary = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_summary, text="Resumen de Confiabilidad")
        self.create_summary_tab()

        # Pestaña 2: Histograma R vs S
        self.tab_histogram = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_histogram, text="Gráfico de Distribución (R vs S)")
        self.fig_hist, self.ax_hist = plt.subplots(figsize=(7, 5), constrained_layout=True)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=self.tab_histogram)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 3: Tablas de Cálculo Determinista
        self.tab_tables = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_tables, text="Tablas de Cálculo Determinista")
        self.create_table_tab()

        # Pestaña 4: Sensibilidad Determinista (Gráficos)
        self.tab_sens_det = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_sens_det, text="Sensibilidad Determinista")
        # Layout de 2x2 para los gráficos de sensibilidad
        self.fig_sens_det, self.ax_sens_det = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        self.canvas_sens_det = FigureCanvasTkAgg(self.fig_sens_det, master=self.tab_sens_det)
        self.canvas_sens_det.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 5: Sensibilidad Probabilística (Tornado)
        self.tab_sens_prob = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_sens_prob, text="Sensibilidad Probabilística")
        self.fig_sens_prob, self.ax_sens_prob = plt.subplots(figsize=(7, 5), constrained_layout=True)
        self.canvas_sens_prob = FigureCanvasTkAgg(self.fig_sens_prob, master=self.tab_sens_prob)
        self.canvas_sens_prob.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña 6: Log de Simulación
        self.tab_log = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_log, text="Log de Simulación")
        self.log_text = scrolledtext.ScrolledText(self.tab_log, state='disabled', wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        # Configurar el manejador de logging
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        logging.getLogger().setLevel(logging.INFO)

        # --- Inicializar ---
        self.scenario_menu.current(0)
        self.on_scenario_select()  # Carga los gráficos y tablas iniciales

    def create_summary_tab(self):
        frame = self.tab_summary
        font_title = ('Calibri', 12, 'bold')
        font_value = ('Calibri', 12)

        ttk.Label(frame, text="Probabilidad de Falla (Pf):", font=font_title).grid(row=0, column=0, sticky='w', pady=5)
        self.pf_label = ttk.Label(frame, text="-", font=font_value)
        self.pf_label.grid(row=0, column=1, sticky='w', padx=10)

        ttk.Label(frame, text="Índice de Confiabilidad (Beta):", font=font_title).grid(row=1, column=0, sticky='w',
                                                                                       pady=5)
        self.beta_label = ttk.Label(frame, text="-", font=font_value)
        self.beta_label.grid(row=1, column=1, sticky='w', padx=10)

        ttk.Label(frame, text="Total Fallas (Nf):", font=font_title).grid(row=2, column=0, sticky='w', pady=5)
        self.nf_label = ttk.Label(frame, text="-", font=font_value)
        self.nf_label.grid(row=2, column=1, sticky='w', padx=10)

        ttk.Label(frame, text="Total Simulaciones (N):", font=font_title).grid(row=3, column=0, sticky='w', pady=5)
        self.n_label = ttk.Label(frame, text="-", font=font_value)
        self.n_label.grid(row=3, column=1, sticky='w', padx=10)

        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky='ew', pady=15)

        ttk.Label(frame, text="Media Solicitación E[S]:", font=font_title).grid(row=5, column=0, sticky='w', pady=5)
        self.s_mean_label = ttk.Label(frame, text="-", font=font_value)
        self.s_mean_label.grid(row=5, column=1, sticky='w', padx=10)

        ttk.Label(frame, text="Media Resistencia E[R]:", font=font_title).grid(row=6, column=0, sticky='w', pady=5)
        self.r_mean_label = ttk.Label(frame, text="-", font=font_value)
        self.r_mean_label.grid(row=6, column=1, sticky='w', padx=10)

    def create_table_tab(self):
        frame = self.tab_tables
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        cols = (
            "theta", "epsilon_g_ax", "termino_axial", "epsilon_p_ax",
            "sigma_ax", "sigma_x_weld", "S_MPa", "FS"
        )
        col_names = {
            "theta": "θ (°)", "epsilon_g_ax": "ε_g,ax", "termino_axial": "Térm. Rigidez",
            "epsilon_p_ax": "ε_p,ax", "sigma_ax": "σ_ax (MPa)", "sigma_x_weld": "σ_x,weld (MPa)",
            "S_MPa": "S (MPa)", "FS": "FS (R/S)"
        }

        self.table_tree = ttk.Treeview(frame, columns=cols, show="headings")

        for col in cols:
            self.table_tree.heading(col, text=col_names[col])
            self.table_tree.column(col, anchor='e', width=100)

        self.table_tree.grid(row=0, column=0, sticky='nsew')

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.table_tree.yview)
        self.table_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky='ns')

    def on_scenario_select(self, event=None):
        """Actualiza la UI cuando se cambia el escenario."""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            return

        params = SCENARIOS[scenario_name]
        for key, p in params.items():
            if key in self.param_labels:
                if p['dist'] == 'lognorm':
                    self.param_labels[key].config(text=f"μ={p['mean']}, COV={p['cov']}")
                elif p['dist'] == 'uniform':
                    self.param_labels[key].config(text=f"[{p['min']}, {p['max']}]")

        # Actualizar la tabla determinista
        self.update_deterministic_table()
        # Actualizar los gráficos deterministas
        self.plot_deterministic_sensitivity()

    def update_deterministic_table(self):
        """Pobla la pestaña de tablas con los datos del escenario seleccionado."""
        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            return

        # Limpiar tabla anterior
        for item in self.table_tree.get_children():
            self.table_tree.delete(item)

        scenario_params = SCENARIOS[scenario_name]
        table_data = calculate_deterministic_table_data(scenario_params)

        for row in table_data:
            values = (
                f"{row['theta']:.0f}",
                f"{row['epsilon_g_ax']:.3e}",
                f"{row['termino_axial']:.4f}",
                f"{row['epsilon_p_ax']:.3e}",
                f"{row['sigma_ax']:.1f}",
                f"{row['sigma_x_weld']:.1f}",
                f"{row['S_MPa']:.1f}",
                f"{row['FS']:.2f}"
            )
            self.table_tree.insert("", "end", values=values)

    def run_simulation_and_plot(self):
        try:
            N_str = self.N_simulations.get()
            N = int(N_str)
            if N <= 0:
                raise ValueError("N debe ser positivo")
        except ValueError:
            messagebox.showerror("Error de Entrada", "El número de simulaciones (N) debe ser un entero positivo.")
            return

        scenario_name = self.current_scenario_name.get()
        if not scenario_name:
            messagebox.showerror("Error de Entrada", "Por favor, seleccione un escenario.")
            return

        scenario = {"name": scenario_name, "params": SCENARIOS[scenario_name]}

        try:
            # Ejecutar simulación
            results, sensitivity, S_results, R_results = run_simulation(scenario, N)

            self.S_results = S_results
            self.R_results = R_results

            # Actualizar Pestaña 1: Resumen
            self.pf_label.config(text=f"{results['Pf']:.4e}")
            self.beta_label.config(text=f"{results['beta']:.4f}")
            self.nf_label.config(text=f"{results['Nf']:,}")
            self.n_label.config(text=f"{results['N']:,}")
            self.s_mean_label.config(text=f"{results['S_mean'] / 1e6:.2f} MPa")
            self.r_mean_label.config(text=f"{results['R_mean'] / 1e6:.2f} MPa")

            # Actualizar Pestaña 2: Histograma
            self.plot_histogram(S_results, R_results, results['S_mean'], results['R_mean'])

            # Actualizar Pestaña 5: Sensibilidad Probabilística
            self.plot_probabilistic_sensitivity(sensitivity)

            # Activar botón de exportar
            self.export_button.config(state=tk.NORMAL)

            messagebox.showinfo("Simulación Completa",
                                f"Simulación de {N:,} iteraciones completada.\nPf = {results['Pf']:.4e}\nBeta = {results['beta']:.4f}")
            self.notebook.select(self.tab_summary)  # Cambiar a la pestaña de resumen

        except Exception as e:
            logging.error(f"Error durante la simulación: {e}")
            messagebox.showerror("Error de Simulación", f"Ocurrió un error: {e}")

    def plot_histogram(self, S_data, R_data, S_mean, R_mean):
        """Actualiza el gráfico de histograma R vs S."""
        self.ax_hist.clear()

        # Graficar histogramas de densidad
        self.ax_hist.hist(S_data / 1e6, bins=50, density=True, color='red', alpha=0.7, label='Solicitación (S)')
        self.ax_hist.hist(R_data / 1e6, bins=50, density=True, color='blue', alpha=0.7, label='Resistencia (R)')

        # Líneas de media
        self.ax_hist.axvline(S_mean / 1e6, color='red', linestyle='--', label=f'Media S ({S_mean / 1e6:.1f} MPa)')
        self.ax_hist.axvline(R_mean / 1e6, color='blue', linestyle='--', label=f'Media R ({R_mean / 1e6:.1f} MPa)')

        self.ax_hist.set_title(
            f"Distribución de Resistencia (R) vs. Solicitación (S)\n{self.current_scenario_name.get()}")
        self.ax_hist.set_xlabel("Esfuerzo (MPa)")
        self.ax_hist.set_ylabel("Densidad de Probabilidad")
        self.ax_hist.legend()
        self.ax_hist.grid(True, linestyle=':', alpha=0.6)
        self.ax_hist.set_xlim(left=0)
        self.canvas_hist.draw()

    def plot_deterministic_sensitivity(self):
        """Grafica los análisis de sensibilidad determinista (FS vs Theta y FS vs Vs)."""
        # Limpiar todos los ejes
        for ax in self.ax_sens_det.flat:
            ax.clear()

        # --- Gráfico 1 (0,0): FS vs. Ángulo (theta) ---
        ax1 = self.ax_sens_det[0, 0]
        for scenario_name, params in SCENARIOS.items():
            table_data = calculate_deterministic_table_data(params)
            thetas = [row['theta'] for row in table_data]
            fss = [row['FS'] for row in table_data]
            ax1.plot(thetas, fss, 'o-', label=scenario_name.split(" ")[1])  # Etiqueta corta

        ax1.set_title("Sensibilidad: FS vs. Ángulo ($\theta$)")
        ax1.set_xlabel("Ángulo de Incidencia $\\theta$ (°)")
        ax1.set_ylabel("Factor de Seguridad (FS)")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.axhline(1.0, color='red', linestyle='--', lw=1)

        # --- Gráfico 2 (0,1): FS vs. Velocidad de Onda (Vs) ---
        ax2 = self.ax_sens_det[0, 1]
        base_params = {
            "PGV": SCENARIOS["Caso Surco (Tipo C)"]["PGV"]["mean"],
            "Es": SCENARIOS["Caso Surco (Tipo C)"]["Es"]["mean"],
            "Kt": SCENARIOS["Caso Surco (Tipo C)"]["Kt"]["mean"],
            "Sy": SCENARIOS["Caso Surco (Tipo C)"]["Sy"]["mean"],
            "theta": 30.0  # Ángulo supuesto
        }
        vs_range = np.linspace(200, 600, 20)
        fs_vs_results = []
        for vs in vs_range:
            params = base_params.copy()
            params["Vs"] = vs
            params["Es"] = (vs / 400.0) ** 2 * SCENARIOS["Caso Surco (Tipo C)"]["Es"][
                "mean"]  # G = rho * Vs^2 => E ~ Vs^2
            res = calculate_single_point_fs(params)
            fs_vs_results.append(res['FS'])

        ax2.plot(vs_range, fs_vs_results, 'o-', color='C0')
        ax2.set_title("Sensibilidad: FS vs. $V_s$ (en Suelo Tipo C)")
        ax2.set_xlabel("Velocidad de Onda de Corte $V_s$ (m/s)")
        ax2.set_ylabel("Factor de Seguridad (FS)")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.axhline(1.0, color='red', linestyle='--', lw=1)

        # --- Gráfico 3 (1,0): Heatmap FS vs (PGV, Vs) ---
        ax3 = self.ax_sens_det[1, 0]
        pgv_vec = np.linspace(0.2, 0.8, 20)
        vs_vec = np.linspace(200, 500, 20)
        pgv_grid, vs_grid = np.meshgrid(pgv_vec, vs_vec)
        fs_grid = np.zeros_like(pgv_grid)

        base_params_hm = {
            "Kt": SCENARIOS["Caso Surco (Tipo C)"]["Kt"]["mean"],
            "Sy": SCENARIOS["Caso Surco (Tipo C)"]["Sy"]["mean"],
            "theta": 30.0  # Ángulo supuesto
        }

        for i in range(pgv_grid.shape[0]):
            for j in range(pgv_grid.shape[1]):
                params = base_params_hm.copy()
                params["PGV"] = pgv_grid[i, j]
                params["Vs"] = vs_grid[i, j]
                params["Es"] = (params["Vs"] / 400.0) ** 2 * SCENARIOS["Caso Surco (Tipo C)"]["Es"]["mean"]
                res = calculate_single_point_fs(params)
                fs_grid[i, j] = res['FS']

        c = ax3.contourf(pgv_grid, vs_grid, fs_grid, levels=np.linspace(0, 3, 16), cmap='RdYlGn', extend='max')
        CS = ax3.contour(pgv_grid, vs_grid, fs_grid, levels=[1.0], colors='black', linestyles='--')
        ax3.clabel(CS, inline=1, fontsize=10, fmt='FS = 1.0')
        plt.colorbar(c, ax=ax3, label='Factor de Seguridad (FS)')
        ax3.set_title("Mapa de Falla Determinista (FS)")
        ax3.set_xlabel("Velocidad Pico del Suelo (PGV) (m/s)")
        ax3.set_ylabel("Velocidad de Onda de Corte ($V_s$) (m/s)")

        # --- Gráfico 4 (1,1): Vacío por ahora ---
        ax4 = self.ax_sens_det[1, 1]
        ax4.text(0.5, 0.5, 'Espacio para futuro gráfico\n(ej. Sensibilidad a Kt o Es)',
                 ha='center', va='center', style='italic', color='gray')
        ax4.set_xticks([])
        ax4.set_yticks([])

        self.canvas_sens_det.draw()

    def plot_probabilistic_sensitivity(self, sensitivity):
        """Grafica el análisis de sensibilidad probabilística (Tornado Plot)."""
        self.ax_sens_prob.clear()

        if not sensitivity:
            self.ax_sens_prob.text(0.5, 0.5, "Ejecute la simulación para ver la sensibilidad.", ha='center',
                                   style='italic', color='gray')
            self.canvas_sens_prob.draw()
            return

        # Ordenar por valor absoluto para el gráfico de tornado
        sorted_sens = sorted(sensitivity.items(), key=lambda item: abs(item[1]), reverse=True)
        labels = [item[0] for item in sorted_sens]
        values = [item[1] for item in sorted_sens]

        colors = ['red' if v < 0 else 'blue' for v in values]

        self.ax_sens_prob.barh(labels, values, color=colors)
        self.ax_sens_prob.set_title(
            f"Sensibilidad Probabilística (Correlación de Spearman)\n{self.current_scenario_name.get()}")
        self.ax_sens_prob.set_xlabel("Coeficiente de Correlación con la Solicitación (S)")
        self.ax_sens_prob.invert_yaxis()  # El más importante arriba
        self.ax_sens_prob.grid(True, axis='x', linestyle=':', alpha=0.6)

        # Añadir etiquetas de valor
        for index, value in enumerate(values):
            self.ax_sens_prob.text(value + (0.01 if value > 0 else -0.01), index, f"{value:.3f}",
                                   va='center', ha='left' if value > 0 else 'right')

        self.canvas_sens_prob.draw()

    def export_to_csv(self):
        """Exporta los datos crudos de R y S a un archivo CSV."""
        if self.S_results is None or self.R_results is None:
            messagebox.showwarning("Sin Datos",
                                   "No hay datos de simulación para exportar. Por favor, ejecute una simulación primero.")
            return

        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
                title="Guardar Datos de Simulación (R, S)"
            )
            if not filepath:
                return  # El usuario canceló

            logging.info(f"Exportando {len(self.S_results)} puntos de datos a {filepath}...")
            df = pd.DataFrame({
                "Resistencia (R_Pa)": self.R_results,
                "Solicitacion (S_Pa)": self.S_results,
                "Resistencia (R_MPa)": self.R_results / 1e6,
                "Solicitacion (S_MPa)": self.S_results / 1e6
            })
            df.to_csv(filepath, index=False, float_format='%.4f')
            logging.info("Exportación completada.")
            messagebox.showinfo("Exportación Completa",
                                f"Los datos (R, S) se han guardado exitosamente en:\n{filepath}")

        except Exception as e:
            logging.error(f"Error durante la exportación a CSV: {e}")
            messagebox.showerror("Error de Exportación", f"No se pudo guardar el archivo.\nError: {e}")


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPipelineAnalysisApp(root)
    root.mainloop()