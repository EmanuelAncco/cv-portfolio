# ================================================================
# SIMULADOR INTEGRADO: DETERMINISTA (2D) Y PROBABILÍSTICO (SMC)
# Autor: Emanuel Edgar Ancco Guaygua
# Versión: 2.0 – Monte Carlo Profesional
# Fecha: 28/10/2025
# ================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import logging

# ---------------- CONFIGURACIÓN GENERAL ----------------
LOG_FILE = "simulador_integrado.log"
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)

E_ACERO = 2.07e11
NU_SUELO = 0.30
F_SISMO = 1.0

SCENARIOS = {
    "Villa El Salvador (Tipo D)": {
        "D": 0.254, "t": 0.00635, "p": 1e6,
        "mu_PGV": 0.60, "cov_PGV": 0.30,
        "mu_Vs": 250.0, "cov_Vs": 0.25,
        "mu_Es": 30e6, "cov_Es": 0.30,
        "mu_Kt": 1.8, "cov_Kt": 0.15,
        "mu_Sy": 448.2e6, "cov_Sy": 0.10,
        "n_simulations": 500000
    },
    "Surco (Tipo C)": {
        "D": 0.508, "t": 0.00953, "p": 5e6,
        "mu_PGV": 0.50, "cov_PGV": 0.30,
        "mu_Vs": 300.0, "cov_Vs": 0.25,
        "mu_Es": 50e6, "cov_Es": 0.20,
        "mu_Kt": 1.8, "cov_Kt": 0.15,
        "mu_Sy": 448.2e6, "cov_Sy": 0.10,
        "n_simulations": 500000
    },
    "Personalizado": {
        "D": 0.762, "t": 0.0095, "p": 5e6,
        "mu_PGV": 0.5, "cov_PGV": 0.30,
        "mu_Vs": 400.0, "cov_Vs": 0.25,
        "mu_Es": 70e6, "cov_Es": 0.20,
        "mu_Kt": 1.8, "cov_Kt": 0.15,
        "mu_Sy": 448.2e6, "cov_Sy": 0.10,
        "n_simulations": 200000
    }
}
DEFAULT_SCENARIO = "Villa El Salvador (Tipo D)"

# ---------------- FUNCIONES BASE ----------------
def get_lognorm_params(mu, cov):
    if cov <= 0:
        return 1e-12, mu
    sigma2 = np.log(1 + cov**2)
    s = np.sqrt(sigma2)
    scale = mu / np.sqrt(1 + cov**2)
    return s, scale

def _sample_lognormal(mu, cov, size, rng):
    if cov <= 0:
        return np.full(size, mu)
    s, scale = get_lognorm_params(mu, cov)
    return stats.lognorm(s=s, scale=scale).rvs(size=size, random_state=rng)

def run_smc_engine(n_samples, specs, D_e, t, seed=12345):
    rng = np.random.default_rng(seed)
    D_i = D_e - 2 * t
    if D_i <= 0:
        raise ValueError("Geometría inválida: D_i <= 0")

    A_p = np.pi * (D_e - t) * t

    # Generar muestras lognormales
    pgv = _sample_lognormal(specs['pgv']['mu'], specs['pgv']['cov'], n_samples, rng)
    vs = _sample_lognormal(specs['vs']['mu'], specs['vs']['cov'], n_samples, rng)
    es = _sample_lognormal(specs['es']['mu'], specs['es']['cov'], n_samples, rng)
    kt = _sample_lognormal(specs['kt']['mu'], specs['kt']['cov'], n_samples, rng)
    sy = _sample_lognormal(specs['sy']['mu'], specs['sy']['cov'], n_samples, rng)

    theta = stats.uniform(loc=0, scale=np.pi / 2).rvs(size=n_samples, random_state=rng)

    G_s = es / (2 * (1 + NU_SUELO))
    k_ax = np.pi * D_e * G_s
    epsilon_g = (pgv / vs) * (np.cos(theta) ** 2)
    k_w = (2 * np.pi * F_SISMO) / vs
    epsilon_p = epsilon_g / (1 + (E_ACERO * A_p * k_w**2) / np.maximum(k_ax, 1e-9))
    sigma_ax = E_ACERO * epsilon_p
    P_i = float(specs['pi']['mu'])
    SIGMA_H = (P_i * D_i) / (2 * t)

    sigma_x_weld = kt * sigma_ax
    S = np.sqrt(sigma_x_weld**2 + SIGMA_H**2 - sigma_x_weld * SIGMA_H)
    R = sy
    n_fallas = np.sum(R <= S)
    return n_fallas, S, R

# ---------------- INTERFAZ GRÁFICA ----------------
class MonteCarloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de Monte Carlo – Análisis Probabilístico de Tuberías")
        self._setup_ui()
        self._load_scenario(DEFAULT_SCENARIO)

    def _setup_ui(self):
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.frame, text="Escenario:", font=('Arial', 11, 'bold')).grid(row=0, column=0, sticky='w')
        self.scenario_var = tk.StringVar(value=DEFAULT_SCENARIO)
        self.combo = ttk.Combobox(self.frame, textvariable=self.scenario_var, values=list(SCENARIOS.keys()), state='readonly')
        self.combo.grid(row=0, column=1, sticky='ew')
        self.combo.bind("<<ComboboxSelected>>", self._on_scenario)

        ttk.Button(self.frame, text="Ejecutar Simulación Monte Carlo", command=self._run_smc, style='SMC.TButton').grid(row=1, column=0, columnspan=2, pady=10)

        self.result_pf = ttk.Label(self.frame, text="P_f: N/A", font=('Arial', 11, 'bold'))
        self.result_pf.grid(row=2, column=0, sticky='w', pady=5)
        self.result_beta = ttk.Label(self.frame, text="β: N/A", font=('Arial', 11, 'bold'))
        self.result_beta.grid(row=2, column=1, sticky='w', pady=5)

        self.result_means = ttk.Label(self.frame, text="Media S: N/A | Media R: N/A", font=('Arial', 10))
        self.result_means.grid(row=3, column=0, columnspan=2, pady=5)

        fig, ax = plt.subplots(figsize=(7, 5))
        self.ax = ax
        self.ax.set_title("Superposición de PDF: Resistencia (R) vs. Solicitación (S)")
        self.ax.set_xlabel("Esfuerzo (MPa)")
        self.ax.set_ylabel("Densidad de Probabilidad")
        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, sticky='nsew', pady=10)
        self.frame.rowconfigure(4, weight=1)
        self.frame.columnconfigure(1, weight=1)

    def _load_scenario(self, name):
        self.params = SCENARIOS[name]

    def _on_scenario(self, event=None):
        self._load_scenario(self.scenario_var.get())

    def _run_smc(self):
        try:
            N = int(self.params["n_simulations"])
            specs = {
                'pi': {'mu': self.params['p']},
                'pgv': {'mu': self.params['mu_PGV'], 'cov': self.params['cov_PGV']},
                'vs': {'mu': self.params['mu_Vs'], 'cov': self.params['cov_Vs']},
                'es': {'mu': self.params['mu_Es'], 'cov': self.params['cov_Es']},
                'kt': {'mu': self.params['mu_Kt'], 'cov': self.params['cov_Kt']},
                'sy': {'mu': self.params['mu_Sy'], 'cov': self.params['cov_Sy']}
            }
            n_f, S, R = run_smc_engine(N, specs, self.params['D'], self.params['t'])
            pf = n_f / N
            beta = -stats.norm.ppf(min(max(pf, 1e-16), 1 - 1e-16))
            self.result_pf.config(text=f"P_f: {pf:.3e}")
            self.result_beta.config(text=f"β: {beta:.3f}")
            self.result_means.config(text=f"Media S: {np.mean(S)/1e6:.2f} MPa | Media R: {np.mean(R)/1e6:.2f} MPa")
            self._plot(S / 1e6, R / 1e6, pf, beta)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error(e)

    def _plot(self, S, R, pf, beta):
        self.ax.clear()
        self.ax.set_title("Distribución Probabilística – Monte Carlo")
        self.ax.set_xlabel("Esfuerzo (MPa)")
        self.ax.set_ylabel("Densidad de Probabilidad")
        try:
            kde_S = stats.gaussian_kde(S)
            kde_R = stats.gaussian_kde(R)
            xs = np.linspace(min(S.min(), R.min()), max(S.max(), R.max()), 400)
            self.ax.plot(xs, kde_S(xs), label="Solicitación (S)", lw=2, color='red')
            self.ax.plot(xs, kde_R(xs), label="Resistencia (R)", lw=2, color='blue')
        except:
            self.ax.hist(S, bins=80, density=True, alpha=0.6, label='S')
            self.ax.hist(R, bins=80, density=True, alpha=0.6, label='R')
        self.ax.legend()
        self.ax.text(0.98, 0.95, f"P_f={pf:.2e}\nβ={beta:.3f}", transform=self.ax.transAxes,
                     ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        self.canvas.draw()

# ---------------- EJECUCIÓN ----------------
if __name__ == "__main__":
    plt.rcParams['mathtext.fontset'] = 'cm'
    root = tk.Tk()
    root.geometry("900x700")
    style = ttk.Style()
    style.configure("SMC.TButton", font=('Arial', 10, 'bold'), background='#007BFF', foreground='white')
    app = MonteCarloApp(root)
    root.mainloop()
