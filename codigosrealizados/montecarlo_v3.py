# ================================================================
# SIMULADOR INTEGRADO: DETERMINISTA (2D) Y PROBABILÍSTICO (SMC)
# v3.0 - Correcciones completas y datos precargados
# Autor: Emanuel Edgar Ancco Guaygua (refactor por asistente)
# Requisitos: numpy, scipy, matplotlib, tkinter, pandas (solo para exportar)
# ================================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats
import pandas as pd
import logging
import os

# ---------------- CONFIGURACIÓN GENERAL ----------------
LOG_FILE = "simulador_integrado.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()]
)

# Propiedades del acero (constantes del modelo)
E_ACERO = 2.07e11        # Pa
NU_ACERO = 0.30
ALPHA_T_ACERO = 1.2e-5
GAMMA_ACERO = 78500       # N/m^3
NU_SUELO = 0.30
F_SISMO = 1.0             # Hz (frecuencia dominante)

STEEL_GRADES = {
    "API 5L X42": 289.6e6, "API 5L X52": 358.5e6, "API 5L X60": 413.7e6,
    "API 5L X65": 448.2e6, "API 5L X70": 482.6e6, "API 5L X80": 551.6e6,
}

SCENARIOS = {
    "Surco (Tipo C)": {
        "D": 0.508, "t": 0.00953, "p": 50e5, "delta_T": -15, "H": 1.93,
        "gamma_sat": 20000, "K0": 0.5, "PGV": 0.50, "C": 400, "alpha_seismic": 1.0,
        "Sy_val": 448.2e6, "Kt_val": 1.8, "If": 1.5,
        "mu_PGV": 0.50, "cov_PGV": 0.30,
        "mu_Vs": 300.0, "cov_Vs": 0.25,
        "mu_Es": 50e6, "cov_Es": 0.20,
        "mu_Kt": 1.8, "cov_Kt": 0.15,
        "mu_Sy": 448.2e6, "cov_Sy": 0.10,
        "n_simulations": 300000
    },
    "Villa El Salvador (Tipo D)": {
        "D": 0.254, "t": 0.00635, "p": 10e5, "delta_T": -15, "H": 2.33,
        "gamma_sat": 20000, "K0": 0.5, "PGV": 0.60, "C": 250, "alpha_seismic": 1.0,
        "Sy_val": 448.2e6, "Kt_val": 1.8, "If": 1.5,
        "mu_PGV": 0.60, "cov_PGV": 0.30,
        "mu_Vs": 250.0, "cov_Vs": 0.25,
        "mu_Es": 30e6, "cov_Es": 0.30,
        "mu_Kt": 1.8, "cov_Kt": 0.15,
        "mu_Sy": 448.2e6, "cov_Sy": 0.10,
        "n_simulations": 300000
    },
    "Personalizado": {
        "D": 0.762, "t": 0.0095, "p": 50e5, "delta_T": -15, "H": 1.5,
        "gamma_sat": 20000, "K0": 0.5, "PGV": 0.5, "C": 400, "alpha_seismic": 1.0,
        "Sy_val": 448.2e6, "Kt_val": 1.8, "If": 1.5,
        "mu_PGV": 0.5, "cov_PGV": 0.30,
        "mu_Vs": 400.0, "cov_Vs": 0.25,
        "mu_Es": 70e6, "cov_Es": 0.20,
        "mu_Kt": 1.8, "cov_Kt": 0.15,
        "mu_Sy": 448.2e6, "cov_Sy": 0.10,
        "n_simulations": 150000
    }
}
DEFAULT_SCENARIO = "Villa El Salvador (Tipo D)"

# ---------------- UTILIDADES ----------------
def get_lognorm_params(mu, cov):
    """Convierte (mu, cov) a (shape s, scale) para scipy.stats.lognorm."""
    cov = max(float(cov), 0.0)
    if cov == 0.0:
        return 1e-12, float(mu)
    sigma2 = np.log(1.0 + cov**2)
    s = np.sqrt(sigma2)
    scale = mu / np.exp(0.5 * sigma2)  # = mu / sqrt(1+cov^2)
    return s, scale

def sample_lognormal(mu, cov, size, rng):
    if cov <= 0.0:
        return np.full(size, float(mu))
    s, scale = get_lognorm_params(mu, cov)
    return stats.lognorm(s=s, scale=scale).rvs(size=size, random_state=rng)

# ---------------- MOTOR DETERMINISTA (2D resumido) ----------------
def calculate_detailed_stress(params):
    """Modelo 2D resumido (mismo espíritu, simplificado y estable)."""
    D_e = params['D']; t = params['t']; p_i = params['p']; delta_T = params['delta_T']
    H = params['H']; gamma_sat = params['gamma_sat']; K0 = params['K0']
    PGV = params['PGV']; C = params['C']; alpha_seismic = params['alpha_seismic']
    Kt = params['Kt']; Sy = params['Sy']; If = params['If']; W_traffic = 35500*If

    R_e = D_e/2; R_i = R_e - t; D_i = 2*R_i
    if t <= 0 or D_i <= 0:
        raise ValueError("Geometría inválida (t<=0 o D_i<=0)")
    A_s = np.pi*(D_e - t)*t
    I = (np.pi/64.0)*(D_e**4 - (D_e-2*t)**4)
    S = I/(D_e/2)

    # Tensiones axiales uniformes
    sigma_h_p = (p_i*D_i)/(2*t)
    sigma_a_p = NU_ACERO*sigma_h_p
    sigma_a_T = E_ACERO*ALPHA_T_ACERO*delta_T
    sigma_a_w = E_ACERO*alpha_seismic*(PGV/max(C,1e-9))
    sigma_L_uniforme = sigma_a_p + sigma_a_T + sigma_a_w

    # Modelo anillo muy simplificado (solo para tener distribución angular estable)
    theta = np.deg2rad(np.arange(0,181,15))
    qv = gamma_sat*max(H - R_e, 0)
    M = (qv*R_e**2/8.0)*(1 - np.cos(2*theta))
    N = qv*R_e/2.0*np.sin(theta)
    sigma_h_flex = (N/t) + (M/S)
    sigma_h_flex_int = (N/t) - (M/S)

    sigma_L_flex_ext = NU_ACERO*sigma_h_flex
    sigma_L_flex_int = NU_ACERO*sigma_h_flex_int

    sigma_L_total_ext = sigma_L_uniforme + sigma_L_flex_ext
    sigma_L_total_int = sigma_L_uniforme + sigma_L_flex_int

    sigma_h_total_ext = sigma_h_p + sigma_h_flex
    sigma_h_total_int = sigma_h_p + sigma_h_flex_int

    # Von Mises en cuerpo liso
    vm_ext = np.sqrt(sigma_L_total_ext**2 - sigma_L_total_ext*sigma_h_total_ext + sigma_h_total_ext**2)
    vm_int = np.sqrt(sigma_L_total_int**2 - sigma_L_total_int*sigma_h_total_int + sigma_h_total_int**2)

    # Con SCF en axial
    sigma_L_SCF_uniforme = sigma_L_uniforme*Kt
    sigma_L_SCF_ext = sigma_L_SCF_uniforme + sigma_L_flex_ext
    sigma_L_SCF_int = sigma_L_SCF_uniforme + sigma_L_flex_int
    vm_SCF_ext = np.sqrt(sigma_L_SCF_ext**2 - sigma_L_SCF_ext*sigma_h_total_ext + sigma_h_total_ext**2)
    vm_SCF_int = np.sqrt(sigma_L_SCF_int**2 - sigma_L_SCF_int*sigma_h_total_int + sigma_h_total_int**2)

    return {
        "theta_deg": np.rad2deg(theta),
        "sigma_VM_ext": vm_ext, "sigma_VM_int": vm_int,
        "sigma_VM_ext_SCF": vm_SCF_ext, "sigma_VM_int_SCF": vm_SCF_int,
        "Sy": Sy, "Kt": Kt
    }

# ---------------- MOTOR MONTE CARLO (SMC corregido) ----------------
def run_smc_engine(n_samples, specs, D_e, t, seed=12345):
    """
    SÓLO muestrea: pgv, vs, es, kt, sy. D, t y pi se mantienen deterministas.
    """
    logging.info(f"SMC N={n_samples}")
    if n_samples <= 0:
        raise ValueError("N debe ser >0")
    D_i = D_e - 2*t
    if t <= 0 or D_i <= 0:
        raise ValueError("Geometría inválida (t<=0 o D_i<=0)")
    rng = np.random.default_rng(seed)

    # Validación entradas
    for k in ['pgv', 'vs', 'es', 'kt', 'sy']:
        if k not in specs or 'mu' not in specs[k] or 'cov' not in specs[k]:
            raise ValueError(f"Falta mu/cov para '{k}'")
        if specs[k]['mu'] <= 0 or specs[k]['cov'] < 0:
            raise ValueError(f"Parámetros inválidos en '{k}' (mu>0, cov>=0)")
    if specs['pi']['mu'] < 0:
        raise ValueError("p_i no puede ser negativa")

    # Área metálica
    A_p = np.pi * (D_e - t) * t

    # Muestras
    pgv = sample_lognormal(specs['pgv']['mu'], specs['pgv']['cov'], n_samples, rng)
    vs  = sample_lognormal(specs['vs']['mu'],  specs['vs']['cov'],  n_samples, rng)
    es  = sample_lognormal(specs['es']['mu'],  specs['es']['cov'],  n_samples, rng)
    kt  = sample_lognormal(specs['kt']['mu'],  specs['kt']['cov'],  n_samples, rng)
    sy  = sample_lognormal(specs['sy']['mu'],  specs['sy']['cov'],  n_samples, rng)

    # Evitar divisiones por cero/negativos
    vs = np.maximum(vs, 1e-6)
    es = np.maximum(es, 1e3)

    theta = stats.uniform(loc=0.0, scale=np.pi/2).rvs(size=n_samples, random_state=rng)

    # Dinámica (axial)
    G_s = es / (2*(1 + NU_SUELO))
    k_ax = np.pi * D_e * G_s
    epsilon_g = (pgv / vs) * (np.cos(theta)**2)
    k_w = (2*np.pi*F_SISMO) / vs
    termino_axial = (E_ACERO * A_p * k_w**2) / np.maximum(k_ax, 1e-9)
    epsilon_p = epsilon_g / (1 + termino_axial)
    sigma_ax = E_ACERO * epsilon_p

    # Tensión circunferencial por presión interna (determinista)
    P_i = float(specs['pi']['mu'])
    sigma_h_const = (P_i * D_i) / (2*t)

    # SCF en axial (soldadura)
    sigma_x_weld = kt * sigma_ax

    # Von Mises plano (σx = axial soldadura, σh = hoop)
    S = np.sqrt(sigma_x_weld**2 + sigma_h_const**2 - sigma_x_weld*sigma_h_const)
    R = sy

    g = R - S
    n_fallas = int(np.sum(g <= 0.0))

    return n_fallas, S, R, g

# ---------------- INTERFAZ ----------------
class IntegratedSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador Integrado: Determinista (2D) y Probabilístico (SMC)")
        self._setup_styles()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs
        self.tab_det = ttk.Frame(self.notebook, padding=10)
        self.tab_smc = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_det, text="1) Análisis Determinista (2D)")
        self.notebook.add(self.tab_smc, text="2) Análisis Probabilístico (SMC)")

        self._build_tab_det()
        self._build_tab_smc()

        # Carga inicial
        self._det_load(DEFAULT_SCENARIO)
        self._smc_load(DEFAULT_SCENARIO)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("SMC.TButton", font=('Arial', 11, 'bold'))
        style.configure("Pass.TLabel", font=('Arial', 14, 'bold'), foreground='white', background='#28A745', anchor='center')
        style.configure("Fail.TLabel", font=('Arial', 14, 'bold'), foreground='white', background='#DC3545', anchor='center')

    # ---------- TAB 1: Determinista ----------
    def _build_tab_det(self):
        self.det_vars = {}
        left = ttk.Frame(self.tab_det); right = ttk.Frame(self.tab_det)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        row=0
        ttk.Label(left, text="Escenario:", font=('Arial', 11, 'bold')).grid(row=row, column=0, sticky='w')
        self.det_scn = tk.StringVar(value=DEFAULT_SCENARIO)
        cb = ttk.Combobox(left, textvariable=self.det_scn, values=list(SCENARIOS.keys()), state="readonly")
        cb.grid(row=row, column=1, sticky='ew'); left.columnconfigure(1, weight=1)
        cb.bind("<<ComboboxSelected>>", lambda e: self._det_load(self.det_scn.get()))
        row+=1

        def add_row(key, label, unit, ro=False):
            nonlocal row
            ttk.Label(left, text=label).grid(row=row, column=0, sticky='w', pady=2)
            var = tk.StringVar()
            self.det_vars[key]=var
            e = ttk.Entry(left, textvariable=var, width=16, justify='center')
            if ro: e.state(['readonly'])
            e.grid(row=row, column=1, sticky='ew', pady=2)
            ttk.Label(left, text=unit).grid(row=row, column=2, sticky='w')
            row+=1

        add_row("D", "Diámetro exterior D", "m")
        add_row("t", "Espesor t", "m")
        add_row("p", "Presión interna p_i", "Pa")
        add_row("delta_T", "ΔT", "°C")
        add_row("H", "Prof. eje H", "m")
        add_row("gamma_sat", "γ_sat", "N/m³")
        add_row("K0", "K0", "-")
        add_row("PGV", "PGV", "m/s")
        add_row("C", "Velocidad de onda C", "m/s")
        add_row("alpha", "Factor sísmico α", "-")
        add_row("Kt", "SCF (K_t)", "-")
        add_row("If", "Factor de impacto (I_f)", "-")
        add_row("Sy", "Límite de fluencia Sy", "Pa", ro=True)

        ttk.Button(left, text="Calcular determinista", command=self._det_run).grid(row=row, column=0, columnspan=3, pady=10)

        self.det_status = ttk.Label(right, text="ESTADO: Pendiente", anchor='center')
        self.det_status.pack(fill=tk.X, pady=(0,10))

        self.det_fig, self.det_ax = plt.subplots(figsize=(6,6), subplot_kw={'projection':'polar'})
        self.det_canvas = FigureCanvasTkAgg(self.det_fig, master=right)
        self.det_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _det_load(self, scn):
        p = SCENARIOS[scn]
        self.det_vars["D"].set(f"{p['D']:.6g}")
        self.det_vars["t"].set(f"{p['t']:.6g}")
        self.det_vars["p"].set(f"{p['p']:.6g}")
        self.det_vars["delta_T"].set(f"{p['delta_T']:.6g}")
        self.det_vars["H"].set(f"{p['H']:.6g}")
        self.det_vars["gamma_sat"].set(f"{p['gamma_sat']:.6g}")
        self.det_vars["K0"].set(f"{p['K0']:.6g}")
        self.det_vars["PGV"].set(f"{p['PGV']:.6g}")
        self.det_vars["C"].set(f"{p['C']:.6g}")
        self.det_vars["alpha"].set(f"{p['alpha_seismic']:.6g}")
        self.det_vars["Kt"].set(f"{p['Kt_val']:.6g}")
        self.det_vars["If"].set(f"{p['If']:.6g}")
        self.det_vars["Sy"].set(f"{p['Sy_val']:.6g}")

    def _det_run(self):
        try:
            params = {
                'D': float(self.det_vars['D'].get()),
                't': float(self.det_vars['t'].get()),
                'p': float(self.det_vars['p'].get()),
                'delta_T': float(self.det_vars['delta_T'].get()),
                'H': float(self.det_vars['H'].get()),
                'gamma_sat': float(self.det_vars['gamma_sat'].get()),
                'K0': float(self.det_vars['K0'].get()),
                'PGV': float(self.det_vars['PGV'].get()),
                'C': float(self.det_vars['C'].get()),
                'alpha_seismic': float(self.det_vars['alpha'].get()),
                'Kt': float(self.det_vars['Kt'].get()),
                'If': float(self.det_vars['If'].get()),
                'Sy': float(self.det_vars['Sy'].get())
            }
            res = calculate_detailed_stress(params)
            vm = np.maximum(res['sigma_VM_ext_SCF'], res['sigma_VM_int_SCF'])
            Sy = res['Sy']
            FS = Sy/np.max(vm) if np.max(vm)>0 else np.inf
            if FS >= 1.0:
                self.det_status.configure(text=f"CUMPLE (FS={FS:.2f})", style="Pass.TLabel")
            else:
                self.det_status.configure(text=f"NO CUMPLE (FS={FS:.2f})", style="Fail.TLabel")
            # Plot polar
            self.det_ax.clear()
            theta = np.deg2rad(res["theta_deg"])
            vm_full = np.concatenate([vm, vm[1:-1]])
            theta_full = np.concatenate([theta, theta[1:-1]+np.pi])
            self.det_ax.plot(theta_full, vm_full/1e6)
            self.det_ax.plot(np.linspace(0,2*np.pi,200), np.full(200, Sy/1e6), linestyle='--')
            self.det_ax.set_title("σ_VM con SCF (MPa)")
            self.det_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error determinista", str(e))
            logging.error(e, exc_info=True)

    # ---------- TAB 2: SMC ----------
    def _build_tab_smc(self):
        self.smc_vars = {}
        top = ttk.Frame(self.tab_smc); top.pack(fill=tk.X)
        ttk.Label(top, text="Escenario:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        self.smc_scn = tk.StringVar(value=DEFAULT_SCENARIO)
        cb = ttk.Combobox(top, textvariable=self.smc_scn, values=list(SCENARIOS.keys()), state="readonly")
        cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self._smc_load(self.smc_scn.get()))

        # Panel resultados
        self.lbl_pf = ttk.Label(self.tab_smc, text="P_f: N/A   β: N/A", font=('Arial', 11, 'bold'))
        self.lbl_pf.pack(anchor='w', pady=(8,2))
        self.lbl_means = ttk.Label(self.tab_smc, text="Media S: N/A | Media R: N/A", font=('Arial', 10))
        self.lbl_means.pack(anchor='w', pady=(0,8))

        # Botones
        btns = ttk.Frame(self.tab_smc); btns.pack(fill=tk.X, pady=5)
        ttk.Button(btns, text="Ejecutar SMC", command=self._smc_run, style="SMC.TButton").pack(side=tk.LEFT)
        ttk.Button(btns, text="Exportar CSV", command=self._smc_export).pack(side=tk.LEFT, padx=8)

        # Gráfico
        self.fig, self.ax = plt.subplots(figsize=(8,5))
        self.ax.set_title("Superposición de PDF: Resistencia (R) vs. Solicitación (S)")
        self.ax.set_xlabel("Esfuerzo (MPa)"); self.ax.set_ylabel("Densidad de Probabilidad")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_smc)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

        self.smc_last = None  # guardará S, R, g, pf, beta, N

    def _smc_load(self, scn):
        self.current = SCENARIOS[scn]

    def _smc_run(self):
        try:
            p = self.current
            N = int(p['n_simulations'])
            specs = {
                'pi': {'mu': p['p']},
                'pgv': {'mu': p['mu_PGV'], 'cov': p['cov_PGV']},
                'vs':  {'mu': p['mu_Vs'],  'cov': p['cov_Vs']},
                'es':  {'mu': p['mu_Es'],  'cov': p['cov_Es']},
                'kt':  {'mu': p['mu_Kt'],  'cov': p['cov_Kt']},
                'sy':  {'mu': p['mu_Sy'],  'cov': p['cov_Sy']}
            }
            self.root.config(cursor="watch"); self.root.update()
            n_f, S, R, g = run_smc_engine(N, specs, p['D'], p['t'], seed=12345)
            pf = n_f / N
            beta = -stats.norm.ppf(min(max(pf, 1e-16), 1-1e-16))
            self.smc_last = {"S": S, "R": R, "g": g, "pf": pf, "beta": beta, "N": N}
            self.lbl_pf.configure(text=f"P_f: {pf:.3e}   β: {beta:.3f}")
            self.lbl_means.configure(text=f"Media S: {np.mean(S)/1e6:.2f} MPa | Media R: {np.mean(R)/1e6:.2f} MPa")
            self._smc_plot(S/1e6, R/1e6, pf, beta)
        except Exception as e:
            messagebox.showerror("Error SMC", str(e))
            logging.error(e, exc_info=True)
        finally:
            self.root.config(cursor="")

    def _smc_plot(self, S_mpa, R_mpa, pf, beta):
        self.ax.clear()
        self.ax.set_title("Distribución Probabilística – Monte Carlo")
        self.ax.set_xlabel("Esfuerzo (MPa)"); self.ax.set_ylabel("Densidad de Probabilidad")
        self.ax.grid(True, linestyle='--', alpha=0.6)

        lo = min(S_mpa.min(), R_mpa.min())
        hi = max(S_mpa.max(), R_mpa.max())
        xs = np.linspace(lo, hi, 400)

        try:
            kde_S = stats.gaussian_kde(S_mpa)
            kde_R = stats.gaussian_kde(R_mpa)
            self.ax.plot(xs, kde_S(xs), label="Solicitación (S)", lw=2, color='red')
            self.ax.plot(xs, kde_R(xs), label="Resistencia (R)", lw=2, color='blue')
        except Exception:
            self.ax.hist(S_mpa, bins=80, density=True, alpha=0.6, label="S")
            self.ax.hist(R_mpa, bins=80, density=True, alpha=0.6, label="R")

        self.ax.legend(loc='upper left')
        self.ax.text(0.98, 0.95, f"P_f={pf:.2e}\nβ={beta:.3f}", transform=self.ax.transAxes,
                     ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        self.canvas.draw()

    def _smc_export(self):
        if not self.smc_last:
            messagebox.showinfo("Exportar", "Primero ejecuta una simulación.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="resultados_smc.csv"
        )
        if not path:
            return
        try:
            df = pd.DataFrame({
                "S_Pa": self.smc_last["S"],
                "R_Pa": self.smc_last["R"],
                "g_R_minus_S_Pa": self.smc_last["g"]
            })
            df.to_csv(path, index=False)
            messagebox.showinfo("Exportar", f"Archivo guardado:\n{os.path.abspath(path)}")
        except Exception as e:
            messagebox.showerror("Exportar", str(e))

# ---------------- MAIN ----------------
if __name__ == "__main__":
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    root = tk.Tk()
    root.geometry("1100x780")
    app = IntegratedSimulatorApp(root)
    root.mainloop()
