# =================================================================
# SIMULADOR PROBABILÍSTICO DE GASODUCTOS (SMC) CON GUI
# Autor: Emanuel Edgar Ancco Guaygua (Asistido por Gemini)
# Fecha: 28 de Octubre de 2025
#
# Descripción:
# Interfaz gráfica (Tkinter) para el Modelo Dinámico-Probabilístico
# (Simulación de Monte Carlo) desarrollado para la tesis. Permite
# ingresar los parámetros probabilísticos y visualizar la Pf y Beta.
#
# Dependencias: numpy, scipy, matplotlib, tkinter (incorporada en Python)
# =================================================================

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import logging
import os

# --- 1. CONFIGURACIÓN DE LOGGING (para auditoría y depuración) ---
# Usamos un archivo de log temporal en el mismo directorio.
LOG_FILE = "simulacion_smc_gui.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logging.info("Iniciando la aplicación del Simulador Probabilístico.")


# --- 2. CLASE PRINCIPAL DE LA APLICACIÓN ---

class ProbabilisticSimulatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Modelo Dinámico-Probabilístico de Gasoductos")

        # --- Parámetros Deterministas (Constantes del Caso Surco) ---
        self.D = 0.762  # Diámetro exterior (m)
        self.t = 0.0095  # Espesor de pared (m)
        self.E_p = 2.07e11  # Módulo de Young (Pa)
        self.NU_SUELO = 0.3  # Coef. de Poisson del suelo
        self.SIGMA_H = 150e6  # Esfuerzo tangencial por presión (Pa)
        self.f = 1.0  # Frecuencia dominante del sismo (Hz)

        # Cálculos de área e inercia (constantes)
        self.A_p = np.pi * (self.D - self.t) * self.t
        self.I_p = (np.pi / 64) * (self.D ** 4 - (self.D - 2 * self.t) ** 4)

        # --- Variables de la Simulación ---
        self.n_simulations = tk.IntVar(value=1000000)
        self.last_results = {}

        # Configurar la interfaz
        self._setup_ui()

    def _setup_ui(self):
        # Frame principal para los inputs
        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.pack(fill='both', expand=True)

        # Título
        ttk.Label(main_frame, text="Configuración de la Simulación de Monte Carlo", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=3, pady=10)

        # Número de simulaciones
        ttk.Label(main_frame, text="Número de Iteraciones (N):", font=("Arial", 10, "bold")).grid(row=1, column=0,
                                                                                                  padx=5, pady=5,
                                                                                                  sticky='w')
        entry_N = ttk.Entry(main_frame, textvariable=self.n_simulations, width=15, justify='center')
        entry_N.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Separador para parámetros
        ttk.Separator(main_frame, orient='horizontal').grid(row=2, column=0, columnspan=3, sticky='ew', pady=10)

        # Frame para la tabla de parámetros (basado en Tabla 3 del LaTeX)
        params_frame = ttk.LabelFrame(main_frame, text="Variables Aleatorias (Distribución Lognormal)", padding="10")
        params_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

        # Encabezados de la tabla de parámetros
        headers = ["Variable", "Símbolo", "Media ($\mu$)", "COV ($\\sigma/\mu$)", "Unidad"]
        for col, header in enumerate(headers):
            ttk.Label(params_frame, text=header, font=("Arial", 9, "bold")).grid(row=0, column=col, padx=5, pady=3,
                                                                                 sticky='w')

        # Diccionario para almacenar las variables de entrada
        self.var_entries = {}

        # Definición de variables de entrada (Basado en la Tabla 3 del documento LaTeX)
        var_data = [
            ("PGV", "PGV", 0.5, 0.30, "m/s"),
            ("Velocidad de Onda de Corte", "V_s", 300.0, 0.25, "m/s"),
            ("Módulo de Young del Suelo", "E_s", 50.0e6, 0.20, "Pa"),
            ("Factor de Concentración", "K_t", 1.8, 0.15, "Adimensional"),
            ("Límite de Fluencia (Resistencia)", "S_y", 448.2e6, 0.10, "Pa")
        ]

        for row, (name, symbol, default_mu, default_cov, unit) in enumerate(var_data):
            # Nombre y Símbolo
            ttk.Label(params_frame, text=name).grid(row=row + 1, column=0, padx=5, pady=3, sticky='w')
            ttk.Label(params_frame, text=symbol).grid(row=row + 1, column=1, padx=5, pady=3, sticky='w')

            # Media (Entry)
            mu_var = tk.DoubleVar(value=default_mu)
            entry_mu = ttk.Entry(params_frame, textvariable=mu_var, width=15, justify='center')
            entry_mu.grid(row=row + 1, column=2, padx=5, pady=3)

            # COV (Entry)
            cov_var = tk.DoubleVar(value=default_cov)
            entry_cov = ttk.Entry(params_frame, textvariable=cov_var, width=10, justify='center')
            entry_cov.grid(row=row + 1, column=3, padx=5, pady=3)

            # Unidad
            ttk.Label(params_frame, text=unit).grid(row=row + 1, column=4, padx=5, pady=3, sticky='w')

            self.var_entries[symbol] = {'mu': mu_var, 'cov': cov_var, 'unit': unit, 'name': name}

        # Separador
        ttk.Separator(main_frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky='ew', pady=10)

        # Botón de Ejecutar
        ttk.Button(main_frame, text="EJECUTAR SIMULACIÓN DE MONTE CARLO", command=self._start_simulation,
                   style='TButton').grid(row=5, column=0, columnspan=3, pady=10, sticky='ew')

        # --- Frame de Resultados ---
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de Confiabilidad (SMC)", padding="10")
        results_frame.grid(row=6, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

        # Labels de Resultados
        self.pf_var = tk.StringVar(value="P_f: N/A")
        self.beta_var = tk.StringVar(value="Beta (β): N/A")
        self.mean_s_var = tk.StringVar(value="Media Solicitación (MPa): N/A")
        self.mean_r_var = tk.StringVar(value="Media Resistencia (MPa): N/A")

        ttk.Label(results_frame, textvariable=self.pf_var, font=("Arial", 11, "bold")).grid(row=0, column=0, padx=5,
                                                                                            pady=5, sticky='w')
        ttk.Label(results_frame, textvariable=self.beta_var, font=("Arial", 11, "bold")).grid(row=0, column=1, padx=5,
                                                                                              pady=5, sticky='w')
        ttk.Label(results_frame, textvariable=self.mean_s_var).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Label(results_frame, textvariable=self.mean_r_var).grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # --- Frame de Gráficos ---
        graph_frame = ttk.Frame(self.master)
        graph_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # Crear la figura inicial de Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.ax.set_title("Distribución de Resistencia (R) vs. Solicitación (S)")
        self.ax.set_xlabel("Esfuerzo (MPa)")
        self.ax.set_ylabel("Densidad de Probabilidad")
        self.ax.grid(True)
        self.ax.text(0.5, 0.5, 'Presione "Ejecutar Simulación"', transform=self.ax.transAxes,
                     fontsize=14, color='gray', ha='center')

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)

    # --- 3. FUNCIONES AUXILIARES DE PROBABILIDAD (Modelo Lognormal) ---

    def _get_lognorm_params(self, mu, cov):
        """
        Convierte (media, cov) a los parámetros (shape, scale)
        requeridos por scipy.stats.lognorm, según Ec. 12 y 13 del LaTeX.
        """
        # Se asegura de que no haya COV = 0 para evitar log(1)=0 y división por cero
        cov = np.maximum(cov, 1e-6)

        sigma_ln_sq = np.log(cov ** 2 + 1)
        s = np.sqrt(sigma_ln_sq)  # s (shape), sigma_Y
        scale = mu / np.exp(sigma_ln_sq / 2)  # scale, exp(mu_Y)
        return s, scale

    def _get_input_data(self):
        """
        Recupera y valida todos los datos de entrada de la GUI.
        """
        data = {}
        try:
            data['N'] = self.n_simulations.get()
            if data['N'] <= 0 or data['N'] > 10000000:
                raise ValueError("N debe ser positivo y razonable (max 10M).")

            data['specs'] = {}
            for symbol, entry in self.var_entries.items():
                mu = entry['mu'].get()
                cov = entry['cov'].get()

                if mu <= 0 or cov < 0:
                    raise ValueError(f"Media ({symbol}) debe ser > 0 y COV debe ser >= 0.")

                data['specs'][symbol.lower()] = {'mu': mu, 'cov': cov}

            # Ángulo de Incidencia (Uniforme - Determinista en la SMC)
            data['specs']['theta'] = {'min': 0, 'max': np.pi / 2}  # 0 a 90 grados

        except Exception as e:
            messagebox.showerror("Error de Entrada",
                                 f"Datos inválidos en la configuración. Revise los números.\nError: {e}")
            logging.error(f"Error de validación de entrada: {e}")
            return None

        return data

    # --- 4. FUNCIÓN DE EJECUCIÓN (MOTOR SMC) ---

    def _run_smc_engine(self, n_samples, specs):
        """
        Implementa el algoritmo de SMC vectorizado. (Objetivos 1, 2, 4, 5)
        """
        logging.info(f"Iniciando SMC con N={n_samples}...")

        # --- Paso 1: Generar Muestras (Vectorizado) ---
        samples = {}
        for symbol, spec in specs.items():
            if symbol != 'theta':
                s, scale = self._get_lognorm_params(spec['mu'], spec['cov'])
                samples[symbol] = stats.lognorm.rvs(s=s, scale=scale, size=n_samples)

        # Generar Theta (Uniforme)
        samples['theta'] = stats.uniform.rvs(loc=specs['theta']['min'],
                                             scale=specs['theta']['max'] - specs['theta']['min'],
                                             size=n_samples)

        # --- Paso 2: Modelo Determinista (Vectorizado) ---

        # 2a. Cálculo de rigideces del suelo
        G_s = samples['e_s'] / (2 * (1 + self.NU_SUELO))
        k_ax = np.pi * self.D * G_s  # Rigidez axial (Ec. 6.2)

        # 2b. Solicitación sísmica (TGD)
        epsilon_g = (samples['pgv'] / samples['v_s']) * (np.cos(samples['theta']) ** 2)  # Ec. 4
        k_w = (2 * np.pi * self.f) / samples['v_s']  # Número de onda

        # 2c. Respuesta de la tubería (Ec. 11)
        termino_axial = (self.E_p * self.A_p * k_w ** 2) / k_ax
        epsilon_p = epsilon_g * (1 / (1 + termino_axial))

        # Esfuerzo axial nominal
        sigma_ax = self.E_p * epsilon_p

        # 2d. Concentración de Esfuerzos (Ec. 14)
        sigma_x_weld = samples['k_t'] * sigma_ax

        # 2e. Cálculo de Solicitación (S) (Von Mises - Ec. 16)
        S_solicitacion = np.sqrt(
            sigma_x_weld ** 2 + self.SIGMA_H ** 2 - (sigma_x_weld * self.SIGMA_H)
        )

        # Resistencia (R)
        R_resistencia = samples['s_y']

        # --- Paso 3: Evaluar Falla (LSF - Ec. 17) ---
        g_lsf = R_resistencia - S_solicitacion

        # Contar fallas
        n_fallas = np.sum(g_lsf <= 0)

        logging.info("Motor de cálculo vectorizado completado.")

        return n_fallas, S_solicitacion, R_resistencia

    # --- 5. FUNCIÓN DE INICIO Y RESULTADOS ---

    def _start_simulation(self):
        """
        Función principal que inicia la simulación y actualiza la GUI.
        """
        input_data = self._get_input_data()
        if input_data is None:
            return

        N = input_data['N']
        specs = input_data['specs']

        try:
            # Deshabilitar botón para evitar ejecuciones múltiples
            self.master.config(cursor="watch")
            self.master.update()

            # Ejecutar el motor SMC
            n_f, S_vec, R_vec = self._run_smc_engine(N, specs)

            # --- 5a. Análisis de Resultados (Objetivo 5) ---

            if n_f == 0:
                # Estimación conservadora si no hay fallas observadas
                pf = 1 / N
                beta = -stats.norm.ppf(pf)
                messagebox.showwarning("Simulación Exitosa",
                                       f"La simulación se completó con N={N}. No se observaron fallas. La Probabilidad de Falla es Pf < {pf:.2e}")
            else:
                pf = n_f / N
                beta = -stats.norm.ppf(pf)

            # Actualizar variables de la GUI (convertir a MPa para display)
            self.pf_var.set(f"P_f: {pf:.4e}")
            self.beta_var.set(f"Beta (β): {beta:.3f}")
            self.mean_s_var.set(f"Media Solicitación: {np.mean(S_vec) / 1e6:.2f} MPa")
            self.mean_r_var.set(f"Media Resistencia: {np.mean(R_vec) / 1e6:.2f} MPa")

            self.last_results = {'S': S_vec, 'R': R_vec, 'pf': pf, 'beta': beta, 'N': N}

            # --- 5b. Generación de Gráfico (Evidencia Visual) ---
            self._plot_results(S_vec / 1e6, R_vec / 1e6, pf, beta)

            logging.info(f"Resultados finales: Pf={pf:.4e}, Beta={beta:.3f}")

        except Exception as e:
            messagebox.showerror("Error Crítico de Simulación",
                                 f"Ocurrió un error inesperado durante el cálculo.\nDetalles: {e}")
            logging.error(f"Error en _start_simulation: {e}", exc_info=True)

        finally:
            # Habilitar botón y restaurar cursor
            self.master.config(cursor="")
            self.master.update()

    def _plot_results(self, S_mpa, R_mpa, pf, beta):
        """
        Genera el gráfico de distribución de R vs S.
        """
        self.ax.clear()  # Limpiar el eje anterior

        # Histograma de Solicitación (S)
        self.ax.hist(S_mpa, bins=100, alpha=0.7, label="Solicitación (S)", density=True, color="#EF4444")

        # Histograma de Resistencia (R)
        self.ax.hist(R_mpa, bins=100, alpha=0.7, label="Resistencia (R)", density=True, color="#2563EB")

        # Título y Ejes
        self.ax.set_title("Superposición de PDF: Resistencia (R) vs. Solicitación (S)", fontsize=14)
        self.ax.set_xlabel("Esfuerzo (MPa)", fontsize=11)
        self.ax.set_ylabel("Densidad de Probabilidad", fontsize=11)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Determinar el área de falla (cruces de la PDF)
        min_val = min(S_mpa.min(), R_mpa.min())
        max_val = max(S_mpa.max(), R_mpa.max())

        # Añadir texto de resultados en el gráfico
        stats_text = (
            f"$N = {self.last_results['N']:,}$\n"
            f"$P_f = {pf:.2e}$\n"
            f"$\\beta = {beta:.3f}$\n"
            f"E[S] = {np.mean(S_mpa):.2f} MPa"
        )
        self.ax.text(0.98, 0.95, stats_text, transform=self.ax.transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))

        self.canvas.draw()
        logging.info("Gráfico actualizado en la GUI.")


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    # Aseguramos la inicialización del motor gráfico y la app.
    root = tk.Tk()
    app = ProbabilisticSimulatorApp(root)
    root.mainloop()
