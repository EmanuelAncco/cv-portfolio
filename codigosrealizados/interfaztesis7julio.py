import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd  # Se necesita para exportar a Excel

# --- Constantes y Parámetros Globales ---
E_ACERO = 2.07e11  # Módulo de Young en Pa
NU_ACERO = 0.3  # Coeficiente de Poisson
ALPHA_T_ACERO = 1.2e-5  # Coeficiente de Expansión Térmica en °C⁻¹
GAMMA_ACERO = 78500  # Peso específico del acero en N/m³

# --- Diccionarios para Personalización ---
STEEL_GRADES = {
    "API 5L X42": 289.6e6,
    "API 5L X52": 358.5e6,
    "API 5L X60": 413.7e6,
    "API 5L X65": 448.2e6,
    "API 5L X70": 482.6e6,
    "API 5L X80": 551.6e6,
}
DEFAULT_STEEL = "API 5L X65"

VEHICLE_TYPES = {
    "AASHTO HS20 (71.2 kN)": 71170,
    "AASHTO HL-93 (72 kN)": 72000,
    "Camión Diseño T3-S2 (83 kN)": 83000,  # Eje delantero de 9t
    "Camión Pesado (Eje 11t - 108 kN)": 108000,
    "Personalizado": 35500
}
DEFAULT_VEHICLE = "Personalizado"

# --- Datos de los Casos de Estudio (Distritos) ---
LOCATION_PARAMS = {
    "Surco": {
        "D": 0.508, "t": 0.00953, "p": 50e5, "delta_T": -15, "H": 1.93,
        "gamma_sat": 20000, "K0": 0.5, "If": 1.5, "PGV": 0.50, "C": 400,
        "alpha_seismic": 1.0, "steel_grade": "API 5L X65", "vehicle": "Personalizado"
    },
    "Villa El Salvador": {
        "D": 0.254, "t": 0.00635, "p": 10e5, "delta_T": -15, "H": 2.33,
        "gamma_sat": 20000, "K0": 0.5, "If": 1.5, "PGV": 0.60, "C": 250,
        "alpha_seismic": 1.0, "steel_grade": "API 5L X65", "vehicle": "Personalizado"
    },
    "S.J. Lurigancho": {
        "D": 0.254, "t": 0.00635, "p": 10e5, "delta_T": -15, "H": 1.45,
        "gamma_sat": 20000, "K0": 0.5, "If": 1.5, "PGV": 0.40, "C": 400,
        "alpha_seismic": 1.0, "steel_grade": "API 5L X65", "vehicle": "Personalizado"
    },
    "Personalizado": {
        "D": 0.508, "t": 0.00953, "p": 50e5, "delta_T": -15, "H": 1.93,
        "gamma_sat": 20000, "K0": 0.5, "If": 1.5, "PGV": 0.50, "C": 400,
        "alpha_seismic": 1.0, "steel_grade": "API 5L X65", "vehicle": "Personalizado"
    }
}
DEFAULT_LOCATION = "Surco"


# --- Motor de Cálculo Detallado ---
def calculate_detailed_stress(params):
    """
    Implementa el modelo de cálculo detallado. Acepta Sy como parte de los params.
    """
    D_e = params['D']
    t = params['t']
    p_i = params['p']
    delta_T = params['delta_T']
    H = params['H']
    gamma_sat = params['gamma_sat']
    K0 = params['K0']
    W_traffic = params['W_traffic'] * params['If']
    PGV = params['PGV']
    C = params['C']
    alpha_seismic = params['alpha_seismic']
    Sy = params['Sy']  # Límite de fluencia dinámico

    R_e = D_e / 2.0
    R_i = R_e - t
    D_i = 2 * R_i
    R = (R_e + R_i) / 2.0
    A_s = np.pi * (D_e - t) * t
    I = np.pi * R ** 3 * t
    S = I / (t / 2) if t > 0 else 0

    sigma_h_p = (p_i * D_i) / (2 * t) if t > 0 else 0
    sigma_a_p = NU_ACERO * sigma_h_p
    sigma_a_T = E_ACERO * ALPHA_T_ACERO * delta_T
    sigma_a_w = E_ACERO * alpha_seismic * (PGV / C) if C > 0 else 0
    sigma_L_uniforme = sigma_a_p + sigma_a_T + sigma_a_w
    N_pi = p_i * R_i

    P_p = GAMMA_ACERO * A_s
    Hc = H - R_e
    P_v_suelo = gamma_sat * Hc
    p_v_trafico = (3 * W_traffic) / (2 * np.pi * Hc ** 2) if Hc > 0 else 0
    P_v_total = P_v_suelo + p_v_trafico

    C1_ext = -P_v_total * R ** 2 * (1 - K0) * (np.pi / 2)
    C2_ext = -P_v_total * R ** 2 * (1 - K0) * (np.pi / 2)
    N0_R_ext = (C1_ext - C2_ext) / (np.pi / 2)
    M0_ext = (C1_ext + N0_R_ext * np.pi) / np.pi
    N0_ext = N0_R_ext / R if R > 0 else 0

    C1_pp = P_p * R ** 2 * (4 - np.pi) / (2 * np.pi)
    C2_pp = P_p * R ** 2 * (8 + np.pi) / (2 * np.pi)
    N0_R_pp = (C1_pp - C2_pp) / (np.pi / 2)
    M0_pp = (C1_pp + N0_R_pp * np.pi) / np.pi
    N0_pp = N0_R_pp / R if R > 0 else 0

    theta = np.deg2rad(np.arange(0, 181, 15))

    M_ext = M0_ext * (1 - np.cos(theta)) - N0_ext * R * (1 - np.cos(theta)) \
            - P_v_total * R ** 2 / 4 * (1 - K0) * (1 - np.cos(2 * theta))
    N_ext = -M0_ext / R * np.sin(theta) - N0_ext * np.cos(theta) \
            - P_v_total * R / 4 * (1 + K0) * np.sin(theta) \
            - P_v_total * R / 4 * (1 - K0) * np.sin(2 * theta)

    M_pp = M0_pp * (1 - np.cos(theta)) - N0_pp * R * (1 - np.cos(theta)) \
           + P_p * R ** 2 / (2 * np.pi) * (theta * np.sin(theta) - 2 * (1 - np.cos(theta)))
    N_pp = -M0_pp / R * np.sin(theta) - N0_pp * np.cos(theta) \
           + P_p * R / (2 * np.pi) * (theta * np.cos(theta) + np.sin(theta))

    M_total = M_ext + M_pp
    N_total = N_ext + N_pp + N_pi

    sigma_h_flex = (N_total / t) + (M_total / S)
    sigma_h_flex_int = (N_total / t) - (M_total / S)
    sigma_h_total_ext = sigma_h_p + sigma_h_flex
    sigma_h_total_int = sigma_h_p + sigma_h_flex_int
    sigma_L_flex_ext = NU_ACERO * sigma_h_flex
    sigma_L_flex_int = NU_ACERO * sigma_h_flex_int
    sigma_L_total_ext = sigma_L_uniforme + sigma_L_flex_ext
    sigma_L_total_int = sigma_L_uniforme + sigma_L_flex_int

    vm_ext = np.sqrt(sigma_L_total_ext ** 2 - sigma_L_total_ext * sigma_h_total_ext + sigma_h_total_ext ** 2)
    vm_int = np.sqrt(sigma_L_total_int ** 2 - sigma_L_total_int * sigma_h_total_int + sigma_h_total_int ** 2)

    results = {
        "theta_deg": np.rad2deg(theta), "M_total": M_total, "N_total": N_total,
        "sigma_h_total_ext": sigma_h_total_ext, "sigma_L_total_ext": sigma_L_total_ext,
        "sigma_VM_ext": vm_ext, "sigma_VM_int": vm_int, "sigma_L_uniforme": sigma_L_uniforme,
        "sigma_h_p": sigma_h_p, "max_vm": np.max([np.max(vm_ext), np.max(vm_int)]), "Sy": Sy
    }
    return results


# --- Interfaz Gráfica (Tkinter) ---
class PipelineStressApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador Avanzado de Esfuerzos en Tuberías (Modelo Detallado)")
        self.last_results = None
        self.current_plot_type = 'polar'

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", padding=5, font=('Arial', 10))
        style.configure("TEntry", padding=5, font=('Arial', 10))
        style.configure("Readonly.TEntry", foreground='black', background='#e9ecef')
        style.configure("TButton", padding=5, font=('Arial', 10, 'bold'))
        style.configure("Reset.TButton", foreground='black', background='#ffc107')
        style.map("Reset.TButton", background=[('active', '#e0a800')])
        style.configure("Export.TButton", foreground='white', background='#17a2b8')
        style.map("Export.TButton", background=[('active', '#138496')])
        style.configure("TFrame", background='#f0f0f0')
        style.configure("Left.TFrame", background='#e8e8e8')
        style.configure("Right.TFrame", background='#ffffff')
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Arial', 9))
        style.map("Treeview", background=[('selected', '#007bff')], foreground=[('selected', 'white')])
        style.configure("Pass.TLabel", font=('Arial', 16, 'bold'), foreground='white', background='green',
                        anchor='center')
        style.configure("Fail.TLabel", font=('Arial', 16, 'bold'), foreground='white', background='red',
                        anchor='center')

        self.main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_frame = self.setup_left_panel(self.main_paned_window)
        self.main_paned_window.add(self.left_frame, weight=1)

        self.right_frame = self.setup_right_panel(self.main_paned_window)
        self.main_paned_window.add(self.right_frame, weight=3)

        self.reset_to_defaults()

    def setup_left_panel(self, parent):
        left_frame_outer = ttk.Frame(parent, style="Left.TFrame")
        canvas = tk.Canvas(left_frame_outer, borderwidth=0, background="#e8e8e8")
        scrollbar = ttk.Scrollbar(left_frame_outer, orient="vertical", command=canvas.yview)
        self.left_frame_inner = ttk.Frame(canvas, padding="10", style="Left.TFrame")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=self.left_frame_inner, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.left_frame_inner.bind("<Configure>", on_frame_configure)

        self.input_vars = {}
        self.create_input_fields(self.left_frame_inner)
        return left_frame_outer

    def create_input_fields(self, parent):
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(parent, text="Caso de Estudio:", font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3,
                                                                                    sticky=tk.W, pady=(0, 5))
        row += 1
        self.location_combobox = ttk.Combobox(parent, values=list(LOCATION_PARAMS.keys()), state="readonly", width=25)
        self.location_combobox.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=2, padx=5)
        self.location_combobox.bind("<<ComboboxSelected>>", self.on_location_selected)
        row += 1

        # --- Grupos de Parámetros ---
        ttk.Label(parent, text="--- Tubería y Material ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                columnspan=3,
                                                                                                sticky=tk.W,
                                                                                                pady=(8, 2), padx=5);
        row += 1
        self.add_entry(parent, row, "D", "Diámetro Ext. D", "m");
        row += 1
        self.add_entry(parent, row, "t", "Espesor t", "m");
        row += 1
        self.add_combobox(parent, row, "steel_grade", "Grado de Acero", list(STEEL_GRADES.keys()),
                          self.on_steel_selected);
        row += 1
        self.add_readonly_entry(parent, row, "Sy_display", "Límite de Fluencia", "MPa");
        row += 1
        self.add_readonly_entry(parent, row, "E_display", "Módulo de Young", "GPa");
        row += 1

        ttk.Label(parent, text="--- Cargas de Operación ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                 columnspan=3,
                                                                                                 sticky=tk.W,
                                                                                                 pady=(8, 2), padx=5);
        row += 1
        self.add_entry(parent, row, "p", "Presión Interna p", "Pa");
        row += 1
        self.add_entry(parent, row, "delta_T", "Cambio de Temp. ΔT", "°C");
        row += 1

        ttk.Label(parent, text="--- Instalación y Geotecnia ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                     columnspan=3,
                                                                                                     sticky=tk.W,
                                                                                                     pady=(8, 2),
                                                                                                     padx=5);
        row += 1
        self.add_entry(parent, row, "H", "Profundidad (eje) H", "m");
        row += 1
        self.add_entry(parent, row, "gamma_sat", "Peso Espec. Sat. γ_sat", "N/m³");
        row += 1
        self.add_entry(parent, row, "K0", "Coef. Empuje K0", "");
        row += 1

        ttk.Label(parent, text="--- Cargas de Tráfico ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                               columnspan=3,
                                                                                               sticky=tk.W, pady=(8, 2),
                                                                                               padx=5);
        row += 1
        self.add_combobox(parent, row, "vehicle", "Vehículo de Diseño", list(VEHICLE_TYPES.keys()),
                          self.on_vehicle_selected);
        row += 1
        self.add_readonly_entry(parent, row, "W_traffic_display", "Carga por Rueda", "kN");
        row += 1
        self.add_entry(parent, row, "If", "Factor Impacto If", "");
        row += 1

        ttk.Label(parent, text="--- Parámetros Sísmicos ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                 columnspan=3,
                                                                                                 sticky=tk.W,
                                                                                                 pady=(8, 2), padx=5);
        row += 1
        self.add_entry(parent, row, "PGV", "Velocidad Pico Suelo PGV", "m/s");
        row += 1
        self.add_entry(parent, row, "C", "Velocidad Onda C", "m/s");
        row += 1
        self.add_entry(parent, row, "alpha_seismic", "Factor Sísmico α", "");
        row += 1

        # --- Botones ---
        button_frame = ttk.Frame(parent, style="Left.TFrame")
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        button_frame.columnconfigure(0, weight=1);
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="Calcular y Graficar", command=self.run_analysis).grid(row=0, column=0, padx=5,
                                                                                             sticky=tk.E)
        ttk.Button(button_frame, text="Restaurar", command=self.reset_to_defaults, style="Reset.TButton").grid(row=0,
                                                                                                               column=1,
                                                                                                               padx=5,
                                                                                                               sticky=tk.W)

    def add_entry(self, parent, row, key, label_text, unit):
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        var = tk.StringVar()
        self.input_vars[key] = var
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=1, sticky=tk.EW, pady=2, padx=5)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)

    def add_readonly_entry(self, parent, row, key, label_text, unit):
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        var = tk.StringVar()
        self.input_vars[key] = var
        entry = ttk.Entry(parent, textvariable=var, width=15, state="readonly", style="Readonly.TEntry")
        entry.grid(row=row, column=1, sticky=tk.EW, pady=2, padx=5)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)

    def add_combobox(self, parent, row, key, label_text, values, command):
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        var = tk.StringVar()
        setattr(self, f"{key}_var", var)
        combobox = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
        combobox.grid(row=row, column=1, columnspan=2, sticky=tk.EW, pady=2, padx=5)
        combobox.bind("<<ComboboxSelected>>", command)

    def setup_right_panel(self, parent):
        right_frame = ttk.Frame(parent, style="Right.TFrame", padding=10)
        right_frame.rowconfigure(0, weight=0);
        right_frame.rowconfigure(1, weight=0);
        right_frame.rowconfigure(2, weight=3);
        right_frame.rowconfigure(3, weight=2)
        right_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(right_frame, text="ESTADO: PENDIENTE", style="TLabel", font=('Arial', 16, 'bold'),
                                      anchor='center')
        self.status_label.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        plot_controls_frame = ttk.Frame(right_frame, style="Right.TFrame")
        plot_controls_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(plot_controls_frame, text="Tipo de Gráfico:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_controls_frame, text="Polar", command=lambda: self.set_plot_type('polar')).pack(side=tk.LEFT,
                                                                                                        padx=2)
        ttk.Button(plot_controls_frame, text="Dispersión", command=lambda: self.set_plot_type('scatter')).pack(
            side=tk.LEFT, padx=2)

        self.plot_container = ttk.Frame(right_frame)
        self.plot_container.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        self.fig_canvas = None

        table_frame = ttk.Frame(right_frame)
        table_frame.grid(row=3, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1);
        table_frame.rowconfigure(0, weight=1)

        self.results_tree = ttk.Treeview(table_frame, style="Treeview")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree['columns'] = ("theta", "M", "N", "s_h_ext", "s_l_ext", "vm_ext", "vm_int")
        self.results_tree.column("#0", width=0, stretch=tk.NO)
        col_defs = {"theta": ("θ (°)", 80), "M": ("M (kNm)", 100), "N": ("N (kN/m)", 100),
                    "s_h_ext": ("σ_h,ext (MPa)", 120), "s_l_ext": ("σ_L,ext (MPa)", 120),
                    "vm_ext": ("σ_VM,ext (MPa)", 120), "vm_int": ("σ_VM,int (MPa)", 120)}
        for col, (text, width) in col_defs.items():
            self.results_tree.column(col, anchor=tk.E, width=width)
            self.results_tree.heading(col, text=text, anchor=tk.CENTER)
        ttk.Button(table_frame, text="Exportar a Excel", command=self.export_to_excel, style="Export.TButton").grid(
            row=1, column=0, columnspan=2, pady=(10, 0))

        return right_frame

    def on_location_selected(self, event=None):
        selected_location = self.location_combobox.get()
        if selected_location in LOCATION_PARAMS:
            params = LOCATION_PARAMS[selected_location]
            for key, value in params.items():
                if key == "steel_grade":
                    self.steel_grade_var.set(value)
                elif key == "vehicle":
                    self.vehicle_var.set(value)
                elif key in self.input_vars:
                    self.input_vars[key].set(f"{value:.6g}")
            self.on_steel_selected()
            self.on_vehicle_selected()

    def on_steel_selected(self, event=None):
        grade = self.steel_grade_var.get()
        if grade in STEEL_GRADES:
            sy_value = STEEL_GRADES[grade]
            self.input_vars['Sy_display'].set(f"{sy_value / 1e6:.1f}")
            self.input_vars['E_display'].set(f"{E_ACERO / 1e9:.1f}")

    def on_vehicle_selected(self, event=None):
        vehicle = self.vehicle_var.get()
        if vehicle in VEHICLE_TYPES:
            w_traffic = VEHICLE_TYPES[vehicle]
            self.input_vars['W_traffic_display'].set(f"{w_traffic / 1000:.1f}")

    def reset_to_defaults(self):
        self.location_combobox.set(DEFAULT_LOCATION)
        self.on_location_selected()
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            self.fig_canvas = None
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.last_results = None
        self.status_label.config(text="ESTADO: PENDIENTE", style="TLabel")

    def run_analysis(self):
        try:
            params = {key: float(var.get()) for key, var in self.input_vars.items() if
                      key not in ['Sy_display', 'E_display', 'W_traffic_display']}
            params['Sy'] = STEEL_GRADES[self.steel_grade_var.get()]
            params['W_traffic'] = VEHICLE_TYPES[self.vehicle_var.get()]

            self.last_results = calculate_detailed_stress(params)
            self.update_results_table(self.last_results)
            self.update_plots()

        except ValueError as ve:
            messagebox.showerror("Error de Entrada",
                                 f"Valor inválido en los campos de entrada.\nAsegúrese de que todos los campos contengan números.\n\nDetalle: {ve}")
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error inesperado.\n{e}")
            import traceback
            traceback.print_exc()

    def update_results_table(self, results):
        for item in self.results_tree.get_children(): self.results_tree.delete(item)
        for i in range(len(results["theta_deg"])):
            values = (f"{results['theta_deg'][i]:.0f}", f"{results['M_total'][i] / 1000:.2f}",
                      f"{results['N_total'][i] / 1000:.2f}", f"{results['sigma_h_total_ext'][i] / 1e6:.2f}",
                      f"{results['sigma_L_total_ext'][i] / 1e6:.2f}", f"{results['sigma_VM_ext'][i] / 1e6:.2f}",
                      f"{results['sigma_VM_int'][i] / 1e6:.2f}")
            self.results_tree.insert("", "end", values=values)

    def set_plot_type(self, plot_type):
        self.current_plot_type = plot_type
        self.update_plots()

    def update_plots(self):
        if not self.last_results: return
        results = self.last_results
        Sy = results['Sy']
        fs = Sy / results['max_vm'] if results['max_vm'] > 0 else float('inf')

        if fs >= 1.0:
            self.status_label.config(text=f"CUMPLE (FS = {fs:.2f})", style="Pass.TLabel")
        else:
            self.status_label.config(text=f"NO CUMPLE (FALLA) (FS = {fs:.2f})", style="Fail.TLabel")

        if self.fig_canvas: self.fig_canvas.get_tk_widget().destroy()
        fig = plt.figure(figsize=(10, 6), tight_layout=True)
        gs = fig.add_gridspec(2, 2)

        if self.current_plot_type == 'polar':
            ax_main = fig.add_subplot(gs[:, 0], polar=True)
            theta_rad = np.deg2rad(results["theta_deg"])
            full_theta = np.concatenate([theta_rad, theta_rad[1:-1] + np.pi])
            full_vm_ext = np.concatenate([results["sigma_VM_ext"], np.flip(results["sigma_VM_ext"][1:-1])])
            ax_main.plot(full_theta, full_vm_ext / 1e6, marker='o', markersize=4, label='$\sigma_{VM,ext}$ (MPa)')
            smys_line_theta = np.linspace(0, 2 * np.pi, 100)
            smys_line_r = np.full_like(smys_line_theta, Sy / 1e6)
            ax_main.plot(smys_line_theta, smys_line_r, 'r--', label=f'Límite Elástico ({Sy / 1e6:.0f} MPa)')
            ax_main.set_title('Distribución de Esfuerzo (Polar)', pad=20)
            ax_main.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        else:
            ax_main = fig.add_subplot(gs[:, 0])
            ax_main.plot(results["theta_deg"], results["sigma_VM_ext"] / 1e6, 'o-', label='$\sigma_{VM,ext}$')
            ax_main.plot(results["theta_deg"], results["sigma_VM_int"] / 1e6, 's--', label='$\sigma_{VM,int}$')
            ax_main.axhline(y=Sy / 1e6, color='r', linestyle='--', label=f'Límite Elástico ({Sy / 1e6:.0f} MPa)')
            ax_main.set_xlabel('Ángulo θ (grados)');
            ax_main.set_ylabel('Esfuerzo de Von Mises (MPa)');
            ax_main.set_title('Distribución de Esfuerzo (Dispersión)')
            ax_main.legend();
            ax_main.grid(True)

        ax_bar1 = fig.add_subplot(gs[0, 1])
        distritos = [d for d in LOCATION_PARAMS if d != "Personalizado"]
        esfuerzos_maximos = []
        for dist in distritos:
            temp_params = LOCATION_PARAMS[dist].copy()
            temp_params['Sy'] = STEEL_GRADES[temp_params['steel_grade']]
            temp_params['W_traffic'] = VEHICLE_TYPES[temp_params['vehicle']]
            temp_res = calculate_detailed_stress(temp_params)
            esfuerzos_maximos.append(temp_res['max_vm'] / 1e6)

        bars = ax_bar1.bar(distritos, esfuerzos_maximos, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        current_dist = self.location_combobox.get()
        if current_dist in distritos:
            bars[distritos.index(current_dist)].set_edgecolor('red');
            bars[distritos.index(current_dist)].set_linewidth(2)
        ax_bar1.axhline(y=Sy / 1e6, color='r', linestyle='--', label='Límite Elástico Actual')
        ax_bar1.set_ylabel('$\sigma_{VM,max}$ (MPa)');
        ax_bar1.set_title('Comparación de Esfuerzos Máximos')
        for bar in bars: ax_bar1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{bar.get_height():.1f}',
                                      va='bottom', ha='center')

        ax_bar2 = fig.add_subplot(gs[1, 1])
        factores_seguridad = [
            STEEL_GRADES[LOCATION_PARAMS[dist]['steel_grade']] / (em * 1e6) if em > 0 else float('inf') for dist, em in
            zip(distritos, esfuerzos_maximos)]
        bars_fs = ax_bar2.bar(distritos, factores_seguridad, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        if current_dist in distritos:
            bars_fs[distritos.index(current_dist)].set_edgecolor('red');
            bars_fs[distritos.index(current_dist)].set_linewidth(2)
        ax_bar2.axhline(y=1.0, color='r', linestyle='--', label='Límite de Falla (FS=1.0)')
        ax_bar2.set_ylabel('Factor de Seguridad (FS)');
        ax_bar2.set_title('Comparación de Factores de Seguridad')
        ax_bar2.set_ylim(bottom=0)
        for bar in bars_fs: ax_bar2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                                         f'{bar.get_height():.2f}', va='bottom', ha='center')

        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def export_to_excel(self):
        if not self.results_tree.get_children():
            messagebox.showwarning("Exportar",
                                   "No hay datos en la tabla para exportar. Por favor, realice un cálculo primero.")
            return

        try:
            filepath = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                    filetypes=[("Archivos de Excel", "*.xlsx"),
                                                               ("Todos los archivos", "*.*")],
                                                    title="Guardar resultados como...")
            if not filepath: return
            headers = [self.results_tree.heading(col)["text"] for col in self.results_tree["columns"]]
            data = [self.results_tree.item(item)["values"] for item in self.results_tree.get_children()]
            df = pd.DataFrame(data, columns=headers)
            df.to_excel(filepath, index=False, engine='openpyxl')
            messagebox.showinfo("Exportar", f"Los datos se han guardado exitosamente en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error de Exportación",
                                 f"No se pudo guardar el archivo de Excel.\nAsegúrese de tener instalada la librería 'openpyxl'.\n\nError: {e}")


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    try:
        import pandas
        import openpyxl
    except ImportError:
        messagebox.showwarning("Dependencias Faltantes",
                               "Para usar la función de exportar a Excel, necesita instalar las librerías 'pandas' y 'openpyxl'.\n\nPuede instalarlas ejecutando:\npip install pandas openpyxl")

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    initial_width = int(screen_width * 0.8)
    initial_height = int(screen_height * 0.8)
    root.geometry(f"{initial_width}x{initial_height}")
    root.minsize(1200, 750)

    app = PipelineStressApp(root)
    root.mainloop()
