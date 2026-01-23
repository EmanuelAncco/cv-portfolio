import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import itertools

# --- Constantes y Parámetros Globales ---
E_ACERO = 2.07e11
NU_ACERO = 0.3
ALPHA_T_ACERO = 1.2e-5
GAMMA_ACERO = 78500

STEEL_GRADES = {
    "API 5L X42": 289.6e6, "API 5L X52": 358.5e6, "API 5L X60": 413.7e6,
    "API 5L X65": 448.2e6, "API 5L X70": 482.6e6, "API 5L X80": 551.6e6,
}
DEFAULT_STEEL = "API 5L X65"

COMMON_DIAMETERS = {
    "10 pulgadas": 0.254, "20 pulgadas": 0.508, "30 pulgadas": 0.762, "Personalizado": 0.0
}
COMMON_THICKNESSES = {
    "6.35 mm (0.250\")": 0.00635, "7.92 mm (0.312\")": 0.00792,
    "9.53 mm (0.375\")": 0.00953, "11.13 mm (0.438\")": 0.01113,
    "12.70 mm (0.500\")": 0.01270, "15.88 mm (0.625\")": 0.01588, "Personalizado": 0.0
}

VEHICLE_TYPES = {
    "AASHTO HS20 (71.2 kN)": 71170, "AASHTO HL-93 (72 kN)": 72000,
    "Camión Diseño T3-S2 (83 kN)": 83000, "Camión Pesado (Eje 11t - 108 kN)": 108000,
    "Personalizado": 35500
}
DEFAULT_VEHICLE = "Personalizado"

DISTRICT_GEO_PARAMS = {
    "Surco (Suelo Tipo C)": {"PGV": 0.50, "C": 400, "gamma_sat": 20000, "K0": 0.5, "p": 50e5, "D": 0.508, "t": 0.00953,
                             "H": 1.93, "If": 1.5, "delta_T": -15, "alpha_seismic": 1.0, "steel_grade": "API 5L X65",
                             "vehicle": "Personalizado"},
    "Villa El Salvador (Suelo Tipo D)": {"PGV": 0.60, "C": 250, "gamma_sat": 20000, "K0": 0.5, "p": 10e5, "D": 0.254,
                                         "t": 0.00635, "H": 2.33, "If": 1.5, "delta_T": -15, "alpha_seismic": 1.0,
                                         "steel_grade": "API 5L X65", "vehicle": "Personalizado"},
    "S.J. Lurigancho (Suelo Tipo C)": {"PGV": 0.40, "C": 400, "gamma_sat": 20000, "K0": 0.5, "p": 10e5, "D": 0.254,
                                       "t": 0.00635, "H": 1.45, "If": 1.5, "delta_T": -15, "alpha_seismic": 1.0,
                                       "steel_grade": "API 5L X65", "vehicle": "Personalizado"},
    "Personalizado": {"PGV": 0.5, "C": 400, "gamma_sat": 20000, "K0": 0.5, "p": 50e5, "D": 0.508, "t": 0.00953,
                      "H": 1.5, "If": 1.5, "delta_T": -15, "alpha_seismic": 1.0, "steel_grade": "API 5L X65",
                      "vehicle": "Personalizado"}
}


# --- Motor de Cálculo ---
def calculate_detailed_stress(params):
    D_e = params['D'];
    t = params['t'];
    p_i = params['p'];
    delta_T = params['delta_T']
    H = params['H'];
    gamma_sat = params['gamma_sat'];
    K0 = params['K0']
    W_traffic = params['W_traffic'] * params['If'];
    PGV = params['PGV'];
    C = params['C']
    alpha_seismic = params['alpha_seismic'];
    Sy = params['Sy']
    R_e = D_e / 2.0;
    R_i = R_e - t;
    D_i = 2 * R_i;
    R = (R_e + R_i) / 2.0
    A_s = np.pi * (D_e - t) * t;
    I = np.pi * R ** 3 * t;
    S = I / (t / 2) if t > 0 else 0
    sigma_h_p = (p_i * D_i) / (2 * t) if t > 0 else 0
    sigma_a_p = NU_ACERO * sigma_h_p
    sigma_a_T = E_ACERO * ALPHA_T_ACERO * delta_T
    sigma_a_w = E_ACERO * alpha_seismic * (PGV / C) if C > 0 else 0
    sigma_L_uniforme = sigma_a_p + sigma_a_T + sigma_a_w
    N_pi = p_i * R_i;
    P_p = GAMMA_ACERO * A_s;
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
    M_ext = M0_ext * (1 - np.cos(theta)) - N0_ext * R * (1 - np.cos(theta)) - P_v_total * R ** 2 / 4 * (1 - K0) * (
                1 - np.cos(2 * theta))
    N_ext = -M0_ext / R * np.sin(theta) - N0_ext * np.cos(theta) - P_v_total * R / 4 * (1 + K0) * np.sin(
        theta) - P_v_total * R / 4 * (1 - K0) * np.sin(2 * theta)
    M_pp = M0_pp * (1 - np.cos(theta)) - N0_pp * R * (1 - np.cos(theta)) + P_p * R ** 2 / (2 * np.pi) * (
                theta * np.sin(theta) - 2 * (1 - np.cos(theta)))
    N_pp = -M0_pp / R * np.sin(theta) - N0_pp * np.cos(theta) + P_p * R / (2 * np.pi) * (
                theta * np.cos(theta) + np.sin(theta))
    M_total = M_ext + M_pp;
    N_total = N_ext + N_pp + N_pi
    sigma_h_flex = (N_total / t) + (M_total / S);
    sigma_h_flex_int = (N_total / t) - (M_total / S)
    sigma_h_total_ext = sigma_h_p + sigma_h_flex;
    sigma_h_total_int = sigma_h_p + sigma_h_flex_int
    sigma_L_flex_ext = NU_ACERO * sigma_h_flex;
    sigma_L_flex_int = NU_ACERO * sigma_h_flex_int
    sigma_L_total_ext = sigma_L_uniforme + sigma_L_flex_ext;
    sigma_L_total_int = sigma_L_uniforme + sigma_L_flex_int
    vm_ext = np.sqrt(sigma_L_total_ext ** 2 - sigma_L_total_ext * sigma_h_total_ext + sigma_h_total_ext ** 2)
    vm_int = np.sqrt(sigma_L_total_int ** 2 - sigma_L_total_int * sigma_h_total_int + sigma_h_total_int ** 2)
    results = {"theta_deg": np.rad2deg(theta), "M_total": M_total, "N_total": N_total,
               "sigma_h_total_ext": sigma_h_total_ext, "sigma_L_total_ext": sigma_L_total_ext, "sigma_VM_ext": vm_ext,
               "sigma_VM_int": vm_int, "sigma_L_uniforme": sigma_L_uniforme, "sigma_h_p": sigma_h_p,
               "max_vm": np.max([np.max(vm_ext), np.max(vm_int)]), "Sy": Sy}
    return results


# --- Interfaz Gráfica (Tkinter) ---
class PipelineStressApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Herramienta de Diseño y Análisis de Tuberías")
        self.setup_styles()
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.setup_single_point_tab()
        self.setup_segment_design_tab()

    def setup_styles(self):
        style = ttk.Style();
        style.theme_use('clam')
        style.configure("TLabel", padding=2, font=('Arial', 10));
        style.configure("TEntry", padding=2, font=('Arial', 10))
        style.configure("Readonly.TEntry", foreground='black', background='#e9ecef')
        style.configure("TButton", padding=5, font=('Arial', 10, 'bold'))
        style.configure("Suggest.TButton", foreground='white', background='#007bff')
        style.map("Suggest.TButton", background=[('active', '#0056b3')])
        style.configure("Reset.TButton", foreground='black', background='#ffc107');
        style.map("Reset.TButton", background=[('active', '#e0a800')])
        style.configure("Export.TButton", foreground='white', background='#17a2b8');
        style.map("Export.TButton", background=[('active', '#138496')])
        style.configure("TFrame", background='#f0f0f0');
        style.configure("Left.TFrame", background='#e8e8e8');
        style.configure("Right.TFrame", background='#ffffff')
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'));
        style.configure("Treeview", rowheight=25, font=('Arial', 9))
        style.map("Treeview", background=[('selected', '#007bff')], foreground=[('selected', 'white')])
        style.configure("Pass.TLabel", font=('Arial', 14, 'bold'), foreground='white', background='green',
                        anchor='center')
        style.configure("Fail.TLabel", font=('Arial', 14, 'bold'), foreground='white', background='red',
                        anchor='center')
        style.configure("TNotebook.Tab", font=('Arial', 10, 'bold'), padding=[10, 5])

    def setup_single_point_tab(self):
        self.single_point_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.single_point_frame, text="Análisis de Punto Único")
        self.sp_input_vars = {};
        self.sp_last_results = None
        sp_paned_window = ttk.PanedWindow(self.single_point_frame, orient=tk.HORIZONTAL)
        sp_paned_window.pack(fill=tk.BOTH, expand=True)
        sp_left_panel = self.create_sp_input_panel(sp_paned_window);
        sp_paned_window.add(sp_left_panel, weight=1)
        sp_right_panel = self.create_sp_results_panel(sp_paned_window);
        sp_paned_window.add(sp_right_panel, weight=2)

    def create_sp_input_panel(self, parent):
        frame = ttk.Frame(parent, style="Left.TFrame", padding=10)
        frame.columnconfigure(1, weight=1);
        row = 0

        def add_row(key, label, unit):
            nonlocal row
            if key == "district_geo":
                self.add_combobox(frame, row, "sp_", "district_geo", label, list(DISTRICT_GEO_PARAMS.keys()),
                                  self.sp_on_district_selected); row += 1
            elif key == "steel_grade":
                self.add_combobox(frame, row, "sp_", "steel_grade", label, list(STEEL_GRADES.keys()),
                                  self.sp_on_steel_selected); row += 1
            elif key == "vehicle":
                self.add_combobox(frame, row, "sp_", "vehicle", label, list(VEHICLE_TYPES.keys()),
                                  self.sp_on_vehicle_selected); row += 1
            elif key == "D":
                self.add_editable_combobox(frame, row, "sp_", "D", "Diámetro Ext. D", list(COMMON_DIAMETERS.keys()),
                                           self.sp_on_diameter_selected, "m"); row += 1
            elif key == "t":
                self.add_editable_combobox(frame, row, "sp_", "t", "Espesor t", list(COMMON_THICKNESSES.keys()),
                                           self.sp_on_thickness_selected, "m"); row += 1
            elif "display" in key:
                self.add_readonly_entry(frame, row, "sp_", key, label, unit); row += 1
            else:
                self.add_entry(frame, row, "sp_", key, label, unit); row += 1

        ttk.Label(frame, text="Cargar Datos de Distrito:", font=('Arial', 11, 'bold')).grid(row=row, column=0,
                                                                                            columnspan=3, sticky="w",
                                                                                            pady=(0, 5));
        row += 1
        add_row("district_geo", "Zona Geotécnica", "")
        ttk.Label(frame, text="--- Tubería y Material ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                               columnspan=3, sticky="w",
                                                                                               pady=(8, 2));
        row += 1
        add_row("D", "Diámetro Ext. D", "m");
        add_row("t", "Espesor t", "m");
        add_row("steel_grade", "Grado de Acero", "")
        add_row("Sy_display", "Límite de Fluencia", "MPa");
        add_row("E_display", "Módulo de Young", "GPa")
        ttk.Label(frame, text="--- Cargas y Geotecnia ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                               columnspan=3, sticky="w",
                                                                                               pady=(8, 2));
        row += 1
        add_row("p", "Presión Interna p", "Pa");
        add_row("delta_T", "Cambio de Temp. ΔT", "°C");
        add_row("H", "Profundidad (eje) H", "m")
        add_row("gamma_sat", "Peso Espec. Sat. γ_sat", "N/m³");
        add_row("K0", "Coef. Empuje K0", "");
        add_row("vehicle", "Vehículo de Diseño", "")
        add_row("W_traffic_display", "Carga por Rueda", "kN");
        add_row("If", "Factor Impacto If", "")
        ttk.Label(frame, text="--- Parámetros Sísmicos ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                columnspan=3,
                                                                                                sticky="w",
                                                                                                pady=(8, 2));
        row += 1
        add_row("PGV", "PGV", "m/s");
        add_row("C", "Velocidad Onda C", "m/s");
        add_row("alpha_seismic", "Factor Sísmico α", "")
        ttk.Button(frame, text="Calcular Análisis", command=self.sp_run_analysis).grid(row=row, column=0, columnspan=3,
                                                                                       pady=20)
        return frame

    def create_sp_results_panel(self, parent):
        frame = ttk.Frame(parent, style="Right.TFrame", padding=10)
        frame.rowconfigure(1, weight=1);
        frame.columnconfigure(0, weight=1)
        self.sp_status_label = ttk.Label(frame, text="ESTADO: PENDIENTE", style="TLabel", font=('Arial', 16, 'bold'),
                                         anchor='center')
        self.sp_status_label.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.sp_plot_container = ttk.Frame(frame);
        self.sp_plot_container.grid(row=1, column=0, sticky="nsew")
        return frame

    def sp_on_district_selected(self, event=None):
        district = self.sp_district_geo_var.get()
        if district in DISTRICT_GEO_PARAMS:
            params = DISTRICT_GEO_PARAMS[district]
            for key, value in params.items():
                if f"sp_{key}" in self.sp_input_vars:
                    self.sp_input_vars[f"sp_{key}"].set(str(value))
                elif key == "steel_grade":
                    self.sp_steel_grade_var.set(value)
                elif key == "vehicle":
                    self.sp_vehicle_var.set(value)
            self.sp_on_steel_selected();
            self.sp_on_vehicle_selected()

    def sp_on_steel_selected(self, event=None):
        grade = self.sp_steel_grade_var.get()
        if grade in STEEL_GRADES: self.sp_input_vars['sp_Sy_display'].set(f"{STEEL_GRADES[grade] / 1e6:.1f}");
        self.sp_input_vars['sp_E_display'].set(f"{E_ACERO / 1e9:.1f}")

    def sp_on_vehicle_selected(self, event=None):
        vehicle = self.sp_vehicle_var.get()
        if vehicle in VEHICLE_TYPES: self.sp_input_vars['sp_W_traffic_display'].set(
            f"{VEHICLE_TYPES[vehicle] / 1000:.1f}")

    def sp_on_diameter_selected(self, event=None):
        d_nominal = self.sp_D_var.get()
        if d_nominal in COMMON_DIAMETERS: self.sp_input_vars['sp_D'].set(str(COMMON_DIAMETERS[d_nominal]))

    def sp_on_thickness_selected(self, event=None):
        t_nominal = self.sp_t_var.get()
        if t_nominal in COMMON_THICKNESSES: self.sp_input_vars['sp_t'].set(str(COMMON_THICKNESSES[t_nominal]))

    def sp_run_analysis(self):
        try:
            params = {}
            for key, var in self.sp_input_vars.items():
                clean_key = key.replace('sp_', '').replace('_display', '')
                if clean_key not in ['Sy', 'E', 'W_traffic', 'district_geo']: params[clean_key] = float(var.get())
            params['Sy'] = STEEL_GRADES[self.sp_steel_grade_var.get()];
            params['W_traffic'] = VEHICLE_TYPES[self.sp_vehicle_var.get()]
            self.sp_last_results = calculate_detailed_stress(params);
            self.sp_update_plot()
        except (ValueError, KeyError) as e:
            messagebox.showerror("Error de Entrada", f"Valor inválido o faltante en los campos de entrada.\n{e}")
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error inesperado.\n{e}")

    def sp_update_plot(self):
        if not self.sp_last_results: return
        results = self.sp_last_results;
        Sy = results['Sy'];
        fs = Sy / results['max_vm'] if results['max_vm'] > 0 else float('inf');
        ratio = 1 / fs if fs > 0 else float('inf')
        if fs >= 1.0:
            self.sp_status_label.config(text=f"CUMPLE (FS = {fs:.2f} | Ratio = {ratio:.2f})", style="Pass.TLabel")
        else:
            self.sp_status_label.config(text=f"NO CUMPLE (FALLA) (FS = {fs:.2f} | Ratio = {ratio:.2f})",
                                        style="Fail.TLabel")
        if hasattr(self, 'sp_fig_canvas') and self.sp_fig_canvas: self.sp_fig_canvas.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'});
        theta_rad = np.deg2rad(results["theta_deg"])
        full_theta = np.concatenate([theta_rad, theta_rad[1:-1] + np.pi]);
        full_vm_ext = np.concatenate([results["sigma_VM_ext"], np.flip(results["sigma_VM_ext"][1:-1])])
        ax.plot(full_theta, full_vm_ext / 1e6, marker='o', markersize=4, label='$\sigma_{VM,ext}$ (MPa)')
        smys_line_theta = np.linspace(0, 2 * np.pi, 100);
        smys_line_r = np.full_like(smys_line_theta, Sy / 1e6)
        ax.plot(smys_line_theta, smys_line_r, 'r--', label=f'Límite Elástico ({Sy / 1e6:.0f} MPa)');
        ax.set_title('Distribución de Esfuerzo', pad=20, fontsize=12);
        ax.legend()
        self.sp_fig_canvas = FigureCanvasTkAgg(fig, master=self.sp_plot_container);
        self.sp_fig_canvas.draw();
        self.sp_fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def setup_segment_design_tab(self):
        self.segment_design_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.segment_design_frame, text="Diseño de Tramos 3D")
        self.segments = [];
        self.segment_id_counter = 0;
        self.last_detailed_results = None;
        self.current_plot_type = 'polar'
        main_paned_window = ttk.PanedWindow(self.segment_design_frame, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)
        left_frame = self.setup_left_panel(main_paned_window);
        main_paned_window.add(left_frame, weight=1)
        right_frame = self.setup_right_panel(main_paned_window);
        main_paned_window.add(right_frame, weight=3)
        self.add_segment()

    def setup_left_panel(self, parent):
        left_frame_outer = ttk.Frame(parent, style="Left.TFrame")
        canvas = tk.Canvas(left_frame_outer, borderwidth=0, background="#e8e8e8")
        scrollbar = ttk.Scrollbar(left_frame_outer, orient="vertical", command=canvas.yview)
        self.left_frame_inner = ttk.Frame(canvas, padding="10", style="Left.TFrame")
        canvas.configure(yscrollcommand=scrollbar.set);
        scrollbar.pack(side="right", fill="y");
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=self.left_frame_inner, anchor="nw")
        self.left_frame_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.input_vars = {};
        self.create_segment_manager(self.left_frame_inner);
        self.create_input_fields(self.left_frame_inner)
        return left_frame_outer

    def create_segment_manager(self, parent):
        manager_frame = ttk.Frame(parent, style="Left.TFrame");
        manager_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        manager_frame.columnconfigure(0, weight=1);
        ttk.Label(manager_frame, text="Tramos de la Tubería", font=('Arial', 12, 'bold')).grid(row=0, column=0,
                                                                                               columnspan=2, sticky="w")
        cols = ("id", "len", "az", "slope");
        self.segment_tree = ttk.Treeview(manager_frame, columns=cols, show="headings", height=5)
        self.segment_tree.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.segment_tree.heading("id", text="Tramo");
        self.segment_tree.column("id", width=80)
        self.segment_tree.heading("len", text="Long. (m)");
        self.segment_tree.column("len", width=60, anchor="e")
        self.segment_tree.heading("az", text="Azimut (°)"), self.segment_tree.column("az", width=70, anchor="e")
        self.segment_tree.heading("slope", text="Pend. (%)"), self.segment_tree.column("slope", width=70, anchor="e")
        self.segment_tree.bind("<<TreeviewSelect>>", self.on_segment_select)
        btn_frame = ttk.Frame(manager_frame, style="Left.TFrame");
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Button(btn_frame, text="Añadir Tramo", command=self.add_segment).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Quitar Tramo", command=self.remove_segment).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Restaurar Diseño", command=self.reset_design, style="Reset.TButton").pack(
            side="left", padx=2)
        ttk.Button(btn_frame, text="Calcular Tramos", command=self.run_analysis).pack(side="left", padx=10)

    def create_input_fields(self, parent):
        props_frame = ttk.Frame(parent, style="Left.TFrame");
        props_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        props_frame.columnconfigure(1, weight=1)
        ttk.Label(props_frame, text="Propiedades del Tramo Seleccionado", font=('Arial', 11, 'bold')).grid(row=0,
                                                                                                           column=0,
                                                                                                           columnspan=3,
                                                                                                           sticky="w",
                                                                                                           pady=5)
        row = 1

        def add_row(key, label, unit):
            nonlocal row
            if key == "district_geo":
                self.add_combobox(props_frame, row, "", "district_geo", label, list(DISTRICT_GEO_PARAMS.keys()),
                                  self.on_district_geo_selected); row += 1
            elif key == "steel_grade":
                self.add_combobox(props_frame, row, "", "steel_grade", label, list(STEEL_GRADES.keys()),
                                  self.on_steel_selected); row += 1
            elif key == "vehicle":
                self.add_combobox(props_frame, row, "", "vehicle", label, list(VEHICLE_TYPES.keys()),
                                  self.on_vehicle_selected); row += 1
            elif key == "D":
                self.add_editable_combobox(props_frame, row, "", "D", "Diámetro Ext. D", list(COMMON_DIAMETERS.keys()),
                                           self.on_diameter_selected, "m"); row += 1
            elif key == "t":
                self.add_editable_combobox(props_frame, row, "", "t", "Espesor t", list(COMMON_THICKNESSES.keys()),
                                           self.on_thickness_selected, "m"); row += 1
            elif "display" in key:
                self.add_readonly_entry(props_frame, row, "", key, label, unit); row += 1
            else:
                self.add_entry(props_frame, row, "", key, label, unit); row += 1

        add_row("length", "Longitud", "m");
        add_row("azimuth", "Azimut", "°");
        add_row("slope", "Pendiente", "%")
        ttk.Label(props_frame, text="--- Tubería y Material ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                     columnspan=3,
                                                                                                     sticky="w",
                                                                                                     pady=(8, 2));
        row += 1
        add_row("D", "Diámetro Ext. D", "m");
        add_row("t", "Espesor t", "m");
        add_row("steel_grade", "Grado de Acero", "")
        add_row("Sy_display", "Límite de Fluencia", "MPa");
        add_row("E_display", "Módulo de Young", "GPa")
        ttk.Label(props_frame, text="--- Cargas y Geotecnia ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                     columnspan=3,
                                                                                                     sticky="w",
                                                                                                     pady=(8, 2));
        row += 1
        add_row("district_geo", "Zona Geotécnica", "")
        add_row("p", "Presión Interna p", "Pa");
        add_row("delta_T", "Cambio de Temp. ΔT", "°C");
        add_row("H", "Profundidad (eje) H", "m")
        add_row("gamma_sat", "Peso Espec. Sat. γ_sat", "N/m³");
        add_row("K0", "Coef. Empuje K0", "");
        add_row("vehicle", "Vehículo de Diseño", "")
        add_row("W_traffic_display", "Carga por Rueda", "kN");
        add_row("If", "Factor Impacto If", "")
        ttk.Label(props_frame, text="--- Parámetros Sísmicos ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                      columnspan=3,
                                                                                                      sticky="w",
                                                                                                      pady=(8, 2));
        row += 1
        add_row("PGV", "PGV", "m/s");
        add_row("C", "Velocidad Onda C", "m/s");
        add_row("alpha_seismic", "Factor Sísmico α", "")
        ttk.Button(props_frame, text="Actualizar Tramo", command=self.update_segment_data).grid(row=row, column=0,
                                                                                                columnspan=3, pady=20)

    def add_entry(self, parent, row, prefix, key, label_text, unit):
        var_dict = self.sp_input_vars if prefix == 'sp_' else self.input_vars
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky="w", pady=1, padx=5);
        var = tk.StringVar();
        var_dict[f"{prefix}{key}"] = var
        ttk.Entry(parent, textvariable=var, width=15).grid(row=row, column=1, sticky="ew", pady=1, padx=5);
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", pady=1, padx=5)

    def add_readonly_entry(self, parent, row, prefix, key, label_text, unit):
        var_dict = self.sp_input_vars if prefix == 'sp_' else self.input_vars
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky="w", pady=1, padx=5);
        var = tk.StringVar();
        var_dict[f"{prefix}{key}"] = var
        ttk.Entry(parent, textvariable=var, width=15, state="readonly", style="Readonly.TEntry").grid(row=row, column=1,
                                                                                                      sticky="ew",
                                                                                                      pady=1, padx=5);
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", pady=1, padx=5)

    def add_combobox(self, parent, row, prefix, key, label_text, values, command):
        var_name = f"{prefix}{key}_var"
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky="w", pady=1, padx=5);
        var = tk.StringVar();
        setattr(self, var_name, var)
        combobox = ttk.Combobox(parent, textvariable=var, values=values, state="readonly");
        combobox.grid(row=row, column=1, columnspan=2, sticky="ew", pady=1, padx=5)
        combobox.bind("<<ComboboxSelected>>", command)

    def add_editable_combobox(self, parent, row, prefix, key, label_text, values, command, unit):
        var_dict = self.sp_input_vars if prefix == 'sp_' else self.input_vars
        var_name = f"{prefix}{key}_var"
        ttk.Label(parent, text=label_text + ":").grid(row=row, column=0, sticky="w", pady=1, padx=5);
        var = tk.StringVar();
        setattr(self, var_name, var)
        combobox = ttk.Combobox(parent, textvariable=var, values=values, state="normal");
        combobox.grid(row=row, column=1, sticky="ew", pady=1, padx=5)
        combobox.bind("<<ComboboxSelected>>", command);
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w", pady=1, padx=5)
        var_dict[f"{prefix}{key}"] = var

    def setup_right_panel(self, parent):
        right_frame = ttk.Frame(parent, style="Right.TFrame", padding=10)
        right_frame.rowconfigure(0, weight=3);
        right_frame.rowconfigure(1, weight=2)
        right_frame.columnconfigure(0, weight=2);
        right_frame.columnconfigure(1, weight=1)
        plot_3d_frame = ttk.Frame(right_frame);
        plot_3d_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5), padx=(0, 5))
        self.plot_3d_container = ttk.Frame(plot_3d_frame);
        self.plot_3d_container.pack(fill=tk.BOTH, expand=True);
        self.fig_3d_canvas = None

        controls_3d_frame = ttk.Frame(plot_3d_frame)
        controls_3d_frame.pack(pady=5)
        ttk.Button(controls_3d_frame, text="Gráfico Resumen", command=self.show_summary_stress_plot).pack(side="left",
                                                                                                          padx=5)
        ttk.Button(controls_3d_frame, text="Zoom +", command=lambda: self.zoom_3d_view(0.9)).pack(side="left", padx=2)
        ttk.Button(controls_3d_frame, text="Zoom -", command=lambda: self.zoom_3d_view(1.1)).pack(side="left", padx=2)
        ttk.Button(controls_3d_frame, text="Restaurar Vista", command=self.update_3d_plot).pack(side="left", padx=5)

        summary_frame = ttk.Frame(right_frame);
        summary_frame.grid(row=0, column=1, sticky="nsew")
        summary_frame.rowconfigure(1, weight=1);
        ttk.Label(summary_frame, text="Resumen por Tramo", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky="w")
        self.summary_tree = ttk.Treeview(summary_frame, columns=("id", "max_vm", "fs", "status"), show="headings",
                                         height=5)
        self.summary_tree.grid(row=1, column=0, sticky="nsew")
        self.summary_tree.heading("id", text="Tramo");
        self.summary_tree.column("id", width=60)
        self.summary_tree.heading("max_vm", text="σ_max (MPa)");
        self.summary_tree.column("max_vm", width=80, anchor="e")
        self.summary_tree.heading("fs", text="FS");
        self.summary_tree.column("fs", width=50, anchor="e")
        self.summary_tree.heading("status", text="Estado");
        self.summary_tree.column("status", width=80, anchor="center")
        self.summary_tree.bind("<<TreeviewSelect>>", self.on_summary_select)
        self.suggest_button = ttk.Button(summary_frame, text="Sugerir Solución", style="Suggest.TButton",
                                         command=self.suggest_solution)

        detailed_frame = ttk.Frame(right_frame);
        detailed_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5, 0))
        detailed_frame.rowconfigure(1, weight=1);
        detailed_frame.columnconfigure(1, weight=1)
        self.status_label = ttk.Label(detailed_frame, text="ESTADO TRAMO: PENDIENTE", style="TLabel",
                                      font=('Arial', 16, 'bold'), anchor='center')
        self.status_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        plot_2d_frame = ttk.Frame(detailed_frame);
        plot_2d_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        plot_2d_frame.rowconfigure(0, weight=1);
        plot_2d_frame.columnconfigure(1, weight=1)

        plot_2d_controls_frame = ttk.Frame(plot_2d_frame, style="Right.TFrame")
        plot_2d_controls_frame.grid(row=0, column=0, sticky="ns", padx=(0, 5))
        ttk.Label(plot_2d_controls_frame, text="Tipo de Gráfico:", font=('Arial', 10, 'bold')).pack(pady=(0, 5))
        ttk.Button(plot_2d_controls_frame, text="Polar", command=lambda: self.set_plot_type('polar')).pack(fill="x",
                                                                                                           padx=2)
        ttk.Button(plot_2d_controls_frame, text="Dispersión", command=lambda: self.set_plot_type('scatter')).pack(
            fill="x", padx=2, pady=5)
        ttk.Button(plot_2d_controls_frame, text="Superponer Todos",
                   command=lambda: self.set_plot_type('superimposed')).pack(fill="x", padx=2)

        self.plot_2d_container = ttk.Frame(plot_2d_frame);
        self.plot_2d_container.grid(row=0, column=1, sticky="nsew")
        self.fig_2d_canvas = None

        table_frame = ttk.Frame(detailed_frame);
        table_frame.grid(row=1, column=1, sticky="nsew")
        table_frame.rowconfigure(0, weight=1);
        table_frame.columnconfigure(0, weight=1)
        self.results_tree = ttk.Treeview(table_frame, style="Treeview");
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns");
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree['columns'] = ("theta", "M", "N", "vm_ext", "vm_int");
        self.results_tree.column("#0", width=0, stretch=tk.NO)
        col_defs = {"theta": ("θ (°)", 60), "M": ("M (kNm)", 80), "N": ("N (kN/m)", 80), "vm_ext": ("σ_ext (MPa)", 90),
                    "vm_int": ("σ_int (MPa)", 90)}
        for col, (text, width) in col_defs.items(): self.results_tree.column(col, anchor="e",
                                                                             width=width); self.results_tree.heading(
            col, text=text, anchor="center")
        ttk.Button(table_frame, text="Exportar Detalle", command=self.export_to_excel, style="Export.TButton").grid(
            row=2, column=0, columnspan=2, pady=(10, 0))
        return right_frame

    def add_segment(self):
        self.segment_id_counter += 1;
        segment_name = f"Tramo {self.segment_id_counter}";
        start_point = np.array([0.0, 0.0, 0.0])
        if self.segments: start_point = self.segments[-1]['end_point']

        try:
            new_segment_data = {key: float(var.get()) for key, var in self.input_vars.items() if
                                key not in ['Sy_display', 'E_display', 'W_traffic_display']}
            new_segment_data['id'] = segment_name
            new_segment_data['start_point'] = start_point
            az_rad = np.deg2rad(new_segment_data['azimuth']);
            dx = new_segment_data['length'] * np.cos(az_rad)
            dy = new_segment_data['length'] * np.sin(az_rad);
            dz = new_segment_data['length'] * (new_segment_data['slope'] / 100.0)
            new_segment_data['end_point'] = start_point + np.array([dx, dy, dz])
            new_segment_data['steel_grade'] = self.steel_grade_var.get()
            new_segment_data['vehicle'] = self.vehicle_var.get()
            new_segment_data['district_geo'] = self.district_geo_var.get()
            self.segments.append(new_segment_data)
        except (ValueError, KeyError):
            new_segment = {"id": segment_name, "length": 6.0, "azimuth": 0.0, "slope": 0.0, "start_point": start_point,
                           "end_point": start_point + np.array([6.0, 0.0, 0.0]), "D": 0.508, "t": 0.00953,
                           "steel_grade": DEFAULT_STEEL, "p": 50e5, "delta_T": -15, "H": 1.5, "gamma_sat": 20000,
                           "K0": 0.5, "vehicle": DEFAULT_VEHICLE, "If": 1.5, "PGV": 0.5, "C": 400, "alpha_seismic": 1.0,
                           "district_geo": "Personalizado"}
            self.segments.append(new_segment)

        self.update_segment_tree();
        self.update_3d_plot()

    def remove_segment(self):
        selected_item = self.segment_tree.selection()
        if not selected_item: messagebox.showwarning("Advertencia",
                                                     "Por favor, seleccione un tramo para quitar."); return
        segment_id_to_remove = self.segment_tree.item(selected_item[0])['values'][0]
        self.segments = [s for s in self.segments if s['id'] != segment_id_to_remove]
        current_pos = np.array([0.0, 0.0, 0.0])
        for segment in self.segments:
            segment['start_point'] = current_pos;
            az_rad = np.deg2rad(segment['azimuth']);
            dx = segment['length'] * np.cos(az_rad);
            dy = segment['length'] * np.sin(az_rad);
            dz = segment['length'] * (segment['slope'] / 100.0)
            current_pos += np.array([dx, dy, dz]);
            segment['end_point'] = current_pos
        self.update_segment_tree();
        self.update_3d_plot()
        for item in self.summary_tree.get_children(): self.summary_tree.delete(item)
        self.status_label.config(text="ESTADO TRAMO: PENDIENTE", style="TLabel")

    def reset_design(self):
        self.segments.clear()
        self.segment_id_counter = 0
        for tree in [self.segment_tree, self.summary_tree, self.results_tree]:
            for item in tree.get_children(): tree.delete(item)
        if hasattr(self,
                   'fig_2d_canvas') and self.fig_2d_canvas: self.fig_2d_canvas.get_tk_widget().destroy(); self.fig_2d_canvas = None
        self.status_label.config(text="ESTADO TRAMO: PENDIENTE", style="TLabel")
        self.add_segment()

    def update_segment_tree(self):
        for item in self.segment_tree.get_children(): self.segment_tree.delete(item)
        for seg in self.segments:
            values = (seg['id'], f"{seg['length']:.1f}", f"{seg['azimuth']:.1f}", f"{seg['slope']:.1f}")
            self.segment_tree.insert("", "end", values=values, iid=seg['id'])
        if self.segments: last_id = self.segments[-1]['id']; self.segment_tree.selection_set(
            last_id); self.segment_tree.focus(last_id)

    def on_segment_select(self, event=None):
        selected_item = self.segment_tree.selection()
        if not selected_item: return
        segment_id = self.segment_tree.item(selected_item[0])['values'][0]
        segment_data = next((s for s in self.segments if s['id'] == segment_id), None)
        if segment_data:
            for key, value in segment_data.items():
                if key == "steel_grade":
                    self.steel_grade_var.set(value)
                elif key == "vehicle":
                    self.vehicle_var.set(value)
                elif key == "district_geo":
                    self.district_geo_var.set(value)
                elif key == "D":
                    self.D_var.set(str(value))
                elif key == "t":
                    self.t_var.set(str(value))
                elif key in self.input_vars:
                    self.input_vars[key].set(f"{value:.6g}")
            self.on_steel_selected();
            self.on_vehicle_selected()

    def update_segment_data(self):
        selected_item = self.segment_tree.selection()
        if not selected_item: messagebox.showwarning("Advertencia", "Seleccione un tramo para actualizar."); return
        segment_id = self.segment_tree.item(selected_item[0])['values'][0]
        segment_index = next((i for i, s in enumerate(self.segments) if s['id'] == segment_id), None)
        if segment_index is not None:
            try:
                for key, var in self.input_vars.items():
                    if key not in ['Sy_display', 'E_display', 'W_traffic_display']: self.segments[segment_index][
                        key] = float(var.get())
                self.segments[segment_index]['steel_grade'] = self.steel_grade_var.get()
                self.segments[segment_index]['vehicle'] = self.vehicle_var.get()
                self.segments[segment_index]['district_geo'] = self.district_geo_var.get()
                start_point = np.array([0.0, 0.0, 0.0]) if segment_index == 0 else self.segments[segment_index - 1][
                    'end_point']
                for i in range(segment_index, len(self.segments)):
                    self.segments[i]['start_point'] = start_point
                    az_rad = np.deg2rad(self.segments[i]['azimuth']);
                    dx = self.segments[i]['length'] * np.cos(az_rad);
                    dy = self.segments[i]['length'] * np.sin(az_rad);
                    dz = self.segments[i]['length'] * (self.segments[i]['slope'] / 100.0)
                    end_point = start_point + np.array([dx, dy, dz]);
                    self.segments[i]['end_point'] = end_point;
                    start_point = end_point
                self.update_segment_tree();
                self.update_3d_plot();
                messagebox.showinfo("Éxito", f"Se actualizaron las propiedades del {segment_id}.")
            except ValueError:
                messagebox.showerror("Error", "Valor inválido en los campos de entrada.")

    def on_steel_selected(self, event=None):
        grade = self.steel_grade_var.get()
        if grade in STEEL_GRADES: self.input_vars['Sy_display'].set(f"{STEEL_GRADES[grade] / 1e6:.1f}");
        self.input_vars['E_display'].set(f"{E_ACERO / 1e9:.1f}")

    def on_vehicle_selected(self, event=None):
        vehicle = self.vehicle_var.get()
        if vehicle in VEHICLE_TYPES: self.input_vars['W_traffic_display'].set(f"{VEHICLE_TYPES[vehicle] / 1000:.1f}")

    def on_diameter_selected(self, event=None):
        d_nominal = self.D_var.get()
        if d_nominal in COMMON_DIAMETERS: self.input_vars['D'].set(str(COMMON_DIAMETERS[d_nominal]))

    def on_thickness_selected(self, event=None):
        t_nominal = self.t_var.get()
        if t_nominal in COMMON_THICKNESSES: self.input_vars['t'].set(str(COMMON_THICKNESSES[t_nominal]))

    def on_district_geo_selected(self, event=None):
        district = self.district_geo_var.get()
        if district in DISTRICT_GEO_PARAMS:
            params = DISTRICT_GEO_PARAMS[district]
            for key, value in params.items():
                if key in self.input_vars:
                    self.input_vars[key].set(str(value))
                elif key == "steel_grade":
                    self.steel_grade_var.set(value)
                elif key == "vehicle":
                    self.vehicle_var.set(value)
                elif key == "D":
                    self.D_var.set(str(value))
                elif key == "t":
                    self.t_var.set(str(value))
            self.on_steel_selected();
            self.on_vehicle_selected()

    def run_analysis(self):
        if not self.segments: messagebox.showwarning("Advertencia", "No hay tramos para analizar."); return
        for item in self.summary_tree.get_children(): self.summary_tree.delete(item)
        for segment in self.segments:
            try:
                params = segment.copy();
                params['Sy'] = STEEL_GRADES[segment['steel_grade']];
                params['W_traffic'] = VEHICLE_TYPES[segment['vehicle']]
                results = calculate_detailed_stress(params);
                segment['results'] = results
                fs = results['Sy'] / results['max_vm'] if results['max_vm'] > 0 else float('inf')
                status = "CUMPLE" if fs >= 1.0 else "FALLA"
                values = (segment['id'], f"{results['max_vm'] / 1e6:.2f}", f"{fs:.2f}", status)
                self.summary_tree.insert("", "end", values=values, iid=segment['id'], tags=(status,))
            except Exception as e:
                messagebox.showerror("Error de Cálculo", f"Error al analizar {segment['id']}:\n{e}"); return
        self.summary_tree.tag_configure("CUMPLE", background="lightgreen");
        self.summary_tree.tag_configure("FALLA", background="#ffcccb")
        if self.summary_tree.get_children():
            first_item = self.summary_tree.get_children()[0];
            self.summary_tree.selection_set(first_item);
            self.summary_tree.focus(first_item)
        self.update_3d_plot()

    def on_summary_select(self, event=None):
        self.suggest_button.grid_remove()
        selected_item = self.summary_tree.selection()
        if not selected_item: return
        item_data = self.summary_tree.item(selected_item[0])
        segment_id = item_data['values'][0];
        status = item_data['values'][3]
        segment_data = next((s for s in self.segments if s['id'] == segment_id), None)
        if segment_data and 'results' in segment_data:
            self.last_detailed_results = segment_data['results'];
            self.update_detailed_results(self.last_detailed_results)
            if status == "FALLA": self.suggest_button.grid(row=2, column=0, pady=5, sticky="w")

    def update_detailed_results(self, results):
        if not results: return
        Sy = results['Sy'];
        fs = Sy / results['max_vm'] if results['max_vm'] > 0 else float('inf');
        ratio = 1 / fs if fs > 0 else float('inf')
        if fs >= 1.0:
            self.status_label.config(text=f"CUMPLE (FS = {fs:.2f} | Ratio = {ratio:.2f})", style="Pass.TLabel")
        else:
            self.status_label.config(text=f"NO CUMPLE (FALLA) (FS = {fs:.2f} | Ratio = {ratio:.2f})",
                                     style="Fail.TLabel")
        for item in self.results_tree.get_children(): self.results_tree.delete(item)
        for i in range(len(results["theta_deg"])):
            values = (f"{results['theta_deg'][i]:.0f}", f"{results['M_total'][i] / 1000:.2f}",
                      f"{results['N_total'][i] / 1000:.2f}", f"{results['sigma_VM_ext'][i] / 1e6:.2f}",
                      f"{results['sigma_VM_int'][i] / 1e6:.2f}")
            self.results_tree.insert("", "end", values=values)
        self.update_2d_plot()

    def set_plot_type(self, plot_type):
        self.current_plot_type = plot_type; self.update_2d_plot()

    def update_2d_plot(self):
        if hasattr(self, 'fig_2d_canvas') and self.fig_2d_canvas: self.fig_2d_canvas.get_tk_widget().destroy()
        fig = plt.figure(figsize=(5, 4), tight_layout=True)

        if self.current_plot_type == 'superimposed':
            ax = fig.add_subplot(111)
            ax.set_title('Superposición de Esfuerzos por Tramo', fontsize=10)
            ax.set_xlabel('Ángulo θ (°)', fontsize=9);
            ax.set_ylabel('Von Mises (MPa)', fontsize=9)
            for seg in self.segments:
                if 'results' in seg:
                    res = seg['results']
                    ax.plot(res["theta_deg"], res["sigma_VM_ext"] / 1e6, 'o-',
                            label=f"{seg['id']} ($\sigma_{{VM,ext}}$)")
            ax.legend(fontsize=8);
            ax.grid(True)
            ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        elif self.last_detailed_results:
            results = self.last_detailed_results;
            Sy = results['Sy']
            if self.current_plot_type == 'polar':
                ax = fig.add_subplot(111, polar=True);
                theta_rad = np.deg2rad(results["theta_deg"])
                full_theta = np.concatenate([theta_rad, theta_rad[1:-1] + np.pi]);
                full_vm_ext = np.concatenate([results["sigma_VM_ext"], np.flip(results["sigma_VM_ext"][1:-1])])
                ax.plot(full_theta, full_vm_ext / 1e6, marker='o', markersize=4, label='$\sigma_{VM,ext}$ (MPa)')
                smys_line_theta = np.linspace(0, 2 * np.pi, 100);
                smys_line_r = np.full_like(smys_line_theta, Sy / 1e6)
                ax.plot(smys_line_theta, smys_line_r, 'r--', label=f'Límite Elástico ({Sy / 1e6:.0f} MPa)');
                ax.set_title('Esfuerzo en Tramo', pad=20, fontsize=10);
                ax.legend(fontsize=8)
            else:  # scatter
                ax = fig.add_subplot(111);
                ax.plot(results["theta_deg"], results["sigma_VM_ext"] / 1e6, 'o-', label='$\sigma_{VM,ext}$')
                ax.plot(results["theta_deg"], results["sigma_VM_int"] / 1e6, 's--', label='$\sigma_{VM,int}$')
                ax.axhline(y=Sy / 1e6, color='r', linestyle='--', label=f'Límite Elástico ({Sy / 1e6:.0f} MPa)')
                ax.set_xlabel('Ángulo θ (°)', fontsize=9);
                ax.set_ylabel('Von Mises (MPa)', fontsize=9);
                ax.set_title('Esfuerzo en Tramo', fontsize=10);
                ax.legend(fontsize=8);
                ax.grid(True)
                ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        else:  # No results to show
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Calcule y seleccione un tramo para ver el detalle', ha='center', va='center')

        self.fig_2d_canvas = FigureCanvasTkAgg(fig, master=self.plot_2d_container);
        self.fig_2d_canvas.draw();
        self.fig_2d_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_3d_plot(self):
        if hasattr(self, 'fig_3d_canvas') and self.fig_3d_canvas: self.fig_3d_canvas.get_tk_widget().destroy()
        fig = plt.figure(figsize=(6, 6));
        self.ax_3d = fig.add_subplot(111, projection='3d')
        if not self.segments:
            self.ax_3d.set_xlabel("X (m)"); self.ax_3d.set_ylabel("Y (m)"); self.ax_3d.set_zlabel(
                "Z (m)"); self.ax_3d.set_title("Vista 3D de la Tubería")
        else:
            all_x, all_y, all_z = [], [], []
            for i, seg in enumerate(self.segments):
                p1 = seg['start_point'];
                p2 = seg['end_point'];
                color = 'gray'
                if 'results' in seg:
                    fs = seg['results']['Sy'] / seg['results']['max_vm'] if seg['results']['max_vm'] > 0 else float(
                        'inf')
                    if fs < 1.0:
                        color = plt.cm.Reds(0.5 + 0.5 * (i / len(self.segments)))
                    else:
                        color = plt.cm.Greens(0.5 + 0.5 * (i / len(self.segments)))
                self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], lw=5, marker='o', color=color)
                all_x.extend([p1[0], p2[0]]);
                all_y.extend([p1[1], p2[1]]);
                all_z.extend([p1[2], p2[2]])
            max_range = np.array([max(all_x) - min(all_x) if all_x else 0, max(all_y) - min(all_y) if all_y else 0,
                                  max(all_z) - min(all_z) if all_z else 0]).max() / 2.0
            if max_range == 0: max_range = 5
            mid_x = (max(all_x) + min(all_x)) * 0.5 if all_x else 0;
            mid_y = (max(all_y) + min(all_y)) * 0.5 if all_y else 0;
            mid_z = (max(all_z) + min(all_z)) * 0.5 if all_z else 0
            self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range);
            self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range);
            self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
            self.ax_3d.set_xlabel("X (m)");
            self.ax_3d.set_ylabel("Y (m)");
            self.ax_3d.set_zlabel("Z (m)");
            self.ax_3d.set_title("Vista 3D de la Tubería")
        self.fig_3d_canvas = FigureCanvasTkAgg(fig, master=self.plot_3d_container);
        self.fig_3d_canvas.draw();
        self.fig_3d_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def zoom_3d_view(self, factor):
        if not hasattr(self, 'ax_3d'): return
        xlim = self.ax_3d.get_xlim();
        ylim = self.ax_3d.get_ylim();
        zlim = self.ax_3d.get_zlim()
        self.ax_3d.set_xlim(np.mean(xlim) + (xlim - np.mean(xlim)) * factor)
        self.ax_3d.set_ylim(np.mean(ylim) + (ylim - np.mean(ylim)) * factor)
        self.ax_3d.set_zlim(np.mean(zlim) + (zlim - np.mean(zlim)) * factor)
        self.fig_3d_canvas.draw()

    def show_summary_stress_plot(self):
        if not self.segments or not any('results' in s for s in self.segments):
            messagebox.showinfo("Gráfico Resumen", "Por favor, calcule los tramos primero.")
            return

        win = tk.Toplevel(self.root);
        win.title("Resumen de Esfuerzos por Tramo")
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)

        ids = [s['id'] for s in self.segments if 'results' in s]
        max_vms = [s['results']['max_vm'] / 1e6 for s in self.segments if 'results' in s]
        sys = [s['results']['Sy'] / 1e6 for s in self.segments if 'results' in s]

        bars = ax.bar(ids, max_vms, label="$\sigma_{VM,max}$")
        ax.plot(ids, sys, color='red', linestyle='--', marker='_', markersize=10, label="Límite Elástico (Sy)")
        ax.set_ylabel("Esfuerzo (MPa)");
        ax.set_title("Esfuerzo Máximo vs. Límite Elástico por Tramo")
        ax.legend();
        ax.grid(axis='y', linestyle=':')
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw();
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_suggestion_dialog(self, title, message):
        win = tk.Toplevel(self.root);
        win.title(title)
        win.configure(bg='#f0f0f0')
        label = tk.Label(win, text=message, font=('Arial', 11), fg='#006400', bg='#f0f0f0', justify=tk.LEFT,
                         wraplength=350, padx=20, pady=20)
        label.pack()
        button = ttk.Button(win, text="Aceptar", command=win.destroy)
        button.pack(pady=10)
        win.transient(self.root);
        win.grab_set()
        self.root.wait_window(win)

    def suggest_solution(self):
        selected_item = self.summary_tree.selection()
        if not selected_item: return
        segment_id = self.summary_tree.item(selected_item[0])['values'][0]
        segment = next((s for s in self.segments if s['id'] == segment_id), None)
        if not segment: return

        current_params = segment.copy();
        current_thickness = current_params['t'];
        current_steel_grade_name = current_params['steel_grade']
        thicker_options = [t for t in COMMON_THICKNESSES.values() if t > current_thickness]
        better_steel_grades = {name: sy for name, sy in STEEL_GRADES.items() if
                               sy > STEEL_GRADES[current_steel_grade_name]}

        test_scenarios = [];
        if thicker_options: test_scenarios.extend([('t', t) for t in sorted(thicker_options)])
        if better_steel_grades: test_scenarios.extend([('steel_grade', name) for name in
                                                       sorted(better_steel_grades.keys(),
                                                              key=lambda k: better_steel_grades[k])])

        best_solution = None
        for param_key, param_value in test_scenarios:
            test_params = current_params.copy();
            test_params[param_key] = param_value
            if param_key == 'steel_grade':
                test_params['Sy'] = STEEL_GRADES[param_value]
            else:
                test_params['Sy'] = STEEL_GRADES[test_params['steel_grade']]
            test_params['W_traffic'] = VEHICLE_TYPES[test_params['vehicle']]
            results = calculate_detailed_stress(test_params)
            fs = results['Sy'] / results['max_vm'] if results['max_vm'] > 0 else float('inf')
            if fs >= 1.0: best_solution = (param_key, param_value); break

        if best_solution:
            key, value = best_solution
            if key == 't':
                thickness_name = next(name for name, val in COMMON_THICKNESSES.items() if val == value)
                self.show_suggestion_dialog("Sugerencia de Solución",
                                            f"Para que el {segment_id} cumpla, se sugiere aumentar el espesor a:\n\n{thickness_name}")
            else:
                self.show_suggestion_dialog("Sugerencia de Solución",
                                            f"Para que el {segment_id} cumpla, se sugiere cambiar el grado de acero a:\n\n{value}")
        else:
            messagebox.showwarning("Sugerencia de Solución",
                                   "No se encontró una solución simple aumentando el espesor o el grado del acero. Se requieren cambios más significativos.")

    def export_to_excel(self):
        if not self.last_detailed_results: messagebox.showwarning("Exportar",
                                                                  "No hay datos detallados para exportar. Calcule y seleccione un tramo."); return
        try:
            filepath = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                    filetypes=[("Archivos de Excel", "*.xlsx"),
                                                               ("Todos los archivos", "*.*")],
                                                    title="Guardar detalle del tramo como...")
            if not filepath: return
            results = self.last_detailed_results
            data = {"Angulo (°)": results['theta_deg'], "Momento M (kNm)": results['M_total'] / 1000,
                    "Axial N (kN/m)": results['N_total'] / 1000, "Esfuerzo VM Ext (MPa)": results['sigma_VM_ext'] / 1e6,
                    "Esfuerzo VM Int (MPa)": results['sigma_VM_int'] / 1e6}
            df = pd.DataFrame(data);
            df.to_excel(filepath, index=False, engine='openpyxl')
            messagebox.showinfo("Exportar", f"Los datos se han guardado exitosamente en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error de Exportación", f"No se pudo guardar el archivo de Excel.\nError: {e}")


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    try:
        import pandas; import openpyxl
    except ImportError:
        messagebox.showwarning("Dependencias Faltantes",
                               "Para usar la función de exportar a Excel, necesita 'pandas' y 'openpyxl'.\n\nPuede instalarlas con: pip install pandas openpyxl")
    root = tk.Tk()
    screen_width = root.winfo_screenwidth();
    screen_height = root.winfo_screenheight()
    initial_width = int(screen_width * 0.85);
    initial_height = int(screen_height * 0.85)
    root.geometry(f"{initial_width}x{initial_height}");
    root.minsize(1300, 800)
    app = PipelineStressApp(root);
    root.mainloop()
