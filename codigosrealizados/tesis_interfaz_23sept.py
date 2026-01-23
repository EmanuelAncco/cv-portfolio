import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

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

VEHICLE_TYPES = {
    "AASHTO HS20 (71.2 kN)": 71170, "Personalizado": 35500
}
DEFAULT_VEHICLE = "Personalizado"

DISTRICT_PARAMS = {
    "Surco (Suelo Tipo C)": {"PGV": 0.50, "C": 400, "gamma_sat": 20000, "K0": 0.5, "p": 50e5, "D": 0.508, "t": 0.00953,
                             "H": 1.93, "soil_type": "arena", "phi": 35, "c": 0, "steel_grade": "API 5L X65",
                             "vehicle": "Personalizado", "delta_T": -15, "If": 1.5, "alpha_seismic": 1.0},
    "Villa El Salvador (Suelo Tipo D)": {"PGV": 0.60, "C": 250, "gamma_sat": 20000, "K0": 0.5, "p": 10e5, "D": 0.254,
                                         "t": 0.00635, "H": 2.33, "soil_type": "arena", "phi": 30, "c": 0,
                                         "steel_grade": "API 5L X65", "vehicle": "Personalizado", "delta_T": -15,
                                         "If": 1.5, "alpha_seismic": 1.0},
    "S.J. Lurigancho (Suelo Tipo C)": {"PGV": 0.40, "C": 400, "gamma_sat": 19000, "K0": 0.6, "p": 10e5, "D": 0.254,
                                       "t": 0.00635, "H": 1.45, "soil_type": "arcilla", "phi": 0, "c": 50000,
                                       "steel_grade": "API 5L X65", "vehicle": "Personalizado", "delta_T": -15,
                                       "If": 1.5, "alpha_seismic": 1.0},
}


# --- MÓDULO DE CÁLCULO DE VIGA (BEF) POR DIFERENCIAS FINITAS ---
class BEFSolver:
    def __init__(self, pipe_params, soil_params, analysis_params):
        self.E = pipe_params['E'];
        self.D_e = pipe_params['D_e'];
        self.t = pipe_params['t'];
        self.L = pipe_params['L']
        self.p_ult_func = soil_params['p_ult_func'];
        self.y_ult = soil_params['y_ult']
        self.n = analysis_params['num_nodes'];
        self.h = self.L / (self.n - 1) if (self.n - 1) > 0 else 1
        self.tolerance = analysis_params.get('tolerance', 1e-6);
        self.max_iter = analysis_params.get('max_iter', 100)
        self.I = np.pi / 64 * (self.D_e ** 4 - (self.D_e - 2 * self.t) ** 4);
        self.EI = self.E * self.I
        self.x_coords = np.linspace(0, self.L, self.n)
        self.y_pipe = np.zeros(self.n);
        self.moment = np.zeros(self.n);
        self.sigma_flex = np.zeros(self.n)
        self.soil_reaction = np.zeros(self.n)

    def _get_soil_stiffness_and_force(self, y_pipe, u_ground):
        k_soil = np.zeros(self.n)
        p_ult_at_node = self.p_ult_func(self.x_coords)
        p_ult_values = np.full(self.n, p_ult_at_node) if isinstance(p_ult_at_node, (int, float)) else p_ult_at_node
        k_initial = p_ult_values / self.y_ult if self.y_ult > 0 else np.zeros(self.n)
        relative_disp = u_ground - y_pipe
        for i in range(self.n):
            if abs(relative_disp[i]) < self.y_ult:
                k_soil[i] = k_initial[i]
            else:
                if abs(relative_disp[i]) > 1e-9:
                    k_soil[i] = (np.sign(relative_disp[i]) * p_ult_values[i]) / relative_disp[i]
                else:
                    k_soil[i] = k_initial[i]
        f_soil = k_soil * u_ground
        return k_soil, f_soil

    def solve(self, u_ground):
        if self.L <= 0: return False
        y_pipe = np.zeros(self.n)
        for it in range(self.max_iter):
            k_soil, f_soil = self._get_soil_stiffness_and_force(y_pipe, u_ground)
            K = np.zeros((self.n, self.n));
            C = self.EI / self.h ** 4
            for i in range(2, self.n - 2):
                K[i, i - 2:i + 3] = [C, -4 * C, 6 * C, -4 * C, C];
                K[i, i] += k_soil[i]
            K[0, 0:3] = [2 * C, -4 * C, 2 * C];
            K[0, 0] += k_soil[0]
            K[1, 0:4] = [-4 * C, 7 * C, -4 * C, C];
            K[1, 1] += k_soil[1]
            K[self.n - 2, self.n - 4:self.n] = [C, -4 * C, 7 * C, -4 * C];
            K[self.n - 2, self.n - 2] += k_soil[self.n - 2]
            K[self.n - 1, self.n - 3:self.n] = [2 * C, -4 * C, 2 * C];
            K[self.n - 1, self.n - 1] += k_soil[self.n - 1]
            try:
                y_new = np.linalg.solve(K, f_soil)
            except np.linalg.LinAlgError:
                return False
            if np.linalg.norm(y_new - y_pipe) < self.tolerance:
                self.y_pipe = y_new;
                self._calculate_results(u_ground);
                return True
            y_pipe = y_new
        self.y_pipe = y_pipe;
        self._calculate_results(u_ground);
        return False

    def _calculate_results(self, u_ground):
        y = self.y_pipe;
        h = self.h;
        self.moment = np.zeros(self.n)
        for i in range(1, self.n - 1): self.moment[i] = -self.EI * (y[i - 1] - 2 * y[i] + y[i + 1]) / h ** 2
        self.moment[0], self.moment[-1] = self.moment[1], self.moment[-2]
        self.sigma_flex = self.moment * (self.D_e / 2) / self.I
        relative_disp = u_ground - self.y_pipe;
        p_ult_values = self.p_ult_func(self.x_coords)
        if isinstance(p_ult_values, (int, float)): p_ult_values = np.full(self.n, p_ult_values)
        self.soil_reaction = np.minimum(p_ult_values, (p_ult_values / self.y_ult) * np.abs(relative_disp)) * np.sign(
            relative_disp)

    def get_results(self):
        return {"x_coords": self.x_coords, "pipe_displacement": self.y_pipe, "moment": self.moment,
                "sigma_flex": self.sigma_flex, "soil_reaction": self.soil_reaction}


# --- MÓDULO DE CÁLCULO COMPLETO ---

def get_soil_py_parameters(params):
    H = params['H'];
    D = params['D'];
    soil_type = params.get('soil_type', 'arena')
    if soil_type == 'arena':
        phi = np.deg2rad(params.get('phi', 30))
        H_D_ratio = H / D if D > 0 else 0
        if H_D_ratio < 2:
            Nqh = 3.0
        elif H_D_ratio < 6:
            Nqh = 3.0 + 1.5 * (H_D_ratio - 2)
        else:
            Nqh = 9.0
        p_ult = params['gamma_sat'] * H * Nqh * D;
        y_ult = 0.03 * H if H > 0 else 0.01
    else:  # arcilla
        c = params.get('c', 20000);
        H_D_ratio = H / D if D > 0 else 0
        if H_D_ratio < 4:
            Nch = 3 + (H_D_ratio) * 1.5
        else:
            Nch = 9.0
        p_ult = Nch * c * D;
        y_ult = 0.05 * H if H > 0 else 0.02
    return {'p_ult_func': lambda x: p_ult, 'y_ult': y_ult}


def run_full_analysis(segment_params):
    params = segment_params.copy()
    params['Sy'] = STEEL_GRADES[params['steel_grade']]
    params['E'] = E_ACERO
    L = params['length']
    lambda_wave = params.get('lambda_wave', L * 2)

    pipe_params = {'E': params['E'], 'D_e': params['D'], 't': params['t'], 'L': L}
    soil_py_params = get_soil_py_parameters(params)
    analysis_params = {'num_nodes': max(21, int(L / 2)), 'max_iter': 200}

    solver_temp = BEFSolver(pipe_params, soil_py_params, analysis_params)
    x = solver_temp.x_coords
    Hs = params.get('Hs', 30);
    C = params.get('C', 400)
    Tg = 4 * Hs / C if C > 0 else 0
    Uh = (Tg ** 2 / (4 * np.pi ** 2)) * (0.5 * 9.81) * np.cos(np.pi * params['H'] / (2 * Hs))
    u_ground = Uh * np.sin(2 * np.pi * x / lambda_wave)

    solver = BEFSolver(pipe_params, soil_py_params, analysis_params)
    converged = solver.solve(u_ground)
    beam_results = solver.get_results()

    sigma_a_p = NU_ACERO * ((params['p'] * (params['D'] - 2 * params['t'])) / (2 * params['t'])) if params[
                                                                                                        't'] > 0 else 0
    sigma_a_T = E_ACERO * ALPHA_T_ACERO * params.get('delta_T', -15)
    sigma_L_uniforme = sigma_a_p + sigma_a_T
    sigma_L_total = sigma_L_uniforme + beam_results['sigma_flex']
    sigma_h_total = (params['p'] * (params['D'] - 2 * params['t'])) / (2 * params['t']) if params['t'] > 0 else 0

    sigma_VM = np.sqrt(sigma_L_total ** 2 - sigma_L_total * sigma_h_total + sigma_h_total ** 2)

    return {
        "beam_results": beam_results, "vm_results": sigma_VM, "max_vm": np.max(sigma_VM),
        "Sy": params['Sy'], "converged": converged, "u_ground": u_ground
    }


# --- CLASE PRINCIPAL DE LA APLICACIÓN GUI ---
class PipelineAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis Estructural de Gasoductos por Tramos v3.1 (Completo y Corregido)")
        style = ttk.Style()
        style.theme_use('clam')
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.design_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.design_frame, text="Diseño y Análisis de Tramos 3D")
        self.setup_design_tab()

    def setup_design_tab(self):
        self.segments = [];
        self.segment_id_counter = 0;
        self.last_detailed_results = None
        main_paned = ttk.PanedWindow(self.design_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        left_panel = self.setup_left_panel(main_paned)
        main_paned.add(left_panel, weight=1)
        right_panel = self.setup_right_panel(main_paned)
        main_paned.add(right_panel, weight=3)
        self.add_segment()

    def setup_left_panel(self, parent):
        frame = ttk.Frame(parent, padding=5)
        manager_frame = ttk.LabelFrame(frame, text="Gestor de Trazado")
        manager_frame.pack(fill=tk.X, pady=5)
        cols = ("id", "len", "az", "slope");
        self.segment_tree = ttk.Treeview(manager_frame, columns=cols, show="headings", height=6)
        self.segment_tree.heading("id", text="Tramo");
        self.segment_tree.column("id", width=60)
        self.segment_tree.heading("len", text="Long.(m)");
        self.segment_tree.column("len", width=60, anchor='e')
        self.segment_tree.heading("az", text="Azimut(°)");
        self.segment_tree.column("az", width=60, anchor='e')
        self.segment_tree.heading("slope", text="Pend.(%)");
        self.segment_tree.column("slope", width=60, anchor='e')
        self.segment_tree.pack(side=tk.TOP, fill=tk.X, expand=True, padx=5, pady=5)
        self.segment_tree.bind("<<TreeviewSelect>>", self.on_segment_select)
        btn_frame = ttk.Frame(manager_frame);
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Añadir", command=self.add_segment).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Quitar", command=self.remove_segment).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Calcular", command=self.run_full_segment_analysis).pack(side=tk.LEFT, expand=True,
                                                                                            fill=tk.X)
        props_frame = ttk.LabelFrame(frame, text="Propiedades del Tramo Seleccionado")
        props_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.create_input_fields(props_frame)
        return frame

    def create_input_fields(self, parent):
        self.input_vars = {};
        row = 0

        def add_row(key, label, unit, values=None, command=None):
            nonlocal row
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar();
            self.input_vars[key] = var
            if values:
                widget = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
                if command: widget.bind("<<ComboboxSelected>>", command)
            else:
                widget = ttk.Entry(parent, textvariable=var)
            widget.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            ttk.Label(parent, text=unit).grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
            row += 1

        add_row("district_geo", "Zona Geotécnica:", "", list(DISTRICT_PARAMS.keys()), self.on_district_geo_selected)
        add_row("length", "Longitud:", "m");
        add_row("azimuth", "Azimut:", "°");
        add_row("slope", "Pendiente:", "%")
        ttk.Label(parent, text="--- Tubería y Cargas ---").grid(row=row, columnspan=3, sticky='w', padx=5, pady=(8, 2));
        row += 1
        add_row("D", "Diámetro Ext. D:", "m");
        add_row("t", "Espesor t:", "m")
        add_row("steel_grade", "Grado de Acero:", "", list(STEEL_GRADES.keys()))
        add_row("p", "Presión Interna:", "Pa")
        ttk.Button(parent, text="Actualizar Tramo", command=self.update_segment_data).grid(row=row, column=0,
                                                                                           columnspan=3, pady=10)

    def setup_right_panel(self, parent):
        frame = ttk.Frame(parent, padding=5)
        top_frame = ttk.Frame(frame);
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        plot_3d_frame = ttk.LabelFrame(top_frame, text="Vista 3D del Trazado");
        plot_3d_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.plot_3d_container = ttk.Frame(plot_3d_frame);
        self.plot_3d_container.pack(fill=tk.BOTH, expand=True)
        summary_frame = ttk.LabelFrame(top_frame, text="Resumen de Análisis");
        summary_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        cols = ("id", "max_vm", "fs", "status");
        self.summary_tree = ttk.Treeview(summary_frame, columns=cols, show="headings", height=10)
        self.summary_tree.heading("id", text="Tramo");
        self.summary_tree.column("id", width=60)
        self.summary_tree.heading("max_vm", text="σ_max(MPa)");
        self.summary_tree.column("max_vm", width=80, anchor='e')
        self.summary_tree.heading("fs", text="FS");
        self.summary_tree.column("fs", width=50, anchor='e')
        self.summary_tree.heading("status", text="Estado");
        self.summary_tree.column("status", width=80, anchor='center')
        self.summary_tree.pack(fill=tk.BOTH, expand=True)
        self.summary_tree.bind("<<TreeviewSelect>>", self.on_summary_select)
        self.detailed_plot_frame = ttk.LabelFrame(frame, text="Análisis Detallado del Tramo");
        self.detailed_plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        return frame

    def add_segment(self):
        self.segment_id_counter += 1;
        segment_name = f"Tramo {self.segment_id_counter}"
        start_point = np.array([0., 0., 0.]) if not self.segments else self.segments[-1]['end_point']
        new_segment = {"id": segment_name, "length": 50.0, "azimuth": 0.0, "slope": 0.0,
                       **DISTRICT_PARAMS["Surco (Suelo Tipo C)"]}
        az_rad = np.deg2rad(new_segment['azimuth']);
        L = new_segment['length']
        new_segment['start_point'] = start_point
        new_segment['end_point'] = start_point + np.array(
            [L * np.sin(az_rad), L * np.cos(az_rad), L * (new_segment['slope'] / 100)])
        self.segments.append(new_segment)
        self.update_segment_tree()
        self.update_3d_plot()

    def remove_segment(self):
        selected_item = self.segment_tree.selection()
        if not selected_item: return
        segment_id = self.segment_tree.item(selected_item[0])['values'][0]
        self.segments = [s for s in self.segments if s['id'] != segment_id]
        self.recalculate_coordinates();
        self.update_segment_tree();
        self.update_3d_plot()

    def update_segment_tree(self):
        for item in self.segment_tree.get_children(): self.segment_tree.delete(item)
        for seg in self.segments:
            values = (seg['id'], f"{seg['length']:.1f}", f"{seg['azimuth']:.1f}", f"{seg['slope']:.1f}")
            self.segment_tree.insert("", "end", values=values, iid=seg['id'])
        if self.segments: self.segment_tree.selection_set(self.segments[-1]['id'])

    def on_segment_select(self, event):
        selected_item = self.segment_tree.selection()
        if not selected_item: return
        segment_id = self.segment_tree.item(selected_item[0])['values'][0]
        segment_data = next((s for s in self.segments if s['id'] == segment_id), None)
        if segment_data:
            for key, value in segment_data.items():
                if key in self.input_vars: self.input_vars[key].set(str(value))

    def on_district_geo_selected(self, event):
        district = self.input_vars["district_geo"].get()
        if district in DISTRICT_PARAMS:
            for key, value in DISTRICT_PARAMS[district].items():
                if key in self.input_vars: self.input_vars[key].set(str(value))

    def update_segment_data(self):
        selected_item = self.segment_tree.selection()
        if not selected_item: return
        segment_id = self.segment_tree.item(selected_item[0])['values'][0]
        idx = next((i for i, s in enumerate(self.segments) if s['id'] == segment_id), None)
        if idx is not None:
            for key, var in self.input_vars.items():
                try:
                    self.segments[idx][key] = type(self.segments[idx][key])(var.get())
                except (ValueError, KeyError):
                    pass  # Ignorar si la clave no existe o el tipo es incorrecto
            self.recalculate_coordinates();
            self.update_segment_tree();
            self.update_3d_plot()

    def recalculate_coordinates(self):
        current_pos = np.array([0., 0., 0.])
        for seg in self.segments:
            seg['start_point'] = current_pos;
            L = seg['length'];
            az_rad = np.deg2rad(seg['azimuth'])
            end_point = current_pos + np.array([L * np.sin(az_rad), L * np.cos(az_rad), L * (seg['slope'] / 100)])
            seg['end_point'] = end_point;
            current_pos = end_point

    def run_full_segment_analysis(self):
        for item in self.summary_tree.get_children(): self.summary_tree.delete(item)
        for segment in self.segments:
            try:
                results = run_full_analysis(segment)
                segment['analysis_results'] = results
                fs = results['Sy'] / results['max_vm'] if results['max_vm'] > 0 else float('inf')
                status = "CUMPLE" if fs >= 1.0 else "FALLA"
                values = (segment['id'], f"{results['max_vm'] / 1e6:.2f}", f"{fs:.2f}", status)
                self.summary_tree.insert("", "end", values=values, iid=segment['id'], tags=(status,))
            except Exception as e:
                messagebox.showerror("Error", f"Error en {segment['id']}: {e}");
                return
        self.summary_tree.tag_configure("CUMPLE", background="lightgreen")
        self.summary_tree.tag_configure("FALLA", background="#ffcccb")
        if self.summary_tree.get_children(): self.summary_tree.selection_set(self.summary_tree.get_children()[0])
        self.update_3d_plot(analysis_results=True)

    def on_summary_select(self, event):
        for widget in self.detailed_plot_frame.winfo_children(): widget.destroy()
        selected_item = self.summary_tree.selection()
        if not selected_item: return
        segment_id = self.summary_tree.item(selected_item[0])['values'][0]
        segment = next((s for s in self.segments if 'analysis_results' in s and s['id'] == segment_id), None)
        if segment:
            results = segment['analysis_results'];
            beam_results = results['beam_results']
            x = beam_results['x_coords'];
            Sy = results['Sy']

            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            axs[0].plot(x, results['u_ground'] * 1000, 'g--', label="Suelo");
            axs[0].plot(x, beam_results['pipe_displacement'] * 1000, 'b-', label="Tubería")
            axs[0].set_ylabel("Desplazamiento (mm)");
            axs[0].set_title(f"Análisis Detallado - {segment_id}");
            axs[0].legend();
            axs[0].grid(True)
            axs[1].plot(x, beam_results['moment'] / 1000, 'r-');
            axs[1].set_ylabel("Momento Flector (kNm)");
            axs[1].grid(True)
            axs[2].plot(x, results['vm_results'] / 1e6, 'k-', label="Esfuerzo VM");
            axs[2].axhline(y=Sy / 1e6, color='r', linestyle='--', label="Límite Elástico")
            axs[2].set_xlabel("Posición (m)");
            axs[2].set_ylabel("Esfuerzo (MPa)");
            axs[2].legend();
            axs[2].grid(True)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.detailed_plot_frame)
            canvas.draw();
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_3d_plot(self, analysis_results=False):
        if hasattr(self, 'fig_3d_canvas'): self.fig_3d_canvas.get_tk_widget().destroy()
        fig = plt.figure();
        self.ax_3d = fig.add_subplot(111, projection='3d')
        if self.segments:
            all_x, all_y, all_z = [], [], []
            for seg in self.segments:
                p1 = seg['start_point'];
                p2 = seg['end_point']
                all_x.extend([p1[0], p2[0]]);
                all_y.extend([p1[1], p2[1]]);
                all_z.extend([p1[2], p2[2]])
                color = 'gray'
                if analysis_results and 'analysis_results' in seg:
                    fs = seg['analysis_results']['Sy'] / seg['analysis_results']['max_vm']
                    color = 'green' if fs >= 1.0 else 'red'
                self.ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, lw=4)

            max_range = np.array(
                [max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]).max() / 2.0
            mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
            self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range);
            self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range);
            self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax_3d.set_xlabel("X (m)");
        self.ax_3d.set_ylabel("Y (m)");
        self.ax_3d.set_zlabel("Z (m)")
        self.fig_3d_canvas = FigureCanvasTkAgg(fig, master=self.plot_3d_container)
        self.fig_3d_canvas.draw();
        self.fig_3d_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x900")
    app = PipelineAnalysisApp(root)
    root.mainloop()

