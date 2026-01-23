import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# --- MÓDULO DE LÓGICA ESTRUCTURAL AVANZADA v4.1 (Cálculo Riguroso y Verificación Numérica) ---

def get_steel_properties():
    """Retorna un diccionario con propiedades de barras de acero comunes."""
    # Diametro (mm), Area (cm^2)
    return {
        '#3 (3/8")': (9.5, 0.71), '#4 (1/2")': (12.7, 1.29),
        '#5 (5/8")': (15.9, 1.99), '#6 (3/4")': (19.1, 2.84),
        '#8 (1")': (25.4, 5.10), '#9 (1 1/8")': (28.7, 6.45),
        '#10 (1 1/4")': (32.3, 8.19), '#11 (1 3/8")': (35.8, 10.06),
    }


def define_rebar_layout(b, h, rec, n_x, d_x_str, n_y, d_y_str):
    """Genera una lista de barras de acero con sus coordenadas y áreas. Lógica mejorada."""
    steel_props = get_steel_properties()
    rebars = []

    d_prima_x = rec + steel_props[d_x_str][0] / 20.0
    d_prima_y = rec + steel_props[d_y_str][0] / 20.0

    # Capas superior e inferior
    if n_x > 0:
        x_coords = np.linspace(-b / 2 + d_prima_x, b / 2 - d_prima_x, n_x) if n_x > 1 else [0]
        for x in x_coords:
            rebars.append({'x': x, 'y': h / 2 - d_prima_y, 'As': steel_props[d_x_str][1]})
            rebars.append({'x': x, 'y': -h / 2 - d_prima_y, 'As': steel_props[d_x_str][1]})

    # Capas laterales (sin contar las esquinas que ya están)
    if n_y > 2:
        y_coords = np.linspace(-h / 2 + d_prima_y, h / 2 - d_prima_y, n_y)
        for y in y_coords[1:-1]:  # Solo las intermedias
            rebars.append({'x': -b / 2 + d_prima_x, 'y': y, 'As': steel_props[d_y_str][1]})
            rebars.append({'x': b / 2 - d_prima_x, 'y': y, 'As': steel_props[d_y_str][1]})

    # Eliminar duplicados de coordenadas
    unique_rebars = []
    seen_coords = set()
    for rebar in rebars:
        coord = (round(rebar['x'], 2), round(rebar['y'], 2))
        if coord not in seen_coords:
            unique_rebars.append(rebar)
            seen_coords.add(coord)

    return unique_rebars


def calculate_load_combinations(loads):
    """Calcula combinaciones de carga basadas en ACI 318-19."""
    CM, CV, Sx, Sy = loads.get('CM', 0), loads.get('CV', 0), loads.get('Sx', 0), loads.get('Sy', 0)

    combinations = []
    combinations.append({'name': '1.4CM', 'Pu': 1.4 * CM, 'Mux': 0, 'Muy': 0})
    combinations.append({'name': '1.2CM+1.6CV', 'Pu': 1.2 * CM + 1.6 * CV, 'Mux': 0, 'Muy': 0})
    combinations.append({'name': '1.2CM+CV+Sx', 'Pu': 1.2 * CM + CV, 'Mux': Sx, 'Muy': 0.3 * Sy})
    combinations.append({'name': '1.2CM+CV-Sx', 'Pu': 1.2 * CM + CV, 'Mux': -Sx, 'Muy': -0.3 * Sy})
    combinations.append({'name': '1.2CM+CV+Sy', 'Pu': 1.2 * CM + CV, 'Mux': 0.3 * Sx, 'Muy': Sy})
    combinations.append({'name': '1.2CM+CV-Sy', 'Pu': 1.2 * CM + CV, 'Mux': -0.3 * Sx, 'Muy': -Sy})
    combinations.append({'name': '0.9CM+Sx', 'Pu': 0.9 * CM, 'Mux': Sx, 'Muy': 0.3 * Sy})
    combinations.append({'name': '0.9CM-Sx', 'Pu': 0.9 * CM, 'Mux': -Sx, 'Muy': -0.3 * Sy})
    combinations.append({'name': '0.9CM+Sy', 'Pu': 0.9 * CM, 'Mux': 0.3 * Sx, 'Muy': Sy})
    combinations.append({'name': '0.9CM-Sy', 'Pu': 0.9 * CM, 'Mux': -0.3 * Sx, 'Muy': -Sy})

    return [c for c in combinations if c['Pu'] >= 0]


def calculate_biaxial_surface_rigorous(params, rebars):
    """Calcula la superficie de interacción P-Mx-My usando discretización de fibras de concreto."""
    fc, fy, Es, b, h = params['fc'], params['fy'], params['Es'], params['b'], params['h']
    ecu = 0.003
    beta1 = max(0.65, 0.85 - 0.05 * (fc - 280) / 70) if fc > 280 else 0.85

    # Crear una malla de fibras de concreto
    n_fibers_y, n_fibers_x = 50, 50
    y_coords, dy = np.linspace(-h / 2, h / 2, n_fibers_y, retstep=True)
    x_coords, dx = np.linspace(-b / 2, b / 2, n_fibers_x, retstep=True)
    fiber_area = dx * dy

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    c_values = np.geomspace(0.1, h * 2, 30)

    pts = []

    for theta in angles:
        for c in c_values:
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            Cc, Mcx, Mcy = 0, 0, 0
            for y_f in y_coords:
                for x_f in x_coords:
                    d_fiber_comp_edge = (h / 2 - y_f) * cos_t + x_f * sin_t
                    es_fiber = ecu * (c - d_fiber_comp_edge) / c

                    if es_fiber > 0 and (c - d_fiber_comp_edge) < beta1 * c:
                        force_fiber = 0.85 * fc * fiber_area
                        Cc += force_fiber
                        Mcx += force_fiber * y_f
                        Mcy += force_fiber * x_f

            Fs, Msx, Msy = 0, 0, 0
            max_tensile_strain = -np.inf
            for rebar in rebars:
                d_rebar_comp_edge = (h / 2 - rebar['y']) * cos_t + rebar['x'] * sin_t
                es_i = ecu * (c - d_rebar_comp_edge) / c
                fs_i = np.clip((es_i - 0.00001) * Es, -fy, fy)  # Leve ajuste para convergencia

                fuerza_acero = fs_i * rebar['As']
                if es_i > 0:
                    fuerza_acero -= 0.85 * fc * rebar['As']

                Fs += fuerza_acero
                Msx += fuerza_acero * rebar['y']
                Msy += fuerza_acero * rebar['x']

                if es_i < 0:
                    max_tensile_strain = max(max_tensile_strain, -es_i)

            Pn = Cc + Fs
            Mnx = Mcx + Msx
            Mny = Mcy + Msy

            esy = fy / Es
            phi = 0.65
            if max_tensile_strain >= 0.005:
                phi = 0.90
            elif max_tensile_strain > esy:
                phi = 0.65 + 0.25 * (max_tensile_strain - esy) / (0.005 - esy)

            pts.append({'Pn': Pn / 1e3, 'Mnx': abs(Mnx / 1e5), 'Mny': abs(Mny / 1e5), 'Phi': phi})

    Ast = sum(r['As'] for r in rebars)
    Pn_max = (0.85 * fc * (b * h - Ast) + fy * Ast) / 1e3
    pts.append({'Pn': Pn_max, 'Mnx': 0, 'Mny': 0, 'Phi': 0.65})

    return pts


def calculate_shear_design(params, rebars, load_combos):
    """Realiza el diseño por cortante de la columna."""
    fc, fy, b, h, Lu = params['fc'], params['fy'], params['b'], params['h'], params['Lu']
    rec, n_legs_x, n_legs_y = params['rec'], params['n_legs_x'], params['n_legs_y']
    stirrup_area = get_steel_properties()[params['d_stirrup']][1]
    phi_v = 0.75

    params_pr = params.copy()
    params_pr['fy'] = 1.25 * fy

    Ast_half = sum(r['As'] for r in rebars if r['y'] > 0)
    d = h - rec
    a_pr = (Ast_half * params_pr['fy']) / (0.85 * fc * b) if (0.85 * fc * b) > 0 else 0
    Mprx = (Ast_half * params_pr['fy'] * (d - a_pr / 2)) / 1e5

    Ast_half_y = sum(r['As'] for r in rebars if r['x'] > 0)
    d_y = b - rec
    a_pr_y = (Ast_half_y * params_pr['fy']) / (0.85 * fc * h) if (0.85 * fc * h) > 0 else 0
    Mpry = (Ast_half_y * params_pr['fy'] * (d_y - a_pr_y / 2)) / 1e5

    Vud_x = 2 * Mprx / Lu if Lu > 0 else 0
    Vud_y = 2 * Mpry / Lu if Lu > 0 else 0

    Pu_max = max((c['Pu'] for c in load_combos), default=0) * 1000
    Ag = b * h
    Vc = 0.53 * np.sqrt(fc) * b * d * (1 + Pu_max / (140 * Ag)) if Ag > 0 else 0
    Vc_ton = Vc / 1000.0

    Vs_req_x = max(0, (Vud_x - phi_v * Vc_ton) / phi_v)
    Vs_req_y = max(0, (Vud_y - phi_v * Vc_ton) / phi_v)

    Av_x = n_legs_x * stirrup_area
    Av_y = n_legs_y * stirrup_area
    s_req_x = (Av_x * fy * d) / (Vs_req_x * 1000) if Vs_req_x > 0 else float('inf')
    s_req_y = (Av_y * fy * (b - rec)) / (Vs_req_y * 1000) if Vs_req_y > 0 else float('inf')
    s_req = min(s_req_x, s_req_y)

    Lo = max(h, Lu * 100 / 6, 50)
    s_max_conf = min(6 * get_steel_properties()[params['d_x']][0] / 10, 15)
    s_max_central = min(d / 2, 60)

    s_final_conf = min(s_req, s_max_conf)
    s_final_central = min(s_req, s_max_central)

    return {
        "Mprx": Mprx, "Mpry": Mpry, "Vud_x": Vud_x, "Vud_y": Vud_y,
        "Vc_ton": Vc_ton, "Vs_req_x": Vs_req_x, "Vs_req_y": Vs_req_y,
        "s_req_cm": s_req, "Lo_cm": Lo,
        "s_final_conf_cm": np.floor(s_final_conf),
        "s_final_central_cm": np.floor(s_final_central),
    }


# --- CLASE PRINCIPAL DE LA APLICACIÓN GUI ---

class ColumnProApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Diseño Guiado de Columnas de Concreto v4.1 - Cálculo Riguroso")
        self.geometry("1600x950")

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook.Tab', font=('Helvetica', 10, 'bold'))

        main_panel = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_panel.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(main_panel, width=450)
        main_panel.add(self.controls_frame, weight=1)

        self.notebook = ttk.Notebook(main_panel)
        main_panel.add(self.notebook, weight=3)

        self.vars = {}
        self.load_combos = []
        self.biaxial_data = {}

        self.create_pages()
        self.create_controls(self.controls_frame)

    def create_pages(self):
        self.page_biaxial_3d = self.add_plot_page("Análisis Biaxial (P-Mx-My)")
        self.page_biaxial_2d = self.add_plot_page("Diagrama Interacción (Mx-My)")
        self.page_shear = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.page_shear, text="Diseño por Corte")
        self.shear_text = tk.Label(self.page_shear, text="-- Esperando cálculo --", font=('Courier', 12),
                                   justify=tk.LEFT, anchor='nw')
        self.shear_text.pack(fill='both', expand=True)
        self.page_results = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.page_results, text="Verificación Final")
        self.results_text = tk.Label(self.page_results, text="-- Esperando cálculo --", font=('Helvetica', 12),
                                     justify=tk.LEFT, anchor='nw')
        self.results_text.pack(fill='both', expand=True)

    def add_plot_page(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        fig, ax = plt.subplots(figsize=(10, 8), tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return {'frame': frame, 'fig': fig, 'ax': ax, 'canvas': canvas}

    def create_controls(self, parent):
        parent.columnconfigure(0, weight=1)

        frame_mat = ttk.LabelFrame(parent, text="1. Materiales y Geometría", padding=10)
        frame_mat.grid(row=0, column=0, sticky='ew', pady=5, padx=5)
        self.vars['fc'] = self.create_entry(frame_mat, "f'c (kg/cm²):", 280.0, 0)
        self.vars['fy'] = self.create_entry(frame_mat, "fy (kg/cm²):", 4200.0, 1)
        self.vars['Es'] = self.create_entry(frame_mat, "Es (kg/cm²):", 2000000.0, 2)
        self.vars['b'] = self.create_entry(frame_mat, "Base (b) (cm):", 50.0, 3)
        self.vars['h'] = self.create_entry(frame_mat, "Peralte (h) (cm):", 70.0, 4)
        self.vars['rec'] = self.create_entry(frame_mat, "Recubrimiento (cm):", 4.0, 5)
        self.vars['Lu'] = self.create_entry(frame_mat, "Longitud Libre (Lu) (m):", 3.0, 6)

        frame_loads = ttk.LabelFrame(parent, text="2. Cargas de Servicio (Axial=Ton, Momento=Ton-m)", padding=10)
        frame_loads.grid(row=1, column=0, sticky='ew', pady=5, padx=5)
        self.vars['CM'] = self.create_entry(frame_loads, "Carga Muerta (CM):", 100.0, 0)
        self.vars['CV'] = self.create_entry(frame_loads, "Carga Viva (CV):", 60.0, 1)
        self.vars['Sx'] = self.create_entry(frame_loads, "Sismo (Mux):", 80.0, 2)
        self.vars['Sy'] = self.create_entry(frame_loads, "Sismo (Muy):", 60.0, 3)

        frame_combos = ttk.LabelFrame(parent, text="3. Combinaciones de Carga (ACI 318-19)", padding=10)
        frame_combos.grid(row=2, column=0, sticky='nsew', pady=5, padx=5)
        parent.rowconfigure(2, weight=1)
        columns = ("name", "Pu", "Mux", "Muy")
        self.combo_tree = ttk.Treeview(frame_combos, columns=columns, show='headings', height=5)
        for col in columns: self.combo_tree.heading(col, text=col); self.combo_tree.column(col, width=80,
                                                                                           anchor='center')
        self.combo_tree.pack(fill=tk.BOTH, expand=True)

        frame_steel = ttk.LabelFrame(parent, text="4. Acero Longitudinal", padding=10)
        frame_steel.grid(row=3, column=0, sticky='ew', pady=5, padx=5)
        steel_opts = list(get_steel_properties().keys())
        self.vars['n_x'] = self.create_entry(frame_steel, "N° Barras cara sup/inf:", 5, 0)
        self.vars['d_x'] = self.create_combobox(frame_steel, "Diám. cara sup/inf:", steel_opts, '#8 (1")', 1)
        self.vars['n_y'] = self.create_entry(frame_steel, "N° Barras cara lat:", 3, 2)
        self.vars['d_y'] = self.create_combobox(frame_steel, "Diám. cara lat:", steel_opts, '#8 (1")', 3)

        frame_stirrup = ttk.LabelFrame(parent, text="5. Acero Transversal (Estribos)", padding=10)
        frame_stirrup.grid(row=4, column=0, sticky='ew', pady=5, padx=5)
        self.vars['d_stirrup'] = self.create_combobox(frame_stirrup, "Diám. Estribo:", steel_opts, '#3 (3/8")', 0)
        self.vars['n_legs_x'] = self.create_entry(frame_stirrup, "Ramas dir. X:", 4, 1)
        self.vars['n_legs_y'] = self.create_entry(frame_stirrup, "Ramas dir. Y:", 3, 2)

        calc_button = ttk.Button(parent, text="DISEÑAR Y VERIFICAR COLUMNA", command=self.run_analysis)
        calc_button.grid(row=5, column=0, pady=20, padx=5, sticky='ew')

    def create_entry(self, parent, label, val, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        var = tk.DoubleVar(value=val)
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        return var

    def create_combobox(self, parent, label, values, default_value, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        var = tk.StringVar(value=default_value)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state='readonly', width=12)
        combo.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        return var

    def run_analysis(self):
        try:
            params = {key: var.get() for key, var in self.vars.items()}
            for k in ['n_x', 'n_y', 'n_legs_x', 'n_legs_y']: params[k] = int(params[k])
        except (ValueError, tk.TclError):
            messagebox.showerror("Error", "Verifique que todos los datos sean números válidos.")
            return

        self.load_combos = calculate_load_combinations(params)
        self.combo_tree.delete(*self.combo_tree.get_children())
        for combo in self.load_combos:
            self.combo_tree.insert('', 'end', values=(combo['name'], f"{combo['Pu']:.2f}", f"{combo['Mux']:.2f}",
                                                      f"{combo['Muy']:.2f}"))

        rebars = define_rebar_layout(params['b'], params['h'], params['rec'],
                                     params['n_x'], params['d_x'], params['n_y'], params['d_y'])

        self.biaxial_data = calculate_biaxial_surface_rigorous(params, rebars)
        self.shear_data = calculate_shear_design(params, rebars, self.load_combos)

        self.update_biaxial_3d_plot()
        self.update_biaxial_2d_plot()
        self.update_shear_page()
        self.update_results_page(rebars)

    def update_biaxial_3d_plot(self):
        page = self.page_biaxial_3d
        page['ax'].remove()
        page['ax'] = page['fig'].add_subplot(111, projection='3d')

        data = self.biaxial_data
        phiPn = np.array([p['Pn'] for p in data]) * np.array([p['Phi'] for p in data])
        phiMnx = np.array([p['Mnx'] for p in data]) * np.array([p['Phi'] for p in data])
        phiMny = np.array([p['Mny'] for p in data]) * np.array([p['Phi'] for p in data])

        page['ax'].plot_trisurf(phiMnx, phiMny, phiPn, cmap='viridis', alpha=0.6, edgecolor='none')

        for combo in self.load_combos:
            page['ax'].scatter(abs(combo['Mux']), abs(combo['Muy']), combo['Pu'], c='r', marker='o', s=50,
                               depthshade=False)

        page['ax'].set_xlabel('Momento Mx (Ton-m)');
        page['ax'].set_ylabel('Momento My (Ton-m)');
        page['ax'].set_zlabel('Carga Axial P (Ton)')
        page['ax'].set_title("Superficie de Interacción de Diseño (φP-φMx-φMy)")
        page['canvas'].draw()

    def update_biaxial_2d_plot(self):
        page = self.page_biaxial_2d
        page['ax'].clear()

        if not self.load_combos: return
        critical_combo = max(self.load_combos, key=lambda x: x['Pu'])
        Pu_design = critical_combo['Pu']

        data = self.biaxial_data
        phiPn = np.array([p['Pn'] for p in data]) * np.array([p['Phi'] for p in data])
        phiMnx = np.array([p['Mnx'] for p in data]) * np.array([p['Phi'] for p in data])
        phiMny = np.array([p['Mny'] for p in data]) * np.array([p['Phi'] for p in data])

        tolerance = 0.05 * np.max(phiPn)
        indices = np.where(np.abs(phiPn - Pu_design) < tolerance)[0]

        if len(indices) > 3:
            contour_mnx, contour_mny = phiMnx[indices], phiMny[indices]
            angles = np.arctan2(contour_mny, contour_mnx)
            sorted_indices = np.argsort(angles)
            closed_loop_mnx = np.append(contour_mnx[sorted_indices], contour_mnx[sorted_indices][0])
            closed_loop_mny = np.append(contour_mny[sorted_indices], contour_mny[sorted_indices][0])
            page['ax'].plot(closed_loop_mnx, closed_loop_mny, 'r-', lw=2,
                            label=f'Contorno Capacidad (Pu≈{Pu_design:.1f} Ton)')

        for combo in self.load_combos:
            page['ax'].plot(abs(combo['Mux']), abs(combo['Muy']), 'bo', markersize=8, alpha=0.7,
                            label=combo['name'] if combo == critical_combo else None)

        page['ax'].set_xlabel('Momento Mx (Ton-m)');
        page['ax'].set_ylabel('Momento My (Ton-m)')
        page['ax'].set_title(f"Diagrama Interacción Mx-My para Carga Axial Crítica")
        page['ax'].grid(True);
        page['ax'].legend(fontsize='small');
        page['ax'].axis('equal')
        page['canvas'].draw()

    def update_shear_page(self):
        d = self.shear_data
        report = f"""
{'--- DISEÑO POR CORTE (DIRECCIÓN X - PERALTE h) ---':^80}
1. Momento Probable (Mprx): {d['Mprx']:.2f} Ton-m
   (Capacidad a flexión con fy*1.25, phi=1.0)

2. Cortante de Diseño por Capacidad (Vud,x):
   Vud,x = 2 * Mprx / Lu = 2 * {d['Mprx']:.2f} / {self.vars['Lu'].get()} = {d['Vud_x']:.2f} Ton

3. Resistencia al Corte del Concreto (Vc,x):
   (Basado en ACI 318-19 para SDE)
   phi*Vc = {0.75:.2f} * {d['Vc_ton']:.2f} = {d['Vc_ton'] * 0.75:.2f} Ton

4. Cortante que debe tomar el Acero (Vs,x):
   Vs_req = (Vud - phi*Vc) / phi = ({d['Vud_x']:.2f} - {d['Vc_ton'] * 0.75:.2f}) / 0.75
   Vs_req,x = {d['Vs_req_x']:.2f} Ton

5. Espaciamiento Requerido vs. Máximo por Norma:
   s_calculado = {d['s_req_cm']:.2f} cm

   Zona de Confinamiento (Lo = {d['Lo_cm']:.1f} cm desde la cara):
     s_max = min(6*db_long, 15cm) = {min(6 * get_steel_properties()[self.vars['d_x'].get()][0] / 10, 15):.1f} cm
     ==> Usar espaciamiento de {d['s_final_conf_cm']:.0f} cm

   Zona Central:
     s_max = min(d/2, 60cm) = {min((self.vars['h'].get() - self.vars['rec'].get()) / 2, 60):.1f} cm
     ==> Usar espaciamiento de {d['s_final_central_cm']:.0f} cm

{'--- PROPUESTA FINAL DE ESTRIBOS ---':^80}
Estribos de {self.vars['d_stirrup'].get()} con {self.vars['n_legs_x'].get()} ramas en dir. X y {self.vars['n_legs_y'].get()} en dir. Y:
  - @ {d['s_final_conf_cm']:.0f} cm en los primeros {d['Lo_cm']:.0f} cm (Zona de Confinamiento)
  - @ {d['s_final_central_cm']:.0f} cm en la zona central.
"""
        self.shear_text.config(text=report)

    def update_results_page(self, rebars):
        report = "--- REPORTE DE VERIFICACIÓN ---\n\n"
        b, h, Ag = self.vars['b'].get(), self.vars['h'].get(), self.vars['b'].get() * self.vars['h'].get()
        Ast = sum(r['As'] for r in rebars)
        rho = Ast / Ag
        report += f"Geometría: {b}x{h} cm | Área Acero: {Ast:.2f} cm² | Cuantía (ρ): {rho:.4f}"
        report += " (OK)\n\n" if 0.01 <= rho <= 0.06 else " (ADVERTENCIA: Cuantía fuera de límites ACI 1%-6%)\n\n"

        report += "1. VERIFICACIÓN A FLEXO-COMPRESIÓN BIAXIAL (Ratio Demanda/Capacidad):\n"
        data = self.biaxial_data
        phiPn = np.array([p['Pn'] for p in data]) * np.array([p['Phi'] for p in data])
        phiMnx = np.array([p['Mnx'] for p in data]) * np.array([p['Phi'] for p in data])
        phiMny = np.array([p['Mny'] for p in data]) * np.array([p['Phi'] for p in data])

        all_safe = True

        for combo in self.load_combos:
            Pu_d, Mux_d, Muy_d = combo['Pu'], abs(combo['Mux']), abs(combo['Muy'])
            tolerance = 0.05 * np.max(phiPn)
            indices = np.where(np.abs(phiPn - Pu_d) < tolerance)[0]

            dcr, status = float('inf'), "NO CUMPLE"

            if len(indices) > 3:
                contour_mnx, contour_mny = phiMnx[indices], phiMny[indices]
                demand_angle = np.arctan2(Muy_d, Mux_d)
                demand_mag = np.sqrt(Mux_d ** 2 + Muy_d ** 2)

                contour_angles = np.arctan2(contour_mny, contour_mnx)
                contour_mags = np.sqrt(contour_mnx ** 2 + contour_mny ** 2)

                sorted_pairs = sorted(zip(contour_angles, contour_mags))
                capacity_mag = np.interp(demand_angle, *zip(*sorted_pairs))

                if capacity_mag > 1e-6:
                    dcr = demand_mag / capacity_mag
                    if dcr <= 1.0:
                        status = "CUMPLE"
                    else:
                        all_safe = False
                elif demand_mag < 1e-6:  # Demanda y capacidad son cero
                    dcr, status = 0.0, "CUMPLE"
                else:
                    all_safe = False
            elif Pu_d < 1e-6:  # Carga axial cero
                dcr, status = 0.0, "CUMPLE"
            else:
                all_safe = False

            report += f"  - Combo '{combo['name']}': DCR = {dcr:.3f} ({status})\n"

        report += "\n2. DISEÑO POR CORTANTE:\n"
        s_conf, s_central = self.shear_data['s_final_conf_cm'], self.shear_data['s_final_central_cm']
        report += f"  - Estribos {self.vars['d_stirrup'].get()} @ {s_conf:.0f} cm en zona confinada ({self.shear_data['Lo_cm']:.0f} cm)\n"
        report += f"  - y @ {s_central:.0f} cm en zona central.\n"

        report += "\n" + "-" * 40 + "\n--- CONCLUSIÓN FINAL ---\n" + "-" * 40 + "\n"
        if all_safe:
            report += "La sección de la columna y el refuerzo propuesto SON ADECUADOS\npara las combinaciones de carga analizadas."
            self.results_text.config(foreground='#006400')  # DarkGreen
        else:
            report += "La sección de la columna o el refuerzo propuesto NO SON SUFICIENTES.\nRevise las combinaciones de carga que no cumplen (DCR > 1.0)."
            self.results_text.config(foreground='#B22222')  # Firebrick

        self.results_text.config(text=report)


if __name__ == "__main__":
    app = ColumnProApp()
    app.mainloop()

