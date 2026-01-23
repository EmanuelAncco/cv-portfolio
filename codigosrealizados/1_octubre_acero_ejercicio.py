# ==================================================================================================
# == SIMULADOR ESTRUCTURAL Y OPTIMIZADOR DE ACERO
# == Enfoque: Lógica de Construcción + Análisis de Capacidad + Verificación de Demanda + GUI
# == Versión: 7.0 (Integración de Módulo de Verificación de Demanda)
# == Autor: Dr. Consultor en Robótica para Construcción
# == Descripción:
# == Versión final que completa el ciclo de diseño. Se añade un módulo interactivo para
# == que el usuario ingrese las cargas de demanda (Pu, Mu) del análisis estructural.
# == La aplicación ahora no solo calcula la capacidad (Diagrama P-M), sino que también
# == grafica la demanda y emite un veredicto ("Diseño Conforme" / "No Conforme"),
# == consolidándose como una herramienta integral de verificación.
# ==================================================================================================

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import math
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import pulp

# --- Módulo de Visualización Científica ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path

# --- Configuración de Logging ---
LOG_FILE = f"simulador_estructural_integrado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])

# --- Constantes de Ingeniería ---
SPLICE_LENGTHS = {'1/2"': 1.10, '5/8"': 1.30, '3/4"': 1.60, '1"': 2.10}
BAR_DIAMETERS = {'3/8"': 0.95, '1/2"': 1.27, '5/8"': 1.59, '3/4"': 1.91, '1"': 2.54}


# =============================================================================
# MÓDULO DE ANÁLISIS ESTRUCTURAL (Diagrama de Interacción)
# =============================================================================
class DiagramaInteraccionColumna:
    """
    Calcula y grafica el diagrama de interacción Pn-Mn para una sección de
    columna de concreto armado rectangular, basado en los principios de
    compatibilidad de deformaciones y equilibrio del ACI 318.
    """

    def __init__(self, b, h, fc, fy, acero_config_layers, alpha_factor=0.80):
        self.b = b;
        self.h = h;
        self.fc = fc;
        self.fy = fy
        self.acero_config_layers = acero_config_layers
        self.alpha_factor = alpha_factor
        self.ecu = 0.003;
        self.Es = 2.0e6
        self.beta1 = self._calcular_beta1()
        self.posiciones_acero, self.areas_acero = self._configurar_acero()
        logging.info(f"Módulo de Análisis Estructural inicializado para columna {b}x{h}.")
        logging.info(f" -> Parámetros: f'c={fc}, fy={fy}, alpha={alpha_factor}, beta1={self.beta1:.3f}")

    def _calcular_beta1(self):
        if self.fc <= 280:
            return 0.85
        elif self.fc >= 550:
            return 0.65
        else:
            return max(0.65, 0.85 - 0.05 * (self.fc - 280) / 70)

    def _configurar_acero(self):
        posiciones, areas = [], []
        for capa in self.acero_config_layers:
            area_barra = np.pi * (capa['d_long'] ** 2) / 4
            areas.append(capa['n_barras'] * area_barra)
            posiciones.append(capa['distancia'])
        return np.array(posiciones), np.array(areas)

    def calcular_punto_diagrama(self, c):
        if c < 1e-5: c = 1e-5
        a = self.beta1 * c
        if a > self.h: a = self.h
        fuerza_c = 0.85 * self.fc * a * self.b
        deform_s = self.ecu * (c - self.posiciones_acero) / c
        esfuerzo_s = np.clip(deform_s * self.Es, -self.fy, self.fy)
        fuerza_s = self.areas_acero * esfuerzo_s
        Pn = (fuerza_c + np.sum(fuerza_s)) / 1000
        momento_c = fuerza_c * (self.h / 2 - a / 2)
        momento_s = fuerza_s * (self.h / 2 - self.posiciones_acero)
        Mn = (momento_c + np.sum(momento_s)) / 100000
        deform_tension_max = np.min(deform_s)
        ety = self.fy / self.Es
        if deform_tension_max <= -0.005:
            phi = 0.90
        elif deform_tension_max >= -ety:
            phi = 0.65
        else:
            phi = 0.65 + 0.25 * (abs(deform_tension_max) - ety) / (0.005 - ety)
        return Pn, Mn, phi

    def generar_diagrama_completo(self):
        c_valores = np.concatenate([np.linspace(1e-4, self.h, 200), np.linspace(self.h, self.h * 3, 100), [1e6]])
        puntos = [self.calcular_punto_diagrama(c) for c in c_valores]
        pn_nominal, mn_nominal, phi_valores = map(list, zip(*puntos))
        Ast = np.sum(self.areas_acero)
        Tn = -self.fy * Ast / 1000
        pn_nominal.insert(0, Tn);
        mn_nominal.insert(0, 0);
        phi_valores.insert(0, 0.90)
        pn_nominal, mn_nominal = np.array(pn_nominal), np.array(mn_nominal)
        pn_diseno, mn_diseno = pn_nominal * np.array(phi_valores), mn_nominal * np.array(phi_valores)
        Ag = self.b * self.h
        Po = 0.85 * self.fc * (Ag - Ast) + self.fy * Ast
        Pn_max = self.alpha_factor * Po / 1000
        phi_Pn_max = 0.65 * Pn_max
        pn_nominal[pn_nominal > Pn_max] = Pn_max
        pn_diseno[pn_diseno > phi_Pn_max] = phi_Pn_max
        logging.info(f" -> Pn,max (nominal) = {Pn_max:.2f} Ton; phi*Pn,max (diseño) = {phi_Pn_max:.2f} Ton")
        return list(pn_nominal), list(mn_nominal), list(pn_diseno), list(mn_diseno)


# --- Clases de Dominio (Ingeniería Civil) ---
class Column:
    def __init__(self, name, width, length, height, cover,
                 long_bar_type, long_bar_qty_x, long_bar_qty_y,
                 stirrup_bar_type, stirrup_spacing_confined,
                 stirrup_spacing_central, fc, fy, supplementary_stirrups_qty=0,
                 confined_length_factor=1 / 6):
        self.name = name;
        self.width = float(width);
        self.length = float(length)
        self.height = float(height);
        self.cover = float(cover)
        self.long_bar_type = long_bar_type;
        self.long_bar_qty_x = int(long_bar_qty_x)
        self.long_bar_qty_y = int(long_bar_qty_y)
        self.long_bar_qty = self.long_bar_qty_x * 2 + (
                    self.long_bar_qty_y - 2) * 2 if self.long_bar_qty_y > 1 else self.long_bar_qty_x * 2
        self.stirrup_bar_type = stirrup_bar_type;
        self.stirrup_spacing_confined = float(stirrup_spacing_confined)
        self.stirrup_spacing_central = float(stirrup_spacing_central)
        self.supplementary_stirrups_qty = int(supplementary_stirrups_qty)
        self.confined_length = self.height * confined_length_factor * 100
        self.fc = fc
        self.fy = fy

    def calculate_requirements(self):
        requirements = {}
        footing_anchor_length = 0.8
        splice_length = SPLICE_LENGTHS.get(self.long_bar_type, 1.20)
        has_splices = self.height > (9.0 - footing_anchor_length)

        if has_splices:
            main_piece_len = 9.0 - splice_length / 2
            splice_piece_len = self.height - main_piece_len + footing_anchor_length + splice_length / 2
            pieces = [round(main_piece_len, 2), round(splice_piece_len, 2)] * self.long_bar_qty
            description = f"{self.long_bar_qty * 2} piezas para empalme (long. aprox. {main_piece_len:.2f}m y {splice_piece_len:.2f}m)"
        else:
            long_bar_length = self.height + footing_anchor_length
            pieces = [round(long_bar_length, 2)] * self.long_bar_qty
            description = f"{self.long_bar_qty} barras de {long_bar_length:.2f}m c/u"

        requirements['longitudinal'] = {
            'bar_type': self.long_bar_type, 'pieces': pieces, 'description': description,
            'has_splices': has_splices, 'splice_length': splice_length
        }

        stirrup_width = self.width - 2 * self.cover
        stirrup_length = self.length - 2 * self.cover
        hook_length = 10 * 2
        total_stirrup_perimeter = (2 * stirrup_width + 2 * stirrup_length + hook_length) / 100
        qty_confined_zone = math.ceil(self.confined_length / self.stirrup_spacing_confined)
        qty_confined_total = qty_confined_zone * 2
        central_length = (self.height * 100) - (2 * self.confined_length)
        qty_central = math.ceil(central_length / self.stirrup_spacing_central) if central_length > 0 else 0
        total_stirrups = qty_confined_total + qty_central
        requirements['stirrups'] = {
            'bar_type': self.stirrup_bar_type, 'pieces': [round(total_stirrup_perimeter, 2)] * total_stirrups,
            'distribution': f"2 zonas confinadas ({qty_confined_total} est. @ {self.stirrup_spacing_confined}cm) y 1 zona central ({qty_central} est. @ {self.stirrup_spacing_central}cm)",
            'total_quantity': total_stirrups, 'qty_confined_zone': qty_confined_zone, 'qty_central': qty_central
        }

        if self.supplementary_stirrups_qty > 0:
            hook_sup_len = (self.width - 2 * self.cover) + 2 * (hook_length / 2)
            total_hooks = total_stirrups * self.supplementary_stirrups_qty
            requirements['supplementary_hooks'] = {
                'bar_type': self.stirrup_bar_type, 'pieces': [round(hook_sup_len / 100, 2)] * total_hooks,
                'description': f"{self.supplementary_stirrups_qty} gancho(s) por nivel de estribo.",
                'total_quantity': total_hooks
            }

        return requirements


# --- Aplicación Principal de la GUI ---
class AdvancedSteelSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador Estructural y Optimizador de Acero v7.0")
        self.geometry("1800x1000")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()
        self.piece_colors = ['#60a5fa', '#facc15', '#4ade80', '#f87171', '#fb923c', '#818cf8', '#a78bfa', '#e879f9',
                             '#22d3ee']
        self.color_map = {}
        self.last_cutting_plan = None
        self.interaction_chart_widget = None
        self.last_analysis_results = None
        self.init_project_data()
        self.create_widgets()

    def configure_styles(self):
        self.style.configure("TFrame", background="#f1f5f9")
        self.style.configure("TLabel", background="#f1f5f9", font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground="#0f172a")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, relief="flat")
        self.style.configure("Accent.TButton", foreground="white", background="#16a34a")
        self.style.map("Accent.TButton", background=[('active', '#15803d')])
        self.style.configure("Info.TButton", foreground="white", background="#0284c7")
        self.style.map("Info.TButton", background=[('active', '#0369a1')])
        self.style.configure("TEntry", padding=5, fieldbackground="#ffffff")
        self.style.configure("TLabelframe", background="#f1f5f9", bordercolor="#cbd5e1", relief="solid")
        self.style.configure("TLabelframe.Label", background="#f1f5f9", font=("Segoe UI", 11, "bold"),
                             foreground="#334155")
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", "#e2e8f0")], foreground=[("selected", "#0f172a")])
        self.style.configure("Success.TLabel", font=("Segoe UI", 10, "bold"), foreground="white", background="#16a34a",
                             padding=5)
        self.style.configure("Fail.TLabel", font=("Segoe UI", 10, "bold"), foreground="white", background="#dc2626",
                             padding=5)

    def init_project_data(self):
        logging.info("Inicializando datos de simulación del proyecto.")
        self.column_types = {
            "C-1 (25x40)": Column("C-1", 25, 40, 3.0, 4, '1/2"', 3, 2, '3/8"', 10, 20, fc=210, fy=4200),
            "C-2 (30x50)": Column("C-2", 30, 50, 3.0, 4, '5/8"', 4, 2, '3/8"', 10, 25, fc=210, fy=4200,
                                  supplementary_stirrups_qty=1),
            "C-3 (25x25)": Column("C-3", 25, 25, 3.0, 4, '1/2"', 2, 2, '3/8"', 15, 25, fc=210, fy=4200),
            "C-4 (60x30)": Column("C-4", 60, 30, 3.2, 5, '5/8"', 5, 2, '3/8"', 10, 20, fc=280, fy=4200,
                                  supplementary_stirrups_qty=2),
            "C-5 (40x40)": Column("C-5", 40, 40, 3.2, 5, '1/2"', 3, 3, '3/8"', 10, 15, fc=280, fy=4200,
                                  supplementary_stirrups_qty=1),
            "C-6 (60x60)": Column("C-6", 60, 60, 3.0, 4, '1"', 3, 3, '1/2"', 10, 15, fc=210, fy=4200,
                                  supplementary_stirrups_qty=1)
        }
        self.initial_stock = {'1/2"': 15, '5/8"': 10, '3/8"': 40, '1"': 20}

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10");
        main_frame.pack(fill="both", expand=True)
        input_panel = ttk.LabelFrame(main_frame, text="Datos del Proyecto", padding="15");
        input_panel.pack(side="left", fill="y", padx=(0, 10))
        # ... (código del panel de entrada sin cambios) ...
        ttk.Label(input_panel, text="Metrado de Columnas:", style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        self.column_inputs = {}
        for name in self.column_types.keys():
            frame = ttk.Frame(input_panel);
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=name, width=15).pack(side="left")
            entry = ttk.Entry(frame, width=8);
            entry.pack(side="left", padx=5);
            entry.insert(0, "0")
            self.column_inputs[name] = entry
        ttk.Separator(input_panel, orient="horizontal").pack(fill="x", pady=15)
        ttk.Label(input_panel, text="Stock Actual en Obra:", style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        self.stock_inputs = {}
        for bar_type, qty in self.initial_stock.items():
            frame = ttk.Frame(input_panel);
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"Barras {bar_type}:", width=15).pack(side="left")
            entry = ttk.Entry(frame, width=8);
            entry.pack(side="left", padx=5);
            entry.insert(0, str(qty))
            self.stock_inputs[bar_type] = entry
        run_button = ttk.Button(input_panel, text="Optimizar Proyecto", command=self.run_optimization,
                                style="Accent.TButton")
        run_button.pack(fill="x", pady=(20, 0), ipady=5)

        output_panel = ttk.Frame(main_frame);
        output_panel.pack(side="left", fill="both", expand=True)
        self.summary_frame = ttk.Frame(output_panel, padding=5);
        self.summary_frame.pack(fill="x")
        notebook = ttk.Notebook(output_panel);
        notebook.pack(fill="both", expand=True)

        vis_tab = ttk.Frame(notebook);
        notebook.add(vis_tab, text="Plan de Corte Visual")
        # ... (código de la pestaña sin cambios) ...
        self.canvas_cutting_plan = Canvas(vis_tab, bg="white", highlightthickness=0)
        v_scroll = ttk.Scrollbar(vis_tab, orient="vertical", command=self.canvas_cutting_plan.yview)
        self.canvas_cutting_plan.config(yscrollcommand=v_scroll.set);
        v_scroll.pack(side="right", fill="y")
        self.canvas_cutting_plan.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.cutting_plan_container = ttk.Frame(self.canvas_cutting_plan)
        self.canvas_cutting_plan.create_window((0, 0), window=self.cutting_plan_container, anchor="nw")
        self.cutting_plan_container.bind("<Configure>", lambda e: self.canvas_cutting_plan.configure(
            scrollregion=self.canvas_cutting_plan.bbox("all")))
        bom_tab = ttk.Frame(notebook, padding=5);
        notebook.add(bom_tab, text="Despiece por Elemento")
        self.bom_text = tk.Text(bom_tab, wrap="word", font=("Consolas", 10), relief="solid", borderwidth=1)
        self.bom_text.pack(fill="both", expand=True)

        # --- PESTAÑA DE DETALLES Y ANÁLISIS MEJORADA ---
        detail_tab = ttk.Frame(notebook, padding=10);
        notebook.add(detail_tab, text="Detalle y Análisis de Columnas")

        # Panel de Controles Superior
        top_control_frame = ttk.Frame(detail_tab);
        top_control_frame.pack(fill="x")

        # Frame para selección
        vis_control_frame = ttk.LabelFrame(top_control_frame, text="1. Visualización", padding=10)
        vis_control_frame.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(vis_control_frame, text="Seleccionar Columna:").pack(anchor="w")
        self.vis_selector = ttk.Combobox(vis_control_frame, values=list(self.column_types.keys()), state="readonly")
        self.vis_selector.pack(anchor="w")
        if self.column_types: self.vis_selector.current(0)
        self.vis_selector.bind("<<ComboboxSelected>>", self.update_detail_visuals)

        # Frame para análisis
        analysis_control_frame = ttk.LabelFrame(top_control_frame, text="2. Verificación de Cargas de Demanda",
                                                padding=10)
        analysis_control_frame.pack(side="left", fill="y")

        demand_grid = ttk.Frame(analysis_control_frame)
        demand_grid.pack()
        ttk.Label(demand_grid, text="Carga Axial, Pu (Ton):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.demand_pu_entry = ttk.Entry(demand_grid, width=10);
        self.demand_pu_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(demand_grid, text="Momento Flector, Mu (Ton-m):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.demand_mu_entry = ttk.Entry(demand_grid, width=10);
        self.demand_mu_entry.grid(row=1, column=1, padx=5, pady=2)

        verify_button = ttk.Button(analysis_control_frame, text="Verificar Punto de Demanda",
                                   command=self.verify_demand_point, style="Info.TButton")
        verify_button.pack(pady=(10, 0))

        self.verification_result_label = ttk.Label(analysis_control_frame, text="")
        self.verification_result_label.pack(pady=(5, 0))

        # Contenedor para los gráficos
        detail_canvases_frame = ttk.Frame(detail_tab)
        detail_canvases_frame.pack(fill="both", expand=True, pady=10)
        self.canvas_cross_section = Canvas(detail_canvases_frame, bg="white");
        self.canvas_cross_section.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.canvas_elevation = Canvas(detail_canvases_frame, bg="white");
        self.canvas_elevation.pack(side="left", fill="both", expand=True, padx=5)
        self.canvas_interaction_diagram = Canvas(detail_canvases_frame, bg="white", highlightthickness=1,
                                                 highlightbackground="#cccccc");
        self.canvas_interaction_diagram.pack(side="left", fill="both", expand=True, padx=(5, 0))

        log_tab = ttk.Frame(notebook, padding=5);
        notebook.add(log_tab, text="Consola de Simulación")
        self.log_text = tk.Text(log_tab, wrap="word", font=("Consolas", 10), relief="solid", borderwidth=1,
                                state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def run_interaction_diagram_analysis(self, demand_point=None):
        selected_col_name = self.vis_selector.get()
        if not selected_col_name: return None
        self.log_message(f"Iniciando análisis de capacidad para: {selected_col_name}")
        column = self.column_types[selected_col_name]
        try:
            d_estribo = BAR_DIAMETERS[column.stirrup_bar_type]
            d_long = BAR_DIAMETERS[column.long_bar_type]
            dist_capa_sup = column.cover + d_estribo + d_long / 2
            dist_capa_inf = column.width - dist_capa_sup

            acero_por_capas = []
            if column.long_bar_qty_y > 0:  # Analizando en la dirección corta (eje X)
                acero_por_capas.append(
                    {'n_barras': column.long_bar_qty_x, 'd_long': d_long, 'distancia': dist_capa_sup})
                acero_por_capas.append(
                    {'n_barras': column.long_bar_qty_x, 'd_long': d_long, 'distancia': dist_capa_inf})
                barras_intermedias = column.long_bar_qty - (2 * column.long_bar_qty_x)
                if barras_intermedias > 0 and column.long_bar_qty_y > 2:
                    n_capas_inter = column.long_bar_qty_y - 2
                    barras_por_capa_inter = barras_intermedias // n_capas_inter
                    espacio_inter = (dist_capa_inf - dist_capa_sup) / (n_capas_inter + 1)
                    for i in range(n_capas_inter):
                        dist = dist_capa_sup + (i + 1) * espacio_inter
                        acero_por_capas.append({'n_barras': barras_por_capa_inter, 'd_long': d_long, 'distancia': dist})

            analizador = DiagramaInteraccionColumna(b=column.length, h=column.width, fc=column.fc, fy=column.fy,
                                                    acero_config_layers=acero_por_capas, alpha_factor=0.725)
            pn_nom, mn_nom, pn_dis, mn_dis = analizador.generar_diagrama_completo()
            self.last_analysis_results = (pn_nom, mn_nom, pn_dis, mn_dis)  # Guardar resultados
            self._draw_interaction_diagram_on_canvas(pn_nom, mn_nom, pn_dis, mn_dis, demand_point=demand_point)
            self.log_message(f"Análisis de capacidad para {selected_col_name} completado.")
            return True
        except Exception as e:
            self.log_message(f"ERROR en el análisis de interacción: {e}")
            messagebox.showerror("Error de Análisis", f"No se pudo generar el diagrama.\nError: {e}")
            return None

    def verify_demand_point(self):
        try:
            pu = float(self.demand_pu_entry.get())
            mu = float(self.demand_mu_entry.get())
        except ValueError:
            messagebox.showerror("Entrada Inválida", "Ingrese valores numéricos para Pu y Mu.")
            return

        # Reutilizar el último análisis si está disponible, sino, ejecutar uno nuevo
        if self.last_analysis_results:
            pn_nom, mn_nom, pn_dis, mn_dis = self.last_analysis_results
            self._draw_interaction_diagram_on_canvas(pn_nom, mn_nom, pn_dis, mn_dis, demand_point=(mu, pu))
        else:
            self.run_interaction_diagram_analysis(demand_point=(mu, pu))

    def _draw_interaction_diagram_on_canvas(self, pn_nom, mn_nom, pn_dis, mn_dis, demand_point=None):
        if self.interaction_chart_widget: self.interaction_chart_widget.get_tk_widget().destroy()
        fig = plt.figure(figsize=(6, 8));
        ax = fig.add_subplot(111)
        ax.plot(mn_nom + [-x for x in mn_nom[::-1]], pn_nom + pn_nom[::-1], label='Capacidad Nominal ($P_n$ vs $M_n$)',
                color='black', linestyle='--')

        design_polygon_points = list(zip(mn_dis + [-x for x in mn_dis[::-1]], pn_dis + pn_dis[::-1]))
        ax.plot([p[0] for p in design_polygon_points], [p[1] for p in design_polygon_points],
                label='Capacidad de Diseño ($\phi P_n$ vs $\phi M_n$)', color='red', linewidth=2)

        if demand_point:
            mu, pu = demand_point
            design_path = Path(design_polygon_points)
            is_safe = design_path.contains_point((mu, pu))

            if is_safe:
                ax.plot(mu, pu, 'go', markersize=12, label=f'Demanda ({mu}, {pu})')
                self.verification_result_label.config(text="DISEÑO CONFORME", style="Success.TLabel")
                self.log_message(f"Verificación para Pu={pu} Ton, Mu={mu} Ton-m: DISEÑO CONFORME.")
            else:
                ax.plot(mu, pu, 'rX', markersize=12, label=f'Demanda ({mu}, {pu})')
                self.verification_result_label.config(text="DISEÑO NO CONFORME", style="Fail.TLabel")
                self.log_message(
                    f"Verificación para Pu={pu} Ton, Mu={mu} Ton-m: ¡DISEÑO NO CONFORME - LA SECCIÓN FALLA!")

        ax.set_title('Diagrama de Interacción P-M', fontsize=12)
        ax.set_xlabel('Momento Flector (Ton-m)', fontsize=9)
        ax.set_ylabel('Carga Axial (Ton)', fontsize=9)
        ax.legend(fontsize=8);
        ax.axhline(0, color='black', linewidth=0.5);
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5);
        fig.tight_layout()
        canvas_widget = FigureCanvasTkAgg(fig, master=self.canvas_interaction_diagram)
        canvas_widget.draw();
        canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.interaction_chart_widget = canvas_widget

    def log_message(self, message):
        logging.info(message)
        self.log_text.config(state="normal");
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END);
        self.log_text.config(state="disabled");
        self.update_idletasks()

    def run_optimization(self):
        self.log_text.config(state="normal");
        self.log_text.delete("1.0", tk.END);
        self.log_text.config(state="disabled")
        self.log_message("=" * 50);
        self.log_message("INICIANDO NUEVA SIMULACIÓN DE PROYECTO.")
        try:
            partida = {name: int(entry.get()) for name, entry in self.column_inputs.items()}
            current_stock = {name: int(entry.get()) for name, entry in self.stock_inputs.items()}
            if not any(partida.values()):
                messagebox.showwarning("Entrada Vacía", "Ingrese la cantidad para al menos un tipo de columna.");
                return
        except ValueError:
            messagebox.showerror("Error de Entrada", "Ingrese solo números enteros.");
            return
        self.log_message("Calculando despiece detallado...")
        detailed_reqs_per_type = {name: col.calculate_requirements() for name, col in self.column_types.items()}
        self.display_bill_of_materials(partida, detailed_reqs_per_type)
        self.log_message("Consolidando requerimientos totales de acero...")
        total_project_requirements = defaultdict(list)
        for name, quantity in partida.items():
            if quantity > 0:
                for req_type, details in detailed_reqs_per_type[name].items():
                    total_project_requirements[details['bar_type']].extend(details['pieces'] * quantity)
        self.log_message("Iniciando motor de optimización ILP para planes de corte...")
        order_request, cutting_plan, final_stock = self.generate_optimal_plans(total_project_requirements,
                                                                               current_stock)
        self.last_cutting_plan = cutting_plan
        self.log_message("Optimización completada.")
        self.display_summary(order_request, cutting_plan)
        self.draw_cutting_plan_visual(cutting_plan, total_project_requirements)
        self.update_detail_visuals()
        self.log_message("Simulación finalizada con éxito.")
        messagebox.showinfo("Optimización Completa", "El análisis ha finalizado. Revise los resultados.")

    def display_bill_of_materials(self, partida, detailed_reqs):
        self.bom_text.delete("1.0", tk.END)
        self.bom_text.tag_configure("header", font=("Consolas", 12, "bold", "underline"))
        self.bom_text.tag_configure("subheader", font=("Consolas", 10, "bold"))
        for name, quantity in partida.items():
            if quantity == 0: continue
            self.bom_text.insert(tk.END, f"ELEMENTO: {name} (CANTIDAD: {quantity})\n", "header")
            reqs = detailed_reqs[name]
            long_req = reqs['longitudinal']
            self.bom_text.insert(tk.END, f"\n  ACERO LONGITUDINAL ({long_req['bar_type']}):\n", "subheader")
            self.bom_text.insert(tk.END, f"    - Detalle: {long_req['description']}\n")
            if long_req['has_splices']: self.bom_text.insert(tk.END,
                                                             f"    - NOTA: Empalme por traslape de {long_req['splice_length']:.2f} m.\n")
            stirrup_req = reqs['stirrups']
            self.bom_text.insert(tk.END, f"\n  ESTRIBOS ({stirrup_req['bar_type']}):\n", "subheader")
            piece_len = stirrup_req['pieces'][0] if stirrup_req['pieces'] else 0
            self.bom_text.insert(tk.END,
                                 f"    - Cant. por Columna: {stirrup_req['total_quantity']} estribos de {piece_len:.2f} m\n")
            self.bom_text.insert(tk.END, f"    - Distribución: {stirrup_req['distribution']}\n")
            if 'supplementary_hooks' in reqs:
                hook_req = reqs['supplementary_hooks']
                self.bom_text.insert(tk.END, f"\n  GANCHOS SUPLEMENTARIOS ({hook_req['bar_type']}):\n", "subheader")
                piece_len_h = hook_req['pieces'][0] if hook_req['pieces'] else 0
                self.bom_text.insert(tk.END,
                                     f"    - Cant. por Columna: {hook_req['total_quantity']} ganchos de {piece_len_h:.2f} m\n")
            self.bom_text.insert(tk.END, "=" * 70 + "\n\n")

    def generate_optimal_plans(self, requirements, current_stock):
        cutting_plan, order_request, stock_length_m = {}, {}, 9.0
        for bar_type, pieces in requirements.items():
            if not pieces: continue
            self.log_message(f"Optimizando cortes para barras de {bar_type}...")
            demand_dict = defaultdict(int)
            for p in pieces: demand_dict[p] += 1
            status, total_bars_needed, plan = self.solve_csp_with_ilp(stock_length_m, demand_dict)
            self.log_message(f"  -> Estado del Solver para {bar_type}: {status}")
            if status != 'Optimal': messagebox.showerror("Error de Optimización",
                                                         f"No se encontró solución para {bar_type}."); continue
            order_request[bar_type] = max(0, math.ceil(total_bars_needed) - current_stock.get(bar_type, 0))
            formatted_plan = []
            if plan:
                for info in plan.values():
                    for _ in range(info['cantidad']):
                        formatted_plan.append(
                            {'cuts': info['patron'], 'waste_m': round(stock_length_m - sum(info['patron']), 2)})
            cutting_plan[bar_type] = formatted_plan
        return order_request, cutting_plan, {}

    def solve_csp_with_ilp(self, stock_length, demand):
        unique_piece_lengths = list(demand.keys())
        patterns = self._generate_patterns(stock_length, unique_piece_lengths)
        if not patterns: return "No Patterns", 0, None
        model = pulp.LpProblem("Optimizacion_Corte_Acero", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("Patron", range(len(patterns)), lowBound=0, cat='Integer')
        model += pulp.lpSum(x[i] for i in range(len(patterns))), "Total_Barras_Utilizadas"
        piece_counts_in_pattern = [defaultdict(int) for _ in patterns]
        for i, p in enumerate(patterns):
            for piece_len in p: piece_counts_in_pattern[i][piece_len] += 1
        for piece_len, required_qty in demand.items():
            model += pulp.lpSum(
                piece_counts_in_pattern[i][piece_len] * x[i] for i in range(len(patterns))) >= required_qty
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[model.status]
        if status == 'Optimal':
            plan = {i: {'patron': patterns[i], 'cantidad': int(round(pulp.value(x[i])))} for i in range(len(patterns))
                    if pulp.value(x[i]) > 0.1}
            return status, pulp.value(model.objective), plan
        return status, 0, None

    def _generate_patterns(self, stock_length, piece_lengths):
        patterns = [];
        sorted_pieces = sorted(list(set(p for p in piece_lengths if p <= stock_length)), reverse=True)
        if not sorted_pieces: return []
        for piece in sorted_pieces: patterns.append([piece] * int(stock_length / piece))
        for i in range(len(sorted_pieces)):
            current_pattern, remaining_space = [], stock_length
            for j in range(i, len(sorted_pieces)):
                piece = sorted_pieces[j]
                while remaining_space >= piece: current_pattern.append(piece); remaining_space -= piece
            if current_pattern: patterns.append(current_pattern)
        return [list(p) for p in set(tuple(sorted(p)) for p in patterns)]

    def get_color_for_piece(self, length):
        if length not in self.color_map: self.color_map[length] = self.piece_colors[
            len(self.color_map) % len(self.piece_colors)]
        return self.color_map[length]

    def display_summary(self, order_request, cutting_plan):
        for widget in self.summary_frame.winfo_children(): widget.destroy()
        total_waste = sum(bar['waste_m'] for plans in cutting_plan.values() for bar in plans)
        total_bars_used = sum(len(plans) for plans in cutting_plan.values())
        efficiency = (1 - total_waste / (total_bars_used * 9.0)) * 100 if total_bars_used > 0 else 100
        ttk.Label(self.summary_frame, text=f"Barras a Utilizar: {total_bars_used}", font=("Segoe UI", 12, "bold"),
                  foreground="#3b82f6").pack(side="left", padx=10)
        ttk.Label(self.summary_frame, text=f"Merma Total Óptima: {total_waste:.2f} m", font=("Segoe UI", 12, "bold"),
                  foreground="#ef4444").pack(side="left", padx=10)
        ttk.Label(self.summary_frame, text=f"Eficiencia del Plan: {efficiency:.2f}%", font=("Segoe UI", 12, "bold"),
                  foreground="#16a34a").pack(side="left", padx=10)

    def draw_cutting_plan_visual(self, cutting_plan, total_project_requirements):
        for widget in self.cutting_plan_container.winfo_children(): widget.destroy()
        self.color_map.clear()
        if not cutting_plan: return
        for bar_type, pattern_summary_list in cutting_plan.items():
            ttk.Label(self.cutting_plan_container, text=f"PLAN DE CORTE PARA BARRAS DE {bar_type}",
                      style="Title.TLabel").pack(anchor="w", pady=(15, 2))
            demand_summary_frame = ttk.Frame(self.cutting_plan_container);
            demand_summary_frame.pack(fill="x", pady=(0, 10))
            ttk.Label(demand_summary_frame, text="Demanda a Cubrir:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            demand_for_type = defaultdict(int)
            for piece in total_project_requirements.get(bar_type, []): demand_for_type[round(piece, 2)] += 1
            row_frame = ttk.Frame(demand_summary_frame);
            row_frame.pack(fill="x")
            for length, qty in sorted(demand_for_type.items()):
                color = self.get_color_for_piece(length)
                item_frame = ttk.Frame(row_frame);
                item_frame.pack(side="left", padx=10, pady=2)
                tk.Label(item_frame, bg=color, width=2, height=1).pack(side="left")
                ttk.Label(item_frame, text=f"{qty} x {length:.2f}m").pack(side="left", padx=5)
            visual_plan = defaultdict(int)
            for bar_info in pattern_summary_list: visual_plan[tuple(sorted(bar_info['cuts']))] += 1
            for pattern, count in visual_plan.items(): self.draw_single_pattern_bar(self.cutting_plan_container,
                                                                                    pattern, count)

    def draw_single_pattern_bar(self, parent, pattern, count):
        bar_height, stock_length = 35, 9.0;
        frame = ttk.Frame(parent);
        frame.pack(fill="x", pady=4, padx=5)
        ttk.Label(frame, text=f"Cortar {count}x:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        bar_canvas = Canvas(frame, height=bar_height, bg="#e5e7eb", highlightthickness=0)
        bar_canvas.pack(side="left", expand=True, fill="x")
        self.after(100, lambda c=bar_canvas, p=pattern: self._draw_bar_pieces(c, p, stock_length))

    def _draw_bar_pieces(self, canvas, pattern, stock_length):
        canvas_width = canvas.winfo_width()
        if canvas_width < 20: return
        current_x = 0
        for piece in pattern:
            color = self.get_color_for_piece(piece)
            piece_width = (piece / stock_length) * canvas_width
            canvas.create_rectangle(current_x, 0, current_x + piece_width, 35, fill=color, outline="#f1f5f9", width=2)
            if piece_width > 40: canvas.create_text(current_x + piece_width / 2, 17.5, text=f"{piece:.2f}",
                                                    fill="white", font=("Segoe UI", 8, "bold"))
            current_x += piece_width
        waste = stock_length - sum(pattern)
        if waste > 0.01: canvas.create_rectangle(current_x, 0, canvas_width, 35, fill="#94a3b8", outline="#f1f5f9",
                                                 stipple="gray25")

    def update_detail_visuals(self, event=None):
        selected_col_name = self.vis_selector.get()
        if not selected_col_name: return
        column = self.column_types[selected_col_name]
        self.draw_cross_section(column)
        self.draw_stirrup_distribution(column)
        if self.interaction_chart_widget: self.interaction_chart_widget.get_tk_widget().destroy(); self.interaction_chart_widget = None
        self.verification_result_label.config(text="")  # Limpiar resultado
        self.last_analysis_results = None  # Limpiar cache de resultados

    def draw_cross_section(self, column):
        canvas = self.canvas_cross_section;
        canvas.delete("all")
        padding, bar_radius = 50, 7;
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 50: self.after(100, lambda: self.draw_cross_section(column)); return
        scale = min((canvas_w - 2 * padding) / column.length, (canvas_h - 2 * padding) / column.width)
        offset_x = (canvas_w - column.length * scale) / 2;
        offset_y = (canvas_h - column.width * scale) / 2
        x1, y1 = offset_x, offset_y;
        x2, y2 = offset_x + column.length * scale, offset_y + column.width * scale
        canvas.create_rectangle(x1, y1, x2, y2, fill="#e2e8f0", outline="#475569")
        sx1, sy1 = x1 + column.cover * scale, y1 + column.cover * scale
        sx2, sy2 = x2 - column.cover * scale, y2 - column.cover * scale
        canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#0f172a", width=2.5)
        bar_positions = []
        if column.long_bar_qty_x > 1:
            x_spacing = (sx2 - sx1) / (column.long_bar_qty_x - 1)
            for i in range(column.long_bar_qty_x): bar_positions.append(
                (sx1 + i * x_spacing, sy1)); bar_positions.append((sx1 + i * x_spacing, sy2))
        if column.long_bar_qty_y > 2:
            y_spacing = (sy2 - sy1) / (column.long_bar_qty_y - 1)
            for i in range(1, column.long_bar_qty_y - 1): bar_positions.append(
                (sx1, sy1 + i * y_spacing)); bar_positions.append((sx2, sy1 + i * y_spacing))
        for (bar_x, bar_y) in bar_positions: canvas.create_oval(bar_x - bar_radius, bar_y - bar_radius,
                                                                bar_x + bar_radius, bar_y + bar_radius, fill="#334155",
                                                                outline="")
        if column.supplementary_stirrups_qty > 0 and column.long_bar_qty_x > 2:
            hook_spacing = (sx2 - sx1) / (column.long_bar_qty_x - 1)
            for i in range(column.supplementary_stirrups_qty):
                hook_x = sx1 + hook_spacing * (i + 1)
                if hook_x < sx2 - hook_spacing / 2:
                    canvas.create_line(hook_x, sy1, hook_x, sy2, fill="#0f172a", width=2)
                    canvas.create_arc(hook_x - bar_radius, sy1 - bar_radius, hook_x + bar_radius, sy1 + bar_radius,
                                      start=180, extent=180, style=tk.ARC, outline="#0f172a", width=2)
                    canvas.create_arc(hook_x - bar_radius, sy2 - bar_radius, hook_x + bar_radius, sy2 + bar_radius,
                                      start=0, extent=180, style=tk.ARC, outline="#0f172a", width=2)
        canvas.create_line(x1, y1 - 20, x2, y1 - 20, arrow=tk.BOTH);
        canvas.create_text((x1 + x2) / 2, y1 - 25, text=f"{column.length} cm", anchor="s")
        canvas.create_line(x1 - 20, y1, x1 - 20, y2, arrow=tk.BOTH);
        canvas.create_text(x1 - 25, (y1 + y2) / 2, text=f"{column.width} cm", angle=90, anchor="s")
        canvas.create_text(canvas_w / 2, 20, text=f"Corte Transversal: {column.name}", font=("Segoe UI", 11, "bold"))

    def draw_stirrup_distribution(self, column):
        canvas = self.canvas_elevation;
        canvas.delete("all")
        padding, canvas_w, canvas_h = 40, canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 50: self.after(100, lambda: self.draw_stirrup_distribution(column)); return
        col_draw_w = 120
        scale_h = (canvas_h - 2 * padding) / (column.height * 100)
        offset_x, bar_offset = (canvas_w - col_draw_w) / 2, 10
        x1, y1 = offset_x, padding;
        x2, y2 = offset_x + col_draw_w, padding + column.height * 100 * scale_h
        canvas.create_rectangle(x1, y1, x2, y2, fill="#e2e8f0", outline="")
        reqs = column.calculate_requirements()
        if reqs['longitudinal']['has_splices']:
            splice_len_px = reqs['longitudinal']['splice_length'] * 100 * scale_h
            splice_mid_y = y1 + (y2 - y1) / 2
            canvas.create_line(x1 + bar_offset, y2, x1 + bar_offset, splice_mid_y - splice_len_px / 2, width=3,
                               fill="#334155")
            canvas.create_line(x1 + bar_offset + 5, splice_mid_y + splice_len_px / 2, x1 + bar_offset + 5, y1, width=3,
                               fill="#334155")
            canvas.create_line(x2 - bar_offset, y2, x2 - bar_offset, splice_mid_y - splice_len_px / 2, width=3,
                               fill="#334155")
            canvas.create_line(x2 - bar_offset - 5, splice_mid_y + splice_len_px / 2, x2 - bar_offset - 5, y1, width=3,
                               fill="#334155")
            dim_x_splice = x1 - 20
            canvas.create_line(dim_x_splice, splice_mid_y - splice_len_px / 2, dim_x_splice,
                               splice_mid_y + splice_len_px / 2, arrow=tk.BOTH)
            canvas.create_text(dim_x_splice - 5, splice_mid_y,
                               text=f"Traslape\n{reqs['longitudinal']['splice_length']:.2f}m", anchor="e")
        else:
            canvas.create_line(x1 + bar_offset, y2, x1 + bar_offset, y1, width=3, fill="#334155")
            canvas.create_line(x2 - bar_offset, y2, x2 - bar_offset, y1, width=3, fill="#334155")

        y_bottom_limit = y2 - column.confined_length * scale_h
        y_top_limit = y1 + column.confined_length * scale_h
        for i in range(4): canvas.create_line(x1, y1 + 5 * scale_h + i * column.stirrup_spacing_confined * scale_h, x2,
                                              y1 + 5 * scale_h + i * column.stirrup_spacing_confined * scale_h, width=2,
                                              fill="#0f172a")
        for i in range(4): canvas.create_line(x1, y2 - 5 * scale_h - i * column.stirrup_spacing_confined * scale_h, x2,
                                              y2 - 5 * scale_h - i * column.stirrup_spacing_confined * scale_h, width=2,
                                              fill="#0f172a")
        for i in range(3): canvas.create_line(x1, (y_bottom_limit + y_top_limit) / 2 + (
                    i - 1) * column.stirrup_spacing_central * scale_h, x2, (y_bottom_limit + y_top_limit) / 2 + (
                                                          i - 1) * column.stirrup_spacing_central * scale_h, width=2,
                                              fill="#0f172a")
        dim_x_dist = x2 + 50
        canvas.create_line(dim_x_dist, y1, dim_x_dist, y_top_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x_dist + 5, (y1 + y_top_limit) / 2, anchor="w",
                           text=f"{reqs['stirrups']['qty_confined_zone']} Est.\n@ {column.stirrup_spacing_confined} cm")
        canvas.create_line(dim_x_dist, y_top_limit, dim_x_dist, y_bottom_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x_dist + 5, (y_top_limit + y_bottom_limit) / 2, anchor="w",
                           text=f"{reqs['stirrups']['qty_central']} Est.\n@ {column.stirrup_spacing_central} cm")
        canvas.create_line(dim_x_dist, y_bottom_limit, dim_x_dist, y2, arrow=tk.BOTH)
        canvas.create_text(dim_x_dist + 5, (y_bottom_limit + y2) / 2, anchor="w",
                           text=f"{reqs['stirrups']['qty_confined_zone']} Est.\n@ {column.stirrup_spacing_confined} cm")
        canvas.create_text(canvas_w / 2, 15, text=f"Elevación: Detalle de Acero ({column.name})",
                           font=("Segoe UI", 11, "bold"))


if __name__ == "__main__":
    app = AdvancedSteelSimulator()
    app.mainloop()

