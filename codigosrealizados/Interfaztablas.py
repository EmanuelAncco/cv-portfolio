import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from PIL import Image, ImageTk # Necesario para mostrar imágenes PNG en Tkinter

# --- Configuración ---
# !!! IMPORTANTE: Verifica que 'mapa_sismico.png' sea el nombre correcto de tu archivo !!!
SEISMIC_MAP_PATH = r'C:\Users\Emanuel\Downloads\mapa_sismico.png' # <-- RUTA ACTUALIZADA

# --- Función de cálculo de esfuerzos (sin cambios) ---
def calculate_pipeline_stress(
    D, t, E, nu, alpha_T, Sy,
    H,
    p, delta_T,
    W_traffic, If,
    PGV, C, alpha_seismic
):
    """
    Calcula los esfuerzos en una tubería enterrada bajo cargas combinadas.
    (Implementación sin cambios respecto a la versión anterior)
    """
    # --- Ecuaciones del Modelo Analítico ---
    sigma_h = (p * D) / (2 * t) if t > 1e-9 else 0
    sigma_a_p = nu * sigma_h
    sigma_a_T = E * alpha_T * delta_T
    Hc = H - D / 2.0
    Qd = 0.0
    gamma_soil = 18000
    pressure_soil = 0
    if Hc > 0:
        pressure_soil = gamma_soil * Hc

    Qd_traffic = 0.0
    if Hc > 1e-6:
         Qd_traffic = (3 * W_traffic) / (2 * np.pi * Hc**2)

    Qd = pressure_soil + Qd_traffic * If
    Wc_load_per_meter = Qd * D

    # Cálculo de sigma_L_traf
    Wt = If * Qd_traffic * D
    X = 2 * Hc
    km = 10
    Zpipe = (np.pi * D**2 * t) / 4.0 if t > 1e-9 else 0
    sigma_L_traf = 0.0
    if km != 0 and Zpipe > 1e-9 and Wt > 0:
        M_traf = (Wt * X**2) / km
        sigma_L_traf = M_traf / Zpipe

    # --- Otros esfuerzos ---
    sigma_a_w = 0.0
    if C != 0:
        sigma_a_w = E * alpha_seismic * (PGV / C)
    sigma_L = sigma_a_p + sigma_a_T + sigma_L_traf + sigma_a_w
    sigma_h_total = sigma_h

    term_inside_sqrt = sigma_L**2 - sigma_L * sigma_h_total + sigma_h_total**2
    if term_inside_sqrt < 0:
        term_inside_sqrt = 0

    sigma_VM = np.sqrt(term_inside_sqrt)
    ratio = float('inf')
    if Sy != 0:
        ratio = sigma_VM / Sy
    results = {
        "sigma_h": sigma_h, "sigma_a_p": sigma_a_p, "sigma_a_T": sigma_a_T,
        "sigma_L_traf": sigma_L_traf, "sigma_a_w": sigma_a_w, "sigma_L_total": sigma_L,
        "sigma_h_total": sigma_h_total, "sigma_VM": sigma_VM, "Ratio_VM_Sy": ratio,
        "Wc_load_per_meter": Wc_load_per_meter
    }
    return results

# --- Datos de vehículos pesados ACTUALIZADOS ---
# Fuentes: AASHTO HS20/HL93, Límites comunes MTC Perú (aproximados por rueda)
VEHICLES = {
    "AASHTO HS20 (Rueda 71 kN)": {"W_traffic": 71170, "A_contacto": 0.15}, # 16 kip wheel load
    "AASHTO HL-93 (Rueda 72 kN)": {"W_traffic": 72000, "A_contacto": 0.15}, # Approx. HL-93 truck wheel
    "Eje Simple Típico (Rueda 55 kN)": {"W_traffic": 55000, "A_contacto": 0.12}, # Basado en límite ~11 Ton/eje
    "Eje Tándem Típico (Rueda 45 kN)": {"W_traffic": 45000, "A_contacto": 0.12}, # Basado en límite ~18 Ton/tándem
    "Eje Trídem Típico (Rueda 42 kN)": {"W_traffic": 42000, "A_contacto": 0.12}, # Basado en límite ~25 Ton/trídem
    "Vehículo Personalizado": {"W_traffic": 35500, "A_contacto": 0.1} # Mantener opción personalizada
}
# Actualizar nombre por defecto si cambió la clave
DEFAULT_VEHICLE_NAME = "AASHTO HS20 (Rueda 71 kN)"
if DEFAULT_VEHICLE_NAME not in VEHICLES:
    DEFAULT_VEHICLE_NAME = list(VEHICLES.keys())[0] # Tomar el primero si el por defecto no existe

# --- Valores por defecto (actualizar W_traffic/A_contacto si cambió el vehículo por defecto) ---
DEFAULT_PARAMS = {
    "D": 0.61, "t": 0.0095, "E": 2.07e11, "nu": 0.3, "alpha_T": 1.2e-5,
    "Sy": 4.48e8, "H": 1.5, "p": 7e6, "delta_T": -15, "If": 1.5,
    "PGV": 0.40, "C": 800, "alpha_seismic": 1.0,
    "W_traffic": VEHICLES[DEFAULT_VEHICLE_NAME]["W_traffic"],
    "A_contacto": VEHICLES[DEFAULT_VEHICLE_NAME]["A_contacto"]
}
DEFAULT_LOCATION = "Miraflores"

# --- Parámetros específicos por ubicación (sin cambios) ---
LOCATION_PARAMS = {
    "Miraflores": {"PGV": 0.40, "C": 800},
    "La Molina": {"PGV": 0.50, "C": 400},
    "Villa El Salvador": {"PGV": 0.60, "C": 250}
}


# --- Función para generar el diagrama de carga de tráfico CORREGIDO ---
def plot_simple_traffic_load(D, H, W_traffic):
    """
    Genera una figura de Matplotlib con una visualización estilo rueda centrada
    de la carga de tráfico (W_traffic) sobre la tubería enterrada. VERSIÓN CORREGIDA.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Escala y Límites
    plot_width = max(D * 4, 3.0)
    # Asegurar que la profundidad sea suficiente para ver tubería y algo debajo
    plot_depth = max(H + D * 1.5, 2.5)
    ax.set_xlim(-plot_width/2, plot_width/2)
    # Establecer límites Y: 0 en superficie, aumenta hacia abajo
    ax.set_ylim(plot_depth, -0.5) # Y va de -0.5 (arriba) a plot_depth (abajo)

    # --- Elementos Visuales ---
    pipe_center_x = 0
    pipe_center_y = H # Coordenada Y del centro de la tubería
    soil_color = '#CDBA96'
    soil_edge_color = '#8B7D6B'
    pipe_color = '#A4B8C4'
    pipe_edge_color = '#556270'
    load_line_color = '#FF7F50'
    load_arrow_color = '#CD5C5C'

    # Suelo
    ax.add_patch(patches.Rectangle((-plot_width/2, 0), plot_width, plot_depth, facecolor=soil_color, edgecolor=soil_edge_color, linewidth=1, alpha=0.8))

    # Superficie (en Y=0)
    ax.plot([-plot_width/2, plot_width/2], [0, 0], color=soil_edge_color, linewidth=1.5)

    # Tubería (centrada en Y=H)
    pipe_outer = patches.Circle((pipe_center_x, pipe_center_y), D/2, facecolor=pipe_color, edgecolor=pipe_edge_color, linewidth=1.5, zorder=5)
    ax.add_patch(pipe_outer)
    ax.text(pipe_center_x, pipe_center_y, f'Tubería (D={D:.2f}m)', ha='center', va='center', color='white', fontsize=8, fontweight='bold', zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc=pipe_edge_color, ec="none", alpha=0.7))

    # Rueda (centrada en la superficie Y=0)
    tire_radius = D * 0.25
    tire_x = 0
    tire_y = 0 # Posición Y de la rueda es la superficie
    ax.add_patch(patches.Circle((tire_x, tire_y), tire_radius, facecolor='#696969', edgecolor='black', linewidth=1, zorder=8))

    # Flecha de Carga (W_traffic) - Desde arriba hacia la rueda
    arrow_start_y = -0.3 # Empezar un poco arriba (Y negativo)
    arrow_end_y = tire_y - tire_radius # Terminar justo encima de la rueda (Y=0 - radio)
    arrow_x = tire_x
    arrow_head_width = plot_width * 0.05
    arrow_head_length = 0.15

    if W_traffic > 0:
        # La flecha va de Y negativo a Y=0 (o un poco menos)
        ax.arrow(arrow_x, arrow_start_y, 0, arrow_end_y - arrow_start_y,
                 head_width=arrow_head_width, head_length=arrow_head_length, fc=load_arrow_color, ec=load_arrow_color, length_includes_head=True, zorder=9)
        # Etiqueta de la carga
        ax.text(arrow_x + plot_width*0.1, arrow_start_y, f'Carga={W_traffic/1000:.1f} kN',
                ha='left', va='center', color=load_arrow_color, fontsize=9, fontweight='bold')

    # Líneas de distribución de carga (desde los bordes de la rueda en Y=0)
    dispersion_angle_rad = np.arctan(1/1) # 45 grados
    start_x_left = tire_x - tire_radius
    start_x_right = tire_x + tire_radius
    start_y = tire_y # Y = 0

    # Calcular puntos finales (hasta la corona de la tubería Y = H - D/2)
    end_y = H - D/2 # Coordenada Y de la corona
    if end_y > start_y: # Solo dibujar si la tubería está debajo de la rueda
        dist_y = end_y - start_y # = H - D/2
        dist_x = dist_y / np.tan(dispersion_angle_rad) # tan(45)=1 -> dist_x = dist_y
        end_x_left = start_x_left - dist_x
        end_x_right = start_x_right + dist_x
        ax.plot([start_x_left, end_x_left], [start_y, end_y], color=load_line_color, linestyle='--', linewidth=1.5, zorder=4)
        ax.plot([start_x_right, end_x_right], [start_y, end_y], color=load_line_color, linestyle='--', linewidth=1.5, zorder=4)

    # Etiqueta Profundidad H (a la izquierda)
    # Asegurarse que la etiqueta esté entre Y=0 y Y=H
    label_h_y = H / 2
    if label_h_y < 0: label_h_y = 0.1 # Evitar ponerla fuera si H es muy pequeño
    ax.text(-plot_width/2 * 0.8, label_h_y, f'H={H:.2f}m', ha='center', va='center', fontsize=9, color='black', rotation=90)


    # --- Configuración Final del Gráfico ---
    ax.set_title('Diagrama Conceptual de Carga de Tráfico', fontsize=11, pad=10)
    ax.set_xlabel('Distancia Horizontal (m)', fontsize=10)
    ax.set_ylabel('Profundidad (m)', fontsize=10)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.7)
    # NO invertir el eje Y, ya que set_ylim(max, min) lo hace implícitamente.
    # ax.invert_yaxis() # <- ESTO NO ES NECESARIO CON ylim(max, min)

    plt.tight_layout(pad=0.5)
    return fig


# --- Interfaz Gráfica (Tkinter) ---
class PipelineStressApp:
    # __init__ y otros métodos de la GUI (create_input_fields, create_output_labels, etc.)
    # permanecen IGUALES que en la versión anterior.
    # Solo se actualiza el diccionario VEHICLES usado en create_input_fields.

    def __init__(self, root):
        self.root = root
        root.title("Calculadora de Esfuerzos en Tuberías Enterradas")
        # Configuración de estilo (igual que antes)
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            print("Tema 'clam' no disponible, usando tema por defecto.")
        style.configure("TLabel", padding=5, font=('Arial', 10))
        style.configure("TEntry", padding=5, font=('Arial', 10))
        style.configure("TButton", padding=5, font=('Arial', 10, 'bold'))
        style.configure("Reset.TButton", foreground='black', background='#ffc107')
        style.map("Reset.TButton", background=[('active', '#e0a800')])
        style.configure("ClearHist.TButton", foreground='white', background='#6c757d')
        style.map("ClearHist.TButton", background=[('active', '#5a6268')])

        style.configure("TCombobox", padding=5, font=('Arial', 10))
        style.configure("TFrame", background='#f0f0f0')
        style.configure("Left.TFrame", background='#e8e8e8')
        style.configure("Right.TFrame", background='#ffffff')
        style.configure("Results.TFrame", background='#ffffff')
        style.configure("Hist.TFrame", background='#ffffff')

        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Arial', 9))
        style.map("Treeview", background=[('selected', '#007bff')], foreground=[('selected', 'white')])


        # --- Estructura Principal: Dos Paneles ---
        self.main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Panel Izquierdo (Entradas) ---
        self.left_frame_outer = ttk.Frame(self.main_paned_window, width=400, style="Left.TFrame")
        self.left_frame_outer.pack(fill=tk.BOTH, expand=False)
        self.main_paned_window.add(self.left_frame_outer, weight=0)

        self.canvas_left = tk.Canvas(self.left_frame_outer, borderwidth=0, background="#e8e8e8")
        self.scrollbar_v_left = ttk.Scrollbar(self.left_frame_outer, orient="vertical", command=self.canvas_left.yview)
        self.left_frame_inner = ttk.Frame(self.canvas_left, padding="10", style="Left.TFrame")

        self.canvas_left.configure(yscrollcommand=self.scrollbar_v_left.set)
        self.scrollbar_v_left.pack(side="right", fill="y")
        self.canvas_left.pack(side="left", fill="both", expand=True)
        self.canvas_left.create_window((0, 0), window=self.left_frame_inner, anchor="nw")

        self.left_frame_inner.columnconfigure(0, weight=1)
        self.left_frame_inner.columnconfigure(1, weight=1)

        self.input_vars = {}
        self.current_row_left = 0
        self.create_input_fields(self.left_frame_inner)

        button_frame = ttk.Frame(self.left_frame_inner, style="Left.TFrame")
        button_frame.grid(row=self.current_row_left, column=0, columnspan=3, pady=20)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="Calcular Esfuerzos", command=self.run_analysis).grid(
            row=0, column=0, padx=5, sticky=tk.E)
        ttk.Button(button_frame, text="Restaurar Valores", command=self.reset_to_defaults, style="Reset.TButton").grid(
            row=0, column=1, padx=5, sticky=tk.W)
        self.current_row_left += 1

        self.left_frame_inner.bind("<Configure>", self.on_frame_configure)


        # --- Panel Derecho (Visualizaciones, Resultados, Historial) ---
        self.right_frame = ttk.Frame(self.main_paned_window, style="Right.TFrame")
        self.right_frame.pack(fill=tk.BOTH, expand=True)
        self.main_paned_window.add(self.right_frame, weight=1)

        self.right_frame.rowconfigure(0, weight=0)
        self.right_frame.rowconfigure(1, weight=1) # Diagrama
        self.right_frame.rowconfigure(2, weight=1) # Historial
        self.right_frame.columnconfigure(0, weight=1)

        # --- Sub-Panel Superior (Mapa y Resultados) ---
        top_right_frame = ttk.Frame(self.right_frame, style="Right.TFrame", padding="5")
        top_right_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        top_right_frame.columnconfigure(0, weight=1)
        top_right_frame.columnconfigure(1, weight=1)
        top_right_frame.rowconfigure(1, weight=1)

        # Mapa Sísmico
        ttk.Label(top_right_frame, text="Mapa Sísmico (Referencial)", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(5, 2), padx=5)
        self.seismic_map_label = ttk.Label(top_right_frame, background='white', anchor='center')
        self.seismic_map_label.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=2)
        self.load_and_display_seismic_map(SEISMIC_MAP_PATH)

        # Frame para Resultados
        results_frame_right = ttk.Frame(top_right_frame, style="Results.TFrame", padding="10")
        results_frame_right.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=2)
        results_frame_right.columnconfigure(0, weight=0)
        results_frame_right.columnconfigure(1, weight=1)

        ttk.Label(top_right_frame, text="Resultados del Análisis", font=('Arial', 10, 'bold')).grid(
             row=0, column=1, sticky=(tk.W, tk.E), pady=(5, 2), padx=5)

        self.output_labels = {}
        self.create_output_labels(results_frame_right)


        # --- Diagrama (Fila 1) ---
        ttk.Label(self.right_frame, text="Diagrama Conceptual de Carga", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 2), padx=10)
        self.traffic_plot_container = ttk.Frame(self.right_frame, style="Right.TFrame")
        self.traffic_plot_container.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=5)
        self.fig_traffic = None
        self.canvas_matplotlib = None
        self.update_conceptual_diagram()


        # --- Tabla de Historial (Fila 2) ---
        hist_frame = ttk.Frame(self.right_frame, style="Hist.TFrame", padding="5")
        hist_frame.grid(row=2, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=(10, 5))
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.columnconfigure(1, weight=0)
        hist_frame.rowconfigure(1, weight=1)

        hist_label_frame = ttk.Frame(hist_frame, style="Hist.TFrame")
        hist_label_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(hist_label_frame, text="Historial de Cálculos", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        ttk.Button(hist_label_frame, text="Limpiar", command=self.clear_history, style="ClearHist.TButton", width=8).pack(side=tk.LEFT, padx=5)

        self.history_tree = ttk.Treeview(hist_frame, style="Treeview")
        self.history_tree.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        hist_scrollbar = ttk.Scrollbar(hist_frame, orient="vertical", command=self.history_tree.yview)
        hist_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.history_tree.configure(yscrollcommand=hist_scrollbar.set)

        self.history_tree['columns'] = ("sigma_l", "sigma_h", "sigma_vm", "ratio")
        self.history_tree.column("#0", width=0, stretch=tk.NO)
        self.history_tree.column("sigma_l", anchor=tk.E, width=120)
        self.history_tree.column("sigma_h", anchor=tk.E, width=120)
        self.history_tree.column("sigma_vm", anchor=tk.E, width=120)
        self.history_tree.column("ratio", anchor=tk.E, width=80)

        self.history_tree.heading("#0", text="", anchor=tk.W)
        self.history_tree.heading("sigma_l", text="σL Total (MPa)", anchor=tk.CENTER)
        self.history_tree.heading("sigma_h", text="σh Total (MPa)", anchor=tk.CENTER)
        self.history_tree.heading("sigma_vm", text="σVM (MPa)", anchor=tk.CENTER)
        self.history_tree.heading("ratio", text="Ratio", anchor=tk.CENTER)

        # Llamar a on_vehicle_selected al final para asegurar que los widgets existen
        self.on_vehicle_selected(None)


    def create_input_fields(self, parent_frame):
        """Crea las etiquetas y campos de entrada en el panel izquierdo."""
        # ... (código igual que antes, pero usa el VEHICLES actualizado) ...
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.columnconfigure(1, weight=2)
        parent_frame.columnconfigure(2, weight=0)
        row = self.current_row_left
        ttk.Label(parent_frame, text="Ubicación (Cargar Parámetros):").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.location_combobox = ttk.Combobox(parent_frame, values=list(LOCATION_PARAMS.keys()), state="readonly", width=27)
        self.location_combobox.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.location_combobox.set(DEFAULT_LOCATION)
        self.location_combobox.bind("<<ComboboxSelected>>", self.on_location_selected)
        row += 1
        ttk.Label(parent_frame, text="--- Tubería y Material ---", font=('Arial', 10, 'italic')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5,2), padx=5)
        row += 1
        parameters_pipe = {"D": ("Diámetro exterior D:", "m"), "t": ("Espesor de pared t:", "m"), "E": ("Módulo de Young E:", "Pa"), "nu": ("Coeficiente de Poisson ν:", ""), "alpha_T": ("Coef. Expansión Térmica α_T:", "°C⁻¹"), "Sy": ("Límite de Fluencia Sy:", "Pa")}
        for key, (label_text, unit) in parameters_pipe.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
            self.input_vars[key] = var
            entry = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
            row += 1
        ttk.Label(parent_frame, text="--- Cargas y Condiciones ---", font=('Arial', 10, 'italic')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5,2), padx=5)
        row += 1
        parameters_load = {"H": ("Profundidad al eje H:", "m"), "p": ("Presión interna p:", "Pa"), "delta_T": ("Cambio de Temperatura ΔT:", "°C"), "If": ("Factor de impacto If:", "")}
        for key, (label_text, unit) in parameters_load.items():
             ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
             var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
             self.input_vars[key] = var
             entry = ttk.Entry(parent_frame, textvariable=var, width=25)
             entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
             ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
             row += 1
        ttk.Label(parent_frame, text="Vehículo de Tráfico:").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        # Usar el diccionario VEHICLES actualizado
        self.vehicle_combobox = ttk.Combobox(parent_frame, values=list(VEHICLES.keys()), state="readonly", width=27)
        self.vehicle_combobox.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.vehicle_combobox.set(DEFAULT_VEHICLE_NAME) # Usar constante actualizada
        self.vehicle_combobox.bind("<<ComboboxSelected>>", self.on_vehicle_selected)
        row += 1
        ttk.Label(parent_frame, text="Carga por rueda W_traffic:").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.input_vars["W_traffic"] = tk.StringVar(value=str(DEFAULT_PARAMS["W_traffic"]))
        self.entry_w_traffic = ttk.Entry(parent_frame, textvariable=self.input_vars["W_traffic"], width=25)
        self.entry_w_traffic.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        ttk.Label(parent_frame, text="N").grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
        row += 1
        ttk.Label(parent_frame, text="Área de contacto A_contacto:").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.input_vars["A_contacto"] = tk.StringVar(value=str(DEFAULT_PARAMS["A_contacto"]))
        self.entry_a_contacto = ttk.Entry(parent_frame, textvariable=self.input_vars["A_contacto"], width=25)
        self.entry_a_contacto.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        ttk.Label(parent_frame, text="m²").grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
        row += 1
        ttk.Label(parent_frame, text="--- Parámetros Sísmicos (TGD) ---", font=('Arial', 10, 'italic')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5,2), padx=5)
        row += 1
        parameters_seismic = {"PGV": ("Velocidad Pico del Suelo PGV:", "m/s"), "C": ("Velocidad de Onda C:", "m/s"), "alpha_seismic": ("Factor α (Sísmico):", "")}
        for key, (label_text, unit) in parameters_seismic.items():
             ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
             var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
             self.input_vars[key] = var
             entry = ttk.Entry(parent_frame, textvariable=var, width=25)
             entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
             ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
             row += 1
        self.current_row_left = row


    def create_output_labels(self, parent_frame):
        """Crea las etiquetas para mostrar los resultados en el panel DERECHO."""
        # ... (código igual que antes) ...
        results_order = [("sigma_h", "σh (Presión):"), ("sigma_a_p", "σa,p (Axial Presión):"), ("sigma_a_T", "σa,T (Axial Temp.):"), ("sigma_L_traf", "σL,traf (Long. Tráfico):"), ("sigma_a_w", "σa,w (Axial TGD):"), ("sigma_L_total", "σL Total:"), ("sigma_h_total", "σh Total:"), ("sigma_VM", "σVM (Von Mises):"), ("Ratio_VM_Sy", "Ratio σVM/Sy:")]
        row = 0
        for key, label_text in results_order:
            ttk.Label(parent_frame, text=label_text, anchor='w').grid(row=row, column=0, sticky=tk.W, pady=1, padx=5)
            label_var = tk.StringVar(value="--")
            result_label = ttk.Label(parent_frame, textvariable=label_var, font=('Arial', 10, 'bold'), anchor='e')
            result_label.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=1, padx=5)
            self.output_labels[key] = label_var
            row += 1


    def on_vehicle_selected(self, event):
        """Actualiza campos de tráfico y redibuja el diagrama simple."""
        # ... (código igual que antes) ...
        selected_vehicle = self.vehicle_combobox.get()
        if selected_vehicle in VEHICLES:
            traffic_data = VEHICLES[selected_vehicle]
            if "W_traffic" in self.input_vars: self.input_vars["W_traffic"].set(str(traffic_data["W_traffic"]))
            if "A_contacto" in self.input_vars: self.input_vars["A_contacto"].set(str(traffic_data["A_contacto"]))
            if selected_vehicle == "Vehículo Personalizado":
                self.entry_w_traffic.config(state="normal")
                self.entry_a_contacto.config(state="normal")
            else:
                # Asegurarse que los widgets existen antes de configurarlos
                if hasattr(self, 'entry_w_traffic'): self.entry_w_traffic.config(state="disabled")
                if hasattr(self, 'entry_a_contacto'): self.entry_a_contacto.config(state="disabled")
        self.update_conceptual_diagram()


    def on_location_selected(self, event):
        """Carga los parámetros para la ubicación seleccionada."""
        # ... (código igual que antes) ...
        selected_location = self.location_combobox.get()
        if selected_location in LOCATION_PARAMS:
            location_data = LOCATION_PARAMS[selected_location]
            for key, value in location_data.items():
                if key in self.input_vars: self.input_vars[key].set(str(value))
                else: print(f"Advertencia: El parámetro '{key}' de la ubicación '{selected_location}' no tiene campo de entrada asociado.")
            self.update_conceptual_diagram()


    def run_analysis(self):
        """Lee inputs, calcula, muestra resultados, actualiza diagrama y añade a historial."""
        # ... (código igual que antes) ...
        params = {}
        try:
            param_keys_for_calculation = [key for key in self.input_vars.keys() if key != "A_contacto"]
            for key in param_keys_for_calculation:
                 var = self.input_vars[key]
                 try: params[key] = float(var.get())
                 except ValueError: raise ValueError(f"Valor inválido para '{key}': '{var.get()}'. Ingrese un número.")

            results = calculate_pipeline_stress(**params)

            self.output_labels["sigma_h"].set(f"{results['sigma_h']/1e6:.3f} MPa")
            self.output_labels["sigma_a_p"].set(f"{results['sigma_a_p']/1e6:.3f} MPa")
            self.output_labels["sigma_a_T"].set(f"{results['sigma_a_T']/1e6:.3f} MPa")
            sigma_L_traf_val = results['sigma_L_traf']
            self.output_labels["sigma_L_traf"].set(f"{sigma_L_traf_val/1e6:.3f} MPa" if abs(sigma_L_traf_val) > 1e-9 else "0.000 MPa")
            self.output_labels["sigma_a_w"].set(f"{results['sigma_a_w']/1e6:.3f} MPa")
            self.output_labels["sigma_L_total"].set(f"{results['sigma_L_total']/1e6:.3f} MPa")
            self.output_labels["sigma_h_total"].set(f"{results['sigma_h_total']/1e6:.3f} MPa")
            self.output_labels["sigma_VM"].set(f"{results['sigma_VM']/1e6:.3f} MPa")
            self.output_labels["Ratio_VM_Sy"].set(f"{results['Ratio_VM_Sy']:.3f}")

            hist_values = (f"{results['sigma_L_total']/1e6:.3f}", f"{results['sigma_h_total']/1e6:.3f}", f"{results['sigma_VM']/1e6:.3f}", f"{results['Ratio_VM_Sy']:.3f}")
            self.history_tree.insert("", 0, values=hist_values)

            self.update_conceptual_diagram()

        except ValueError as ve: messagebox.showerror("Error de Entrada", str(ve))
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error durante el cálculo:\n{e}")
            import traceback
            traceback.print_exc()


    def update_conceptual_diagram(self):
        """Actualiza el diagrama SIMPLIFICADO de carga de tráfico."""
        # ... (código igual que antes) ...
        try:
            D_str = self.input_vars.get("D", tk.StringVar(value="0")).get()
            H_str = self.input_vars.get("H", tk.StringVar(value="0")).get()
            W_traffic_str = self.input_vars.get("W_traffic", tk.StringVar(value="0")).get()
            D = float(D_str); H = float(H_str); W_traffic = float(W_traffic_str)

            if self.canvas_matplotlib: self.canvas_matplotlib.get_tk_widget().destroy()
            if self.fig_traffic: plt.close(self.fig_traffic)

            self.fig_traffic = plot_simple_traffic_load(D, H, W_traffic)

            self.canvas_matplotlib = FigureCanvasTkAgg(self.fig_traffic, master=self.traffic_plot_container)
            canvas_widget = self.canvas_matplotlib.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas_matplotlib.draw()
        except ValueError:
             print("Advertencia: No se pudo actualizar el diagrama debido a valores no numéricos.")
             for widget in self.traffic_plot_container.winfo_children(): widget.destroy()
             placeholder = ttk.Label(self.traffic_plot_container, text="(Esperando datos válidos para el diagrama)", style="Right.TFrame", foreground="gray")
             placeholder.pack(expand=True)
        except Exception as e:
             print(f"Error al actualizar el diagrama simple: {e}")
             for widget in self.traffic_plot_container.winfo_children(): widget.destroy()
             error_label = ttk.Label(self.traffic_plot_container, text=f"Error al generar diagrama:\n{e}", foreground="red", wraplength=300)
             error_label.pack(padx=10, pady=10)


    def reset_to_defaults(self):
        """Restaura todos los campos de entrada y limpia resultados/historial."""
        # ... (código igual que antes) ...
        print("Restaurando valores por defecto...")
        for key, default_value in DEFAULT_PARAMS.items():
            if key in self.input_vars: self.input_vars[key].set(str(default_value))
        self.location_combobox.set(DEFAULT_LOCATION)
        self.vehicle_combobox.set(DEFAULT_VEHICLE_NAME)
        self.on_vehicle_selected(None)
        for key in self.output_labels: self.output_labels[key].set("--")
        self.clear_history()
        self.update_conceptual_diagram()
        print("Valores restaurados.")


    def clear_history(self):
        """Borra todas las entradas de la tabla de historial."""
        # ... (código igual que antes) ...
        print("Limpiando historial...")
        for item in self.history_tree.get_children(): self.history_tree.delete(item)
        print("Historial limpiado.")


    def load_and_display_seismic_map(self, image_path):
        """Carga y muestra la imagen del mapa sísmico."""
        # ... (código igual que antes) ...
        max_width = 300; max_height = 280
        try:
            if not os.path.exists(image_path):
                 try: script_dir = os.path.dirname(__file__); alt_path = os.path.join(script_dir, os.path.basename(image_path))
                 except NameError: alt_path = os.path.join(os.getcwd(), os.path.basename(image_path))
                 if not os.path.exists(alt_path): raise FileNotFoundError(f"No se encontró el archivo en:\n{image_path}\nNi en:\n{alt_path}")
                 else: image_path = alt_path
            img = Image.open(image_path)
            img_width, img_height = img.size
            ratio = min(max_width / img_width, max_height / img_height)
            if ratio < 1.0:
                 new_width = int(img_width * ratio); new_height = int(img_height * ratio)
                 img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.seismic_map_photo = ImageTk.PhotoImage(img)
            self.seismic_map_label.config(image=self.seismic_map_photo)
            self.seismic_map_label.image = self.seismic_map_photo
        except FileNotFoundError as fnf_error:
            error_text = str(fnf_error)
            self.seismic_map_label.config(text=f"Error: Imagen no encontrada.\n{error_text}", image='', compound=tk.CENTER, wraplength=max_width-20, foreground='red')
            print(f"Error: {error_text}")
        except Exception as e:
            self.seismic_map_label.config(text=f"Error al cargar imagen:\n{e}", image='', compound=tk.CENTER, wraplength=max_width-20, foreground='red')
            print(f"Error al cargar la imagen {image_path}: {e}")


    def on_frame_configure(self, event):
        """Actualiza la región de desplazamiento del canvas izquierdo."""
        # ... (código igual que antes) ...
        self.root.after_idle(lambda: self.canvas_left.configure(scrollregion=self.canvas_left.bbox("all")))


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    try:
        screen_width = root.winfo_screenwidth(); screen_height = root.winfo_screenheight()
        initial_width = int(screen_width * 0.75); initial_height = int(screen_height * 0.75)
        root.geometry(f"{initial_width}x{initial_height}")
    except Exception:
        root.geometry("1000x750")

    app = PipelineStressApp(root)
    root.minsize(900, 700)
    root.mainloop()
