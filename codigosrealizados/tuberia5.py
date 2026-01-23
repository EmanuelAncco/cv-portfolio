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
    sigma_h = (p * D) / (2 * t) if t > 1e-9 else 0 # Evitar división por cero
    sigma_a_p = nu * sigma_h
    sigma_a_T = E * alpha_T * delta_T
    Hc = H - D / 2.0
    Qd = 0.0
    gamma_soil = 18000
    pressure_soil = 0
    if Hc > 0:
        pressure_soil = gamma_soil * Hc

    Qd_traffic = 0.0
    if Hc > 1e-6: # Evitar división por cero
         Qd_traffic = (3 * W_traffic) / (2 * np.pi * Hc**2)

    Qd = pressure_soil + Qd_traffic * If
    Wc_load_per_meter = Qd * D

    # Cálculo de sigma_L_traf
    Wt = If * Qd_traffic * D
    X = 2 * Hc
    km = 10
    Zpipe = (np.pi * D**2 * t) / 4.0 if t > 1e-9 else 0 # Evitar división por cero
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

# --- Datos de vehículos pesados (sin cambios) ---
VEHICLES = {
    "Camión ligero (35.5 kN/rueda)": {"W_traffic": 35500, "A_contacto": 0.1},
    "Camión 2 ejes (carga por eje ~100 kN)": {"W_traffic": 50000, "A_contacto": 0.15},
    "Camión 3 ejes (carga por eje ~150 kN)": {"W_traffic": 50000, "A_contacto": 0.15},
    "Tráiler (carga por eje > 150 kN)": {"W_traffic": 65000, "A_contacto": 0.2},
    "Vehículo Personalizado": {"W_traffic": 35500, "A_contacto": 0.1}
}

# --- Valores por defecto (sin cambios) ---
DEFAULT_PARAMS = {
    "D": 0.61, "t": 0.0095, "E": 2.07e11, "nu": 0.3, "alpha_T": 1.2e-5,
    "Sy": 4.48e8, "H": 1.5, "p": 7e6, "delta_T": -15, "If": 1.5,
    "PGV": 0.40, "C": 800, "alpha_seismic": 1.0,
    "W_traffic": VEHICLES["Camión ligero (35.5 kN/rueda)"]["W_traffic"],
    "A_contacto": VEHICLES["Camión ligero (35.5 kN/rueda)"]["A_contacto"]
}

# --- Parámetros específicos por ubicación (sin cambios) ---
LOCATION_PARAMS = {
    "Miraflores": {"PGV": 0.40, "C": 800},
    "La Molina": {"PGV": 0.50, "C": 400},
    "Villa El Salvador": {"PGV": 0.60, "C": 250}
}


# --- NUEVA Función para generar el diagrama de carga de tráfico SIMPLE ---
def plot_simple_traffic_load(D, H, W_traffic):
    """
    Genera una figura de Matplotlib con una visualización SIMPLE
    de la carga de tráfico sobre la tubería enterrada.

    Args:
        D (float): Diámetro exterior de la tubería (m).
        H (float): Profundidad desde la superficie hasta el eje de la tubería (m).
        W_traffic (float): Carga por rueda (N).

    Returns:
        matplotlib.figure.Figure: La figura de Matplotlib creada.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5)) # Ajustar tamaño

    # Escala y Límites
    h_scale = D * 2.0  # Escala horizontal basada en diámetro
    v_scale = (H + D/2) * 1.3 # Escala vertical
    ax.set_xlim(-h_scale, h_scale)
    ax.set_ylim(H + D/2 + v_scale*0.2, -v_scale*0.3) # Invertir eje Y
    ax.set_aspect('equal', adjustable='box')

    # --- Elementos Visuales ---
    pipe_center_x = 0
    pipe_center_y = H
    soil_color = '#e3d1b1'
    soil_edge_color = '#a1887f'

    # Suelo
    ax.add_patch(patches.Rectangle((-h_scale, 0), 2*h_scale, H + D/2 + v_scale*0.2, facecolor=soil_color, edgecolor=soil_edge_color, linewidth=1, alpha=0.7))

    # Superficie
    ax.plot([-h_scale, h_scale], [0, 0], color=soil_edge_color, linewidth=2)
    ax.text(h_scale * 0.95, -0.02*v_scale, 'Superficie', ha='right', va='top', fontsize=9)

    # Tubería
    pipe_outer = patches.Circle((pipe_center_x, pipe_center_y), D/2, facecolor='#cadbeb', edgecolor='#5b8fbd', linewidth=1.5, zorder=5)
    ax.add_patch(pipe_outer)
    ax.text(pipe_center_x, pipe_center_y, f'D={D:.2f}m', ha='center', va='center', color='#191970', fontsize=8, fontweight='bold', zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Vehículo (simplificado como un rectángulo)
    vehicle_width = D * 1.5 # Ancho relativo al diámetro
    vehicle_height = D * 0.6
    vehicle_x = -vehicle_width / 2
    vehicle_y = -vehicle_height
    ax.add_patch(patches.Rectangle((vehicle_x, vehicle_y), vehicle_width, vehicle_height, facecolor='#d3d3d3', edgecolor='#808080', linewidth=1, zorder=3))

    # Llantas (simplificadas como círculos)
    tire_radius = D * 0.15
    tire_y = -tire_radius
    # Asumir dos llantas visibles en esta vista 2D
    tire_x_offset = vehicle_width * 0.35
    tire1_x = -tire_x_offset
    tire2_x = tire_x_offset
    ax.add_patch(patches.Circle((tire1_x, tire_y), tire_radius, facecolor='#5a5a5a', edgecolor='black', zorder=4))
    ax.add_patch(patches.Circle((tire2_x, tire_y), tire_radius, facecolor='#5a5a5a', edgecolor='black', zorder=4))

    # Flecha de Carga por Llanta (W_traffic)
    # Colocar la flecha sobre una de las llantas
    arrow_start_y = vehicle_y - vehicle_height * 0.3 # Un poco por encima del vehículo
    arrow_end_y = tire_y + tire_radius # Hasta la parte superior de la llanta
    arrow_x = tire1_x # Alinear con la llanta izquierda
    arrow_head_width = h_scale * 0.08
    arrow_head_length = v_scale * 0.08
    load_color = '#dc143c' # Crimson red

    if W_traffic > 0:
        ax.arrow(arrow_x, arrow_start_y, 0, arrow_end_y - arrow_start_y,
                 head_width=arrow_head_width, head_length=arrow_head_length, fc=load_color, ec=load_color, length_includes_head=True, zorder=6)
        # Etiqueta de la carga
        ax.text(arrow_x + h_scale*0.1, arrow_start_y - (arrow_start_y - arrow_end_y)/2, f'W_traffic\n{W_traffic/1000:.1f} kN\n(por llanta)',
                ha='left', va='center', color=load_color, fontsize=9)

    # Etiqueta Profundidad H
    ax.annotate(f'H={H:.2f}m', xy=(pipe_center_x + D/2, pipe_center_y), xytext=(h_scale*0.8, H * 0.5),
                ha='center', va='center', fontsize=9, color='black',
                arrowprops=dict(arrowstyle="->", color='gray', shrinkB=5))


    # --- Configuración Final del Gráfico ---
    ax.set_title('Diagrama Simplificado: Carga de Tráfico', fontsize=11, pad=10)
    # Ocultar ejes y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)

    plt.tight_layout(pad=0.5)
    return fig


# --- Interfaz Gráfica (Tkinter) ---
class PipelineStressApp:
    # __init__ y otros métodos de la GUI (create_input_fields, create_output_labels, etc.)
    # permanecen mayormente IGUALES.
    # Los cambios principales están en run_analysis y update_conceptual_diagram
    # para llamar a la nueva función de ploteo.

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
        style.configure("TCombobox", padding=5, font=('Arial', 10))
        style.configure("TFrame", background='#f0f0f0')
        style.configure("Left.TFrame", background='#e8e8e8')
        style.configure("Right.TFrame", background='#ffffff')

        # --- Estructura Principal: Dos Paneles ---
        self.main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Panel Izquierdo (Entradas y Resultados con Scroll) ---
        self.left_frame_outer = ttk.Frame(self.main_paned_window, width=450, style="Left.TFrame")
        self.left_frame_outer.pack(fill=tk.BOTH, expand=True)
        self.main_paned_window.add(self.left_frame_outer, weight=1)

        self.canvas_left = tk.Canvas(self.left_frame_outer, borderwidth=0, background="#e8e8e8")
        self.scrollbar_v_left = ttk.Scrollbar(self.left_frame_outer, orient="vertical", command=self.canvas_left.yview)
        self.left_frame_inner = ttk.Frame(self.canvas_left, padding="10", style="Left.TFrame")

        self.canvas_left.configure(yscrollcommand=self.scrollbar_v_left.set)
        self.scrollbar_v_left.pack(side="right", fill="y")
        self.canvas_left.pack(side="left", fill="both", expand=True)
        self.canvas_left.create_window((0, 0), window=self.left_frame_inner, anchor="nw")

        self.left_frame_inner.columnconfigure(0, weight=1)
        self.left_frame_inner.columnconfigure(1, weight=2)
        self.left_frame_inner.columnconfigure(2, weight=0)

        self.input_vars = {}
        self.current_row_left = 0
        self.create_input_fields(self.left_frame_inner)

        ttk.Button(self.left_frame_inner, text="Calcular Esfuerzos", command=self.run_analysis).grid(
            row=self.current_row_left, column=0, columnspan=3, pady=15)
        self.current_row_left += 1

        self.results_label = ttk.Label(self.left_frame_inner, text="Resultados del Análisis:", font=('Arial', 12, 'bold'), foreground='#0056b3')
        self.results_label.grid(row=self.current_row_left, column=0, columnspan=3, sticky=tk.W, pady=(10, 5), padx=5)
        self.current_row_left += 1

        self.output_labels = {}
        self.create_output_labels(self.left_frame_inner, row_start=self.current_row_left)

        self.left_frame_inner.bind("<Configure>", self.on_frame_configure)

        # --- Panel Derecho (Visualizaciones) ---
        self.right_frame = ttk.Frame(self.main_paned_window, style="Right.TFrame")
        self.right_frame.pack(fill=tk.BOTH, expand=True)
        self.main_paned_window.add(self.right_frame, weight=2)

        self.right_frame.rowconfigure(1, weight=1) # Mapa
        self.right_frame.rowconfigure(3, weight=2) # Diagrama
        self.right_frame.columnconfigure(0, weight=1)

        ttk.Label(self.right_frame, text="Mapa Sísmico (Referencial)", font=('Arial', 11, 'bold')).grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(10, 2), padx=10)

        self.seismic_map_label = ttk.Label(self.right_frame, background='white', anchor='center')
        self.seismic_map_label.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=5)
        self.load_and_display_seismic_map(SEISMIC_MAP_PATH) # Cargar imagen

        # Cambiar título para el nuevo diagrama
        ttk.Label(self.right_frame, text="Diagrama Simplificado de Carga", font=('Arial', 11, 'bold')).grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 2), padx=10)

        self.traffic_plot_container = ttk.Frame(self.right_frame, style="Right.TFrame")
        self.traffic_plot_container.grid(row=3, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=5)
        self.fig_traffic = None
        self.canvas_matplotlib = None
        # Inicializar diagrama simple
        self.update_conceptual_diagram() # Llamada inicial

        self.on_vehicle_selected(None) # Configurar estado inicial


    def create_input_fields(self, parent_frame):
        """Crea las etiquetas y campos de entrada (sin cambios estructurales)."""
        row = self.current_row_left
        # Selector de Ubicación
        ttk.Label(parent_frame, text="Ubicación (Cargar Parámetros):").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.location_combobox = ttk.Combobox(parent_frame, values=list(LOCATION_PARAMS.keys()), state="readonly", width=27)
        self.location_combobox.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.location_combobox.set("Miraflores")
        self.location_combobox.bind("<<ComboboxSelected>>", self.on_location_selected)
        row += 1

        # Tubería y Material
        ttk.Label(parent_frame, text="--- Tubería y Material ---", font=('Arial', 10, 'italic')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5,2), padx=5)
        row += 1
        parameters_pipe = {
            "D": ("Diámetro exterior D:", "m"), "t": ("Espesor de pared t:", "m"),
            "E": ("Módulo de Young E:", "Pa"), "nu": ("Coeficiente de Poisson ν:", ""),
            "alpha_T": ("Coef. Expansión Térmica α_T:", "°C⁻¹"), "Sy": ("Límite de Fluencia Sy:", "Pa"),
        }
        for key, (label_text, unit) in parameters_pipe.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS[key]))
            entry = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
            self.input_vars[key] = var
            row += 1

        # Cargas y Condiciones
        ttk.Label(parent_frame, text="--- Cargas y Condiciones ---", font=('Arial', 10, 'italic')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5,2), padx=5)
        row += 1
        parameters_load = {
             "H": ("Profundidad al eje H:", "m"), "p": ("Presión interna p:", "Pa"),
             "delta_T": ("Cambio de Temperatura ΔT:", "°C"), "If": ("Factor de impacto If:", ""),
        }
        for key, (label_text, unit) in parameters_load.items():
             ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
             var = tk.StringVar(value=str(DEFAULT_PARAMS[key]))
             entry = ttk.Entry(parent_frame, textvariable=var, width=25)
             entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
             ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
             self.input_vars[key] = var
             row += 1

        # Vehículo de Tráfico
        ttk.Label(parent_frame, text="Vehículo de Tráfico:").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.vehicle_combobox = ttk.Combobox(parent_frame, values=list(VEHICLES.keys()), state="readonly", width=27)
        self.vehicle_combobox.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.vehicle_combobox.set("Camión ligero (35.5 kN/rueda)")
        self.vehicle_combobox.bind("<<ComboboxSelected>>", self.on_vehicle_selected)
        row += 1
        # W_traffic y A_contacto
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

        # Parámetros Sísmicos
        ttk.Label(parent_frame, text="--- Parámetros Sísmicos (TGD) ---", font=('Arial', 10, 'italic')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5,2), padx=5)
        row += 1
        parameters_seismic = {
            "PGV": ("Velocidad Pico del Suelo PGV:", "m/s"), "C": ("Velocidad de Onda C:", "m/s"),
            "alpha_seismic": ("Factor α (Sísmico):", ""),
        }
        for key, (label_text, unit) in parameters_seismic.items():
             ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
             var = tk.StringVar(value=str(DEFAULT_PARAMS[key]))
             entry = ttk.Entry(parent_frame, textvariable=var, width=25)
             entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
             ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
             self.input_vars[key] = var
             row += 1

        self.current_row_left = row


    def create_output_labels(self, parent_frame, row_start):
        """Crea las etiquetas para mostrar los resultados (sin cambios estructurales)."""
        results_order = [
            ("sigma_h", "σh (Presión):"), ("sigma_a_p", "σa,p (Axial Presión):"),
            ("sigma_a_T", "σa,T (Axial Temp.):"), ("sigma_L_traf", "σL,traf (Long. Tráfico):"),
            ("sigma_a_w", "σa,w (Axial TGD):"), ("sigma_L_total", "σL Total:"),
            ("sigma_h_total", "σh Total:"), ("sigma_VM", "σVM (Von Mises):"),
            ("Ratio_VM_Sy", "Ratio σVM/Sy:")
        ]
        row = row_start
        for key, label_text in results_order:
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=1, padx=5)
            label_var = tk.StringVar(value="--")
            result_label = ttk.Label(parent_frame, textvariable=label_var, font=('Arial', 10, 'bold'), anchor='w')
            result_label.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=1, padx=5)
            self.output_labels[key] = label_var
            row += 1
        self.current_row_left = row


    def on_vehicle_selected(self, event):
        """Actualiza campos de tráfico y redibuja el diagrama simple."""
        selected_vehicle = self.vehicle_combobox.get()
        if selected_vehicle in VEHICLES:
            traffic_data = VEHICLES[selected_vehicle]
            self.input_vars["W_traffic"].set(str(traffic_data["W_traffic"]))
            self.input_vars["A_contacto"].set(str(traffic_data["A_contacto"]))
            if selected_vehicle == "Vehículo Personalizado":
                self.entry_w_traffic.config(state="normal")
                self.entry_a_contacto.config(state="normal")
            else:
                self.entry_w_traffic.config(state="disabled")
                self.entry_a_contacto.config(state="disabled")
        # Redibujar el diagrama simple cada vez que cambia el vehículo (para actualizar W_traffic)
        self.update_conceptual_diagram()


    def on_location_selected(self, event):
        """Carga parámetros sísmicos (sin cambios lógicos)."""
        selected_location = self.location_combobox.get()
        if selected_location in LOCATION_PARAMS:
            location_data = LOCATION_PARAMS[selected_location]
            if "PGV" in location_data:
                self.input_vars["PGV"].set(str(location_data["PGV"]))
            if "C" in location_data:
                self.input_vars["C"].set(str(location_data["C"]))

    def run_analysis(self):
        """Lee inputs, calcula, muestra resultados y actualiza diagrama simple."""
        params = {}
        try:
            # Leer parámetros para cálculo
            param_keys_for_calculation = [key for key in self.input_vars.keys() if key != "A_contacto"]
            for key in param_keys_for_calculation:
                 var = self.input_vars[key]
                 try:
                     params[key] = float(var.get())
                 except ValueError:
                      raise ValueError(f"Valor inválido para '{key}': '{var.get()}'. Ingrese un número.")

            # Realizar cálculo
            results = calculate_pipeline_stress(**params)

            # Mostrar resultados
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

            # Actualizar el diagrama simple (ya no necesita Wc)
            self.update_conceptual_diagram()

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", str(ve))
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error durante el cálculo:\n{e}")
            import traceback
            traceback.print_exc()

    # MÉTODO ACTUALIZADO para usar el diagrama simple
    def update_conceptual_diagram(self):
        """Actualiza el diagrama SIMPLIFICADO de carga de tráfico."""
        try:
            # Obtener parámetros necesarios para el diagrama SIMPLE
            D = float(self.input_vars["D"].get())
            H = float(self.input_vars["H"].get())
            W_traffic = float(self.input_vars["W_traffic"].get())

            # Limpiar contenedor
            if self.canvas_matplotlib:
                self.canvas_matplotlib.get_tk_widget().destroy()
            if self.fig_traffic:
                 plt.close(self.fig_traffic) # Cerrar figura anterior

            # Generar el nuevo gráfico SIMPLE
            self.fig_traffic = plot_simple_traffic_load(D, H, W_traffic)

            # Incrustar gráfico
            self.canvas_matplotlib = FigureCanvasTkAgg(self.fig_traffic, master=self.traffic_plot_container)
            canvas_widget = self.canvas_matplotlib.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas_matplotlib.draw()

        except ValueError:
             # Error de valor numérico en los inputs.
             # No mostrar error aquí, se maneja en run_analysis.
             # Podríamos mostrar un placeholder si quisiéramos.
             pass
        except Exception as e:
             print(f"Error al actualizar el diagrama simple: {e}")
             # Mostrar mensaje de error en el área del gráfico
             for widget in self.traffic_plot_container.winfo_children(): widget.destroy()
             error_label = ttk.Label(self.traffic_plot_container, text=f"Error al generar diagrama:\n{e}", foreground="red", wraplength=300)
             error_label.pack(padx=10, pady=10)


    def load_and_display_seismic_map(self, image_path):
        """Carga y muestra la imagen del mapa sísmico (sin cambios lógicos)."""
        max_width = 350
        max_height = 300
        try:
            if not os.path.exists(image_path):
                 script_dir = os.path.dirname(__file__)
                 alt_path = os.path.join(script_dir, os.path.basename(image_path))
                 if not os.path.exists(alt_path):
                     raise FileNotFoundError(f"No se encontró el archivo en:\n{image_path}\nNi en:\n{alt_path}")
                 else:
                     image_path = alt_path

            img = Image.open(image_path)
            img_width, img_height = img.size
            ratio = min(max_width / img_width, max_height / img_height)
            if ratio < 1.0:
                 new_width = int(img_width * ratio)
                 new_height = int(img_height * ratio)
                 img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.seismic_map_photo = ImageTk.PhotoImage(img)
            self.seismic_map_label.config(image=self.seismic_map_photo)
            self.seismic_map_label.image = self.seismic_map_photo
        except FileNotFoundError as fnf_error:
            error_text = str(fnf_error)
            self.seismic_map_label.config(text=f"Error: Imagen no encontrada.\n{error_text}",
                                           image='', compound=tk.CENTER, wraplength=max_width-20, foreground='red')
            print(f"Error: {error_text}")
        except Exception as e:
            self.seismic_map_label.config(text=f"Error al cargar imagen:\n{e}",
                                          image='', compound=tk.CENTER, wraplength=max_width-20, foreground='red')
            print(f"Error al cargar la imagen {image_path}: {e}")


    def on_frame_configure(self, event):
        """Actualiza la región de desplazamiento del canvas izquierdo."""
        self.root.after_idle(lambda: self.canvas_left.configure(scrollregion=self.canvas_left.bbox("all")))


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    try:
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        initial_width = int(screen_width * 0.7)
        initial_height = int(screen_height * 0.7)
        root.geometry(f"{initial_width}x{initial_height}")
    except Exception:
        root.geometry("900x700")

    app = PipelineStressApp(root)
    root.minsize(850, 650)
    root.mainloop()
