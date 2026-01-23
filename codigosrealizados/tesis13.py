import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from PIL import Image, ImageTk  # Necesario para mostrar imágenes PNG en Tkinter

# --- Configuración ---
# !!! IMPORTANTE: Verifica que 'mapa_sismico.png' sea el nombre correcto de tu archivo !!!
# Si el mapa no está en la misma carpeta que el script, actualiza la ruta completa.
# Ejemplo de ruta absoluta: SEISMIC_MAP_PATH = r'C:\Users\TuUsuario\Documents\mapa_sismico.png'
# Ejemplo de ruta relativa (si está en una subcarpeta 'images'): SEISMIC_MAP_PATH = r'images\mapa_sismico.png'
SEISMIC_MAP_PATH = 'mapa_sismico.png'  # Asume que está en la misma carpeta o ajusta la ruta


# --- Función de cálculo de esfuerzos (sin cambios respecto a tu versión) ---
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
    # gamma_soil se usa aquí, pero es un valor fijo.
    # Para cálculos de deformación, usaremos un gamma_soil_val que puede variar por ubicación.
    gamma_soil_stress_calc = 18000  # N/m^3, valor usado en la función original
    pressure_soil = 0
    if Hc > 0:
        pressure_soil = gamma_soil_stress_calc * Hc

    Qd_traffic = 0.0
    if Hc > 1e-6:  # Evitar división por cero si Hc es muy pequeño
        Qd_traffic = (3 * W_traffic) / (2 * np.pi * Hc ** 2) if Hc ** 2 > 1e-9 else 0

    Qd = pressure_soil + Qd_traffic * If
    Wc_load_per_meter = Qd * D

    # Cálculo de sigma_L_traf
    Wt = If * Qd_traffic * D
    X = 2 * Hc
    km = 10  # Factor según documento, podría ser un input
    # Zpipe = (np.pi * (D**2 - (D-2*t)**2) * D / 8.0) # Aproximación para tubo delgado Z = I / (D/2) ; I ~ pi*D^3*t/8
    # O más preciso para sección anular: I = np.pi/64 * (D**4 - (D-2*t)**4); Zpipe = I / (D/2)
    # Usando la formula original del usuario: Zpipe = (np.pi * D**2 * t) / 4.0 if t > 1e-9 else 0
    # La fórmula Zpipe = (np.pi * D**2 * t) / 4.0 no es estándar para módulo de sección.
    # Usaremos I y A calculados más adelante para consistencia, o una Zpipe más estándar si se revisa.
    # Por ahora, mantenemos la lógica original para esta parte:
    _Zpipe_original = (np.pi * D ** 2 * t) / 4.0 if t > 1e-9 else 0

    sigma_L_traf = 0.0
    if km != 0 and _Zpipe_original > 1e-9 and Wt > 0 and X > 0:  # Asegurar X > 0
        M_traf = (Wt * X ** 2) / km  # Esto parece una simplificación, M podría ser Wt*L_eff / factor
        sigma_L_traf = M_traf / _Zpipe_original

    # --- Otros esfuerzos ---
    sigma_a_w = 0.0
    if C != 0:
        sigma_a_w = E * alpha_seismic * (PGV / C)
    sigma_L = sigma_a_p + sigma_a_T + sigma_L_traf + sigma_a_w
    sigma_h_total = sigma_h

    term_inside_sqrt = sigma_L ** 2 - sigma_L * sigma_h_total + sigma_h_total ** 2
    if term_inside_sqrt < 0:  # Puede ocurrir por errores numéricos si los términos son muy grandes y se cancelan
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


# --- Funciones de cálculo de deformaciones ---
def calculate_pipeline_deformations(
        D, t, E, nu, H, Sy,
        gamma_soil_val,  # Peso unitario del suelo N/m^3
        n_s,  # Coeficiente de reacción de la subrasante (adimensional)
        sigma_L_total,  # Esfuerzo longitudinal total (Pa) (de calc_stress)
        alpha_T,  # Coef. Expansión Térmica (de calc_stress)
        delta_T  # Cambio de Temperatura (de calc_stress)
):
    """
    Calcula varios parámetros de deformación para una tubería enterrada.
    """
    results_def = {}

    # 1. Ovalización por presión externa
    # P_cr = (2*E / (1-nu^2)) * (t/D)^3
    if D > 1e-9 and (1 - nu ** 2) > 1e-9:
        P_cr = (2 * E / (1 - nu ** 2)) * (t / D) ** 3
    else:
        P_cr = 0.0
    results_def["P_cr"] = P_cr

    # sigma_v = gamma_soil_val * H (Presión vertical del suelo al eje de la tubería)
    # H es profundidad al eje.
    sigma_v = gamma_soil_val * H
    results_def["sigma_v"] = sigma_v

    # FS_ext = P_cr / sigma_v
    if sigma_v > 1e-9:  # Evitar división por cero
        FS_ext = P_cr / sigma_v
    else:
        FS_ext = float('inf') if P_cr > 0 else 0.0
    results_def["FS_ext"] = FS_ext

    # 2. Pandeo axial suelo-tubería
    # k_soil_modulus = ns * gamma_soil_val * D / 2  (Rigidez de subrasante, N/m^2 o Pa)
    # El documento lo llama 'k'.
    if D > 1e-9:  # Asegurar D no es cero
        k_soil_modulus = n_s * gamma_soil_val * D / 2.0
    else:
        k_soil_modulus = 0.0
    results_def["k_soil_modulus"] = k_soil_modulus

    # I_pipe = pi * t * (D-t)^3 / 8 (Momento de inercia según documento)
    # Esta fórmula para I de un tubo de pared delgada es una aproximación. Una más común es pi * D_mean^3 * t / 8 o pi/64 * (D_outer^4 - D_inner^4)
    # Usaremos la del documento: pi * t * (D-t)^3 / 8
    if D > t and t > 1e-9:  # D debe ser mayor que t
        I_pipe = np.pi * t * (D - t) ** 3 / 8.0
    else:
        I_pipe = 0.0
    results_def["I_pipe"] = I_pipe

    # A_pipe = pi * t * (D-t) (Área de la sección transversal del tubo)
    # Similar a I, esta es una aproximación. Más común: pi * (D_outer^2 - D_inner^2) / 4 o pi * D_mean * t
    # Usaremos la del documento: pi * t * (D-t)
    if D > t and t > 1e-9:
        A_pipe = np.pi * t * (D - t)
    else:
        A_pipe = 0.0
    results_def["A_pipe"] = A_pipe

    # L_eq = pi * (E * I_pipe / k_soil_modulus)^(1/4) (Longitud de pandeo equivalente)
    L_eq = 0.0  # Inicializar
    if k_soil_modulus > 1e-9 and E * I_pipe >= 0:  # k_soil_modulus no debe ser cero y E*I_pipe no negativo
        term_leq = E * I_pipe / k_soil_modulus
        if term_leq >= 0:  # El término dentro de la raíz cuarta debe ser no negativo
            L_eq = np.pi * (term_leq) ** 0.25
        else:
            L_eq = float('nan')
    else:  # Si k_soil_modulus es cero (o muy pequeño)
        L_eq = float('inf') if E * I_pipe > 0 else (0.0 if E * I_pipe == 0 else float('nan'))
    results_def["L_eq"] = L_eq

    # sigma_cr_long = (pi^2 * E * I_pipe) / (A_pipe * L_eq^2) (Esfuerzo crítico de pandeo longitudinal)
    sigma_cr_long = 0.0  # Inicializar
    if A_pipe > 1e-9 and L_eq > 1e-9 and not np.isinf(L_eq) and not np.isnan(
            L_eq):  # A_pipe y L_eq no deben ser cero/inf/nan
        sigma_cr_long = (np.pi ** 2 * E * I_pipe) / (A_pipe * L_eq ** 2)
    else:  # Si A_pipe o L_eq son problemáticos
        sigma_cr_long = float('inf') if E * I_pipe > 0 and A_pipe > 1e-9 else (0.0 if E * I_pipe == 0 else float('nan'))
    results_def["sigma_cr_long"] = sigma_cr_long

    # 3. Deformación axial total
    # epsilon_total = abs(sigma_L_total)/E + alpha_T * delta_T
    if E > 1e-9:  # E no debe ser cero
        epsilon_mech = abs(sigma_L_total) / E  # Deformación mecánica
        epsilon_therm = alpha_T * delta_T  # Deformación térmica
        epsilon_total = epsilon_mech + epsilon_therm  # Suma algebraica según el signo de delta_T
    else:
        epsilon_total = float('nan')  # O algún indicador de error
    results_def["epsilon_total"] = epsilon_total * 100  # Convertir a porcentaje

    return results_def


# --- Datos de vehículos pesados ACTUALIZADOS (sin cambios) ---
VEHICLES = {
    "AASHTO HS20 (Rueda 71 kN)": {"W_traffic": 71170, "A_contacto": 0.15},
    "AASHTO HL-93 (Rueda 72 kN)": {"W_traffic": 72000, "A_contacto": 0.15},
    "Eje Simple Típico (Rueda 55 kN)": {"W_traffic": 55000, "A_contacto": 0.12},
    "Eje Tándem Típico (Rueda 45 kN)": {"W_traffic": 45000, "A_contacto": 0.12},
    "Eje Trídem Típico (Rueda 42 kN)": {"W_traffic": 42000, "A_contacto": 0.12},
    "Vehículo Personalizado": {"W_traffic": 35500, "A_contacto": 0.1}
}
DEFAULT_VEHICLE_NAME = "AASHTO HS20 (Rueda 71 kN)"
if DEFAULT_VEHICLE_NAME not in VEHICLES:
    DEFAULT_VEHICLE_NAME = list(VEHICLES.keys())[0]

# --- Valores por defecto (actualizados con gamma_soil_val y n_s) ---
DEFAULT_PARAMS = {
    "D": 0.61, "t": 0.0095, "E": 2.07e11, "nu": 0.3, "alpha_T": 1.2e-5,
    "Sy": 4.48e8, "H": 1.5, "p": 7e6, "delta_T": -15, "If": 1.5,
    "PGV": 0.40, "C": 800, "alpha_seismic": 1.0,
    "W_traffic": VEHICLES[DEFAULT_VEHICLE_NAME]["W_traffic"],
    "A_contacto": VEHICLES[DEFAULT_VEHICLE_NAME]["A_contacto"],
    "gamma_soil_val": 18000,  # N/m^3 (Ej: La Molina por defecto)
    "n_s": 3.0  # Adimensional (Ej: para arena densa, según documento)
}
DEFAULT_LOCATION = "Miraflores"

# --- Parámetros específicos por ubicación (actualizados con gamma_soil_val) ---
LOCATION_PARAMS = {  # Valores de PGV, C, gamma_soil_val basados en el documento
    "Miraflores": {"PGV": 0.40, "C": 800, "gamma_soil_val": 19000},  # N/m^3
    "La Molina": {"PGV": 0.50, "C": 400, "gamma_soil_val": 18000},  # N/m^3
    "Villa El Salvador": {"PGV": 0.60, "C": 250, "gamma_soil_val": 17000}  # N/m^3
}
# Actualizar gamma_soil_val en DEFAULT_PARAMS si la DEFAULT_LOCATION tiene uno específico
if DEFAULT_LOCATION in LOCATION_PARAMS and "gamma_soil_val" in LOCATION_PARAMS[DEFAULT_LOCATION]:
    DEFAULT_PARAMS["gamma_soil_val"] = LOCATION_PARAMS[DEFAULT_LOCATION]["gamma_soil_val"]


# --- Función para generar el diagrama de carga de tráfico (sin cambios) ---
def plot_simple_traffic_load(D, H, W_traffic):
    fig, ax = plt.subplots(figsize=(5.5, 5))  # Ajustar tamaño si es necesario
    plot_width = max(D * 4, 3.0)
    plot_depth = max(H + D * 1.5, 2.5)
    ax.set_xlim(-plot_width / 2, plot_width / 2)
    ax.set_ylim(plot_depth, -0.5)  # Y invertido: 0 arriba, aumenta hacia abajo

    pipe_center_x = 0
    pipe_center_y = H
    soil_color = '#CDBA96'
    soil_edge_color = '#8B7D6B'
    pipe_color = '#A4B8C4'
    pipe_edge_color = '#556270'
    load_line_color = '#FF7F50'  # Naranja coral
    load_arrow_color = '#CD5C5C'  # Rojo indio

    # Suelo
    ax.add_patch(
        patches.Rectangle((-plot_width / 2, 0), plot_width, plot_depth, facecolor=soil_color, edgecolor=soil_edge_color,
                          linewidth=1, alpha=0.8))
    # Línea de superficie
    ax.plot([-plot_width / 2, plot_width / 2], [0, 0], color=soil_edge_color, linewidth=1.5)

    # Tubería
    pipe_outer = patches.Circle((pipe_center_x, pipe_center_y), D / 2, facecolor=pipe_color, edgecolor=pipe_edge_color,
                                linewidth=1.5, zorder=5)
    ax.add_patch(pipe_outer)
    ax.text(pipe_center_x, pipe_center_y, f'Tubería (D={D:.2f}m)', ha='center', va='center', color='white', fontsize=8,
            fontweight='bold', zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc=pipe_edge_color, ec="none", alpha=0.7))

    # Rueda (simplificada en la superficie)
    tire_radius = D * 0.25  # Radio de la rueda proporcional al diámetro de la tubería
    tire_x = 0
    tire_y = 0  # Rueda en la superficie
    ax.add_patch(
        patches.Circle((tire_x, tire_y), tire_radius, facecolor='#696969', edgecolor='black', linewidth=1, zorder=8))

    # Flecha de Carga (W_traffic)
    arrow_start_y = -0.3  # Un poco por encima de la superficie para la flecha
    arrow_end_y = tire_y - tire_radius  # Flecha apunta al borde superior de la rueda
    arrow_x = tire_x
    arrow_head_width = plot_width * 0.05  # Ancho de la cabeza de la flecha
    arrow_head_length = 0.15  # Largo de la cabeza de la flecha

    if W_traffic > 0:
        ax.arrow(arrow_x, arrow_start_y, 0, arrow_end_y - arrow_start_y,  # dx=0, dy hacia abajo
                 head_width=arrow_head_width, head_length=arrow_head_length, fc=load_arrow_color, ec=load_arrow_color,
                 length_includes_head=True, zorder=9)
        # Etiqueta de la carga
        ax.text(arrow_x + plot_width * 0.1, arrow_start_y + (arrow_end_y - arrow_start_y) / 2,
                f'Carga={W_traffic / 1000:.1f} kN',  # Centrar etiqueta en la flecha
                ha='left', va='center', color=load_arrow_color, fontsize=9, fontweight='bold')

    # Líneas de distribución de carga (simplificadas a 45 grados)
    dispersion_angle_rad = np.deg2rad(45)  # np.arctan(1/1) es 45 grados
    start_x_left = tire_x - tire_radius  # Borde izquierdo de la rueda
    start_x_right = tire_x + tire_radius  # Borde derecho de la rueda
    start_y = tire_y  # Desde la superficie

    # Corona de la tubería
    pipe_crown_y = H - D / 2
    if pipe_crown_y > start_y:  # Solo dibujar si la tubería está debajo de la rueda
        dist_y = pipe_crown_y - start_y
        dist_x = dist_y / np.tan(dispersion_angle_rad) if np.tan(dispersion_angle_rad) > 1e-9 else dist_y

        end_x_left = start_x_left - dist_x
        end_x_right = start_x_right + dist_x
        ax.plot([start_x_left, end_x_left], [start_y, pipe_crown_y], color=load_line_color, linestyle='--',
                linewidth=1.5, zorder=4)
        ax.plot([start_x_right, end_x_right], [start_y, pipe_crown_y], color=load_line_color, linestyle='--',
                linewidth=1.5, zorder=4)

    # Etiqueta Profundidad H
    label_h_y = H / 2  # Posición Y de la etiqueta de profundidad
    if label_h_y < 0: label_h_y = 0.1
    ax.text(-plot_width / 2 * 0.85, label_h_y, f'H={H:.2f}m', ha='center', va='center', fontsize=9, color='black',
            rotation=90)  # Un poco más a la izquierda

    ax.set_title('Diagrama Conceptual de Carga de Tráfico', fontsize=11, pad=10)
    ax.set_xlabel('Distancia Horizontal (m)', fontsize=10)
    ax.set_ylabel('Profundidad (m)', fontsize=10)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.7)
    plt.tight_layout(pad=0.5)  # Ajustar layout para evitar superposiciones
    return fig


# --- Interfaz Gráfica (Tkinter) ---
class PipelineStressApp:
    def __init__(self, root):
        self.root = root
        root.title("Calculadora de Esfuerzos y Deformaciones en Tuberías")
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            print("Tema 'clam' no disponible, usando tema por defecto.")
        style.configure("TLabel", padding=5, font=('Arial', 10))
        style.configure("TEntry", padding=5, font=('Arial', 10))
        style.configure("TButton", padding=5, font=('Arial', 10, 'bold'))
        style.configure("Reset.TButton", foreground='black', background='#ffc107')  # Amarillo para reset
        style.map("Reset.TButton", background=[('active', '#e0a800')])
        style.configure("ClearHist.TButton", foreground='white', background='#6c757d')  # Gris para limpiar historial
        style.map("ClearHist.TButton", background=[('active', '#5a6268')])
        style.configure("TCombobox", padding=5, font=('Arial', 10))
        style.configure("TFrame", background='#f0f0f0')  # Fondo general
        style.configure("Left.TFrame", background='#e8e8e8')
        style.configure("Right.TFrame", background='#ffffff')
        style.configure("Results.TFrame", background='#ffffff')  # Fondo para el frame de resultados de esfuerzos
        style.configure("DeformationResults.TFrame",
                        background='#ffffff')  # Fondo para el frame de resultados de deformaciones
        style.configure("Hist.TFrame", background='#ffffff')  # Fondo para el frame de historial
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Arial', 9))
        style.map("Treeview", background=[('selected', '#007bff')], foreground=[('selected', 'white')])

        self.main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_frame_outer = ttk.Frame(self.main_paned_window, width=450, style="Left.TFrame")
        self.left_frame_outer.pack(fill=tk.BOTH, expand=False)
        self.main_paned_window.add(self.left_frame_outer, weight=0)

        self.canvas_left = tk.Canvas(self.left_frame_outer, borderwidth=0, background="#e8e8e8")
        self.scrollbar_v_left = ttk.Scrollbar(self.left_frame_outer, orient="vertical", command=self.canvas_left.yview)
        self.left_frame_inner = ttk.Frame(self.canvas_left, padding="10 10 10 20", style="Left.TFrame")

        self.canvas_left.configure(yscrollcommand=self.scrollbar_v_left.set)
        self.scrollbar_v_left.pack(side="right", fill="y")
        self.canvas_left.pack(side="left", fill="both", expand=True)
        self.canvas_left_window = self.canvas_left.create_window((0, 0), window=self.left_frame_inner, anchor="nw")

        self.left_frame_inner.bind("<Configure>", self.on_frame_configure_left)
        self.canvas_left.bind("<Configure>", self.on_canvas_configure_left)

        self.input_vars = {}
        self.current_row_left = 0
        self.create_input_fields(self.left_frame_inner)

        button_frame = ttk.Frame(self.left_frame_inner, style="Left.TFrame")
        button_frame.grid(row=self.current_row_left, column=0, columnspan=3, pady=20, sticky=tk.EW)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="Calcular", command=self.run_analysis).grid(
            row=0, column=0, padx=5, sticky=tk.E)
        ttk.Button(button_frame, text="Restaurar Valores", command=self.reset_to_defaults, style="Reset.TButton").grid(
            row=0, column=1, padx=5, sticky=tk.W)
        self.current_row_left += 1

        # --- Panel Derecho (Visualizaciones, Resultados, Historial) ---
        self.right_frame = ttk.Frame(self.main_paned_window, style="Right.TFrame")
        self.right_frame.pack(fill=tk.BOTH, expand=True)
        self.main_paned_window.add(self.right_frame, weight=1)

        self.right_frame.rowconfigure(0, weight=0)
        self.right_frame.rowconfigure(1, weight=1)
        self.right_frame.rowconfigure(2, weight=1)
        self.right_frame.columnconfigure(0, weight=1)

        top_right_frame = ttk.Frame(self.right_frame, style="Right.TFrame", padding="5")
        top_right_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        top_right_frame.columnconfigure(0, weight=1)
        top_right_frame.columnconfigure(1, weight=1)
        top_right_frame.columnconfigure(2, weight=1)
        top_right_frame.rowconfigure(1, weight=1)

        ttk.Label(top_right_frame, text="Mapa Sísmico (Referencial)", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=(tk.W, tk.E), pady=(5, 2), padx=5)
        self.seismic_map_label = ttk.Label(top_right_frame, background='white', anchor='center')
        self.seismic_map_label.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=2)
        self.load_and_display_seismic_map(SEISMIC_MAP_PATH)

        results_frame_stress = ttk.Frame(top_right_frame, style="Results.TFrame", padding="10")
        results_frame_stress.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=2)
        results_frame_stress.columnconfigure(0, weight=0)
        results_frame_stress.columnconfigure(1, weight=1)

        ttk.Label(top_right_frame, text="Resultados del Análisis (Esfuerzos)", font=('Arial', 10, 'bold')).grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=(5, 2), padx=5)
        self.output_labels_stress = {}
        self.create_stress_output_labels(results_frame_stress)

        results_frame_deformation = ttk.Frame(top_right_frame, style="DeformationResults.TFrame", padding="10")
        results_frame_deformation.grid(row=1, column=2, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=2)
        results_frame_deformation.columnconfigure(0, weight=0)
        results_frame_deformation.columnconfigure(1, weight=1)

        ttk.Label(top_right_frame, text="Resultados del Análisis (Deformaciones)", font=('Arial', 10, 'bold')).grid(
            row=0, column=2, sticky=(tk.W, tk.E), pady=(5, 2), padx=5)
        self.output_labels_deformation = {}
        self.create_deformation_output_labels(results_frame_deformation)

        ttk.Label(self.right_frame, text="Diagrama Conceptual de Carga", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 2), padx=10)
        self.traffic_plot_container = ttk.Frame(self.right_frame, style="Right.TFrame")
        self.traffic_plot_container.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=5)
        self.fig_traffic = None
        self.canvas_matplotlib = None

        hist_frame = ttk.Frame(self.right_frame, style="Hist.TFrame", padding="5")
        hist_frame.grid(row=2, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=(10, 5))
        hist_frame.columnconfigure(0, weight=1)
        hist_frame.columnconfigure(1, weight=0)
        hist_frame.rowconfigure(1, weight=1)

        hist_label_frame = ttk.Frame(hist_frame, style="Hist.TFrame")
        hist_label_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(hist_label_frame, text="Historial de Cálculos (Esfuerzos)", font=('Arial', 10, 'bold')).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(hist_label_frame, text="Limpiar", command=self.clear_history, style="ClearHist.TButton",
                   width=8).pack(side=tk.LEFT, padx=5)

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
        self.history_tree.heading("ratio", text="Ratio σVM/Sy", anchor=tk.CENTER)

        self.reset_to_defaults()

    def on_frame_configure_left(self, event=None):
        self.canvas_left.configure(scrollregion=self.canvas_left.bbox("all"))

    def on_canvas_configure_left(self, event=None):
        canvas_width = event.width
        self.canvas_left.itemconfig(self.canvas_left_window, width=canvas_width)

    def create_input_fields(self, parent_frame):
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.columnconfigure(1, weight=2)
        parent_frame.columnconfigure(2, weight=0)
        row = 0

        ttk.Label(parent_frame, text="Ubicación (Cargar Parámetros):").grid(row=row, column=0, sticky=tk.W, pady=2,
                                                                            padx=5)
        self.location_combobox = ttk.Combobox(parent_frame, values=list(LOCATION_PARAMS.keys()), state="readonly",
                                              width=27)
        self.location_combobox.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.location_combobox.set(DEFAULT_LOCATION)
        self.location_combobox.bind("<<ComboboxSelected>>", self.on_location_selected)
        row += 1

        ttk.Label(parent_frame, text="--- Tubería y Material ---", font=('Arial', 10, 'italic')).grid(row=row, column=0,
                                                                                                      columnspan=3,
                                                                                                      sticky=tk.W,
                                                                                                      pady=(8, 2),
                                                                                                      padx=5)
        row += 1
        parameters_pipe = {
            "D": ("Diámetro exterior D:", "m"), "t": ("Espesor de pared t:", "m"),
            "E": ("Módulo de Young E:", "Pa"), "nu": ("Coeficiente de Poisson ν:", ""),
            "alpha_T": ("Coef. Expansión Térmica α_T:", "°C⁻¹"), "Sy": ("Límite de Fluencia Sy:", "Pa")
        }
        for key, (label_text, unit) in parameters_pipe.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
            self.input_vars[key] = var
            entry = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=2)
            row += 1

        ttk.Label(parent_frame, text="--- Cargas y Condiciones (Esfuerzos) ---", font=('Arial', 10, 'italic')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 2), padx=5)
        row += 1
        parameters_load_stress = {
            "H": ("Profundidad al eje H:", "m"), "p": ("Presión interna p:", "Pa"),
            "delta_T": ("Cambio de Temperatura ΔT:", "°C"), "If": ("Factor de impacto If (tráfico):", "")
        }
        for key, (label_text, unit) in parameters_load_stress.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
            self.input_vars[key] = var
            entry = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=2)
            row += 1

        ttk.Label(parent_frame, text="Vehículo de Tráfico:").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.vehicle_combobox = ttk.Combobox(parent_frame, values=list(VEHICLES.keys()), state="readonly", width=27)
        self.vehicle_combobox.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.vehicle_combobox.set(DEFAULT_VEHICLE_NAME)
        self.vehicle_combobox.bind("<<ComboboxSelected>>", self.on_vehicle_selected)
        row += 1

        parameters_traffic = {
            "W_traffic": ("Carga por rueda W_traffic:", "N"),
            "A_contacto": ("Área de contacto A_contacto:", "m²")
        }
        for key, (label_text, unit) in parameters_traffic.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
            self.input_vars[key] = var
            entry_widget = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry_widget.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=2)
            if key == "W_traffic": self.entry_w_traffic = entry_widget
            if key == "A_contacto": self.entry_a_contacto = entry_widget
            row += 1

        ttk.Label(parent_frame, text="--- Parámetros Sísmicos (TGD) ---", font=('Arial', 10, 'italic')).grid(row=row,
                                                                                                             column=0,
                                                                                                             columnspan=3,
                                                                                                             sticky=tk.W,
                                                                                                             pady=(8,
                                                                                                                   2),
                                                                                                             padx=5)
        row += 1
        parameters_seismic = {
            "PGV": ("Velocidad Pico del Suelo PGV:", "m/s"),
            "C": ("Velocidad de Onda C:", "m/s"),
            "alpha_seismic": ("Factor α (Sísmico TGD):", "")
        }
        for key, (label_text, unit) in parameters_seismic.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
            self.input_vars[key] = var
            entry = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=2)
            row += 1

        ttk.Label(parent_frame, text="--- Parámetros de Suelo (Deformación) ---", font=('Arial', 10, 'italic')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 2), padx=5)
        row += 1
        parameters_deformation_soil = {
            "gamma_soil_val": ("Peso Unitario Suelo γ_suelo:", "N/m³"),
            "n_s": ("Coef. Reacción Subrasante n_s:", "")
        }
        for key, (label_text, unit) in parameters_deformation_soil.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS.get(key, '')))
            self.input_vars[key] = var
            entry = ttk.Entry(parent_frame, textvariable=var, width=25)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            if key == "gamma_soil_val": self.entry_gamma_soil_val = entry
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=2)
            row += 1

        self.current_row_left = row

    def create_stress_output_labels(self, parent_frame):
        results_order = [
            ("sigma_h", "σh (Presión):"), ("sigma_a_p", "σa,p (Axial Presión):"),
            ("sigma_a_T", "σa,T (Axial Temp.):"), ("sigma_L_traf", "σL,traf (Long. Tráfico):"),
            ("sigma_a_w", "σa,w (Axial TGD):"), ("sigma_L_total", "σL Total:"),
            ("sigma_h_total", "σh Total:"), ("sigma_VM", "σVM (Von Mises):"),
            ("Ratio_VM_Sy", "Ratio σVM/Sy:")
        ]
        row = 0
        for key, label_text in results_order:
            ttk.Label(parent_frame, text=label_text, anchor='w').grid(row=row, column=0, sticky=tk.W, pady=1, padx=5)
            label_var = tk.StringVar(value="--")
            # CORRECCIÓN: Removido style="Results.TFrame" de la etiqueta de valor
            result_label = ttk.Label(parent_frame, textvariable=label_var, font=('Arial', 10, 'bold'), anchor='e')
            result_label.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=1, padx=5)
            self.output_labels_stress[key] = label_var
            row += 1

    def create_deformation_output_labels(self, parent_frame):
        results_order_def = [
            ("P_cr", "P_cr (Ovalización):"), ("sigma_v", "σv (Presión Suelo):"),
            ("FS_ext", "FS_ext (Ovalización):"),
            ("k_soil_modulus", "k (Mód. Subrasante):"),
            ("I_pipe", "I (Inercia Tubo):"), ("A_pipe", "A (Área Tubo):"),
            ("L_eq", "L_eq (Pandeo):"), ("sigma_cr_long", "σ_cr,long (Pandeo):"),
            ("epsilon_total", "ε_total (Axial):")
        ]
        units_def = {
            "P_cr": "MPa", "sigma_v": "MPa", "FS_ext": "",
            "k_soil_modulus": "N/m²", "I_pipe": "m⁴", "A_pipe": "m²",
            "L_eq": "m", "sigma_cr_long": "MPa", "epsilon_total": "%"
        }
        row = 0
        for key, label_text in results_order_def:
            ttk.Label(parent_frame, text=label_text, anchor='w').grid(row=row, column=0, sticky=tk.W, pady=1, padx=5)
            label_var = tk.StringVar(value="--")  # Para el valor numérico crudo
            unit_text = units_def.get(key, "")
            full_text_var = tk.StringVar(value=f"-- {unit_text}")  # Para valor con unidad

            self.output_labels_deformation[key] = label_var
            self.output_labels_deformation[key + "_display"] = full_text_var

            # CORRECCIÓN: Removido style="DeformationResults.TFrame" de la etiqueta de valor
            result_label = ttk.Label(parent_frame, textvariable=full_text_var, font=('Arial', 10, 'bold'), anchor='e')
            result_label.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=1, padx=5)
            row += 1

    def on_vehicle_selected(self, event=None):
        selected_vehicle = self.vehicle_combobox.get()
        if selected_vehicle in VEHICLES:
            traffic_data = VEHICLES[selected_vehicle]
            if "W_traffic" in self.input_vars: self.input_vars["W_traffic"].set(str(traffic_data["W_traffic"]))
            if "A_contacto" in self.input_vars: self.input_vars["A_contacto"].set(str(traffic_data["A_contacto"]))

            is_custom = (selected_vehicle == "Vehículo Personalizado")
            if hasattr(self, 'entry_w_traffic'): self.entry_w_traffic.config(
                state="normal" if is_custom else "disabled")
            if hasattr(self, 'entry_a_contacto'): self.entry_a_contacto.config(
                state="normal" if is_custom else "disabled")
        self.update_conceptual_diagram()

    def on_location_selected(self, event=None):
        selected_location = self.location_combobox.get()
        if selected_location in LOCATION_PARAMS:
            location_data = LOCATION_PARAMS[selected_location]
            for key, value in location_data.items():
                if key in self.input_vars:
                    self.input_vars[key].set(str(value))
                # El caso de 'gamma_soil_val' ya está cubierto si está en input_vars.
                # Si se quisiera manejar por separado si no está en input_vars (aunque debería estarlo):
                # elif key == "gamma_soil_val" and hasattr(self, 'entry_gamma_soil_val'):
                #     self.entry_gamma_soil_val.delete(0, tk.END)
                #     self.entry_gamma_soil_val.insert(0, str(value))
                else:
                    print(
                        f"Advertencia: El parámetro '{key}' de la ubicación '{selected_location}' no tiene campo de entrada asociado directo en input_vars.")
            # No es necesario llamar a update_conceptual_diagram aquí a menos que D o H cambien con la ubicación.
            # self.update_conceptual_diagram()

    def run_analysis(self):
        params_stress = {}
        params_deformation_inputs = {}

        try:
            all_input_params = {}
            for key, var_tk in self.input_vars.items():
                try:
                    all_input_params[key] = float(var_tk.get())
                except ValueError:
                    raise ValueError(f"Valor inválido para '{key}': '{var_tk.get()}'. Ingrese un número.")

            stress_param_keys = ["D", "t", "E", "nu", "alpha_T", "Sy", "H", "p", "delta_T", "W_traffic", "If", "PGV",
                                 "C", "alpha_seismic"]
            for key in stress_param_keys:
                if key in all_input_params:
                    params_stress[key] = all_input_params[key]
                else:
                    raise ValueError(f"Falta el parámetro de esfuerzo requerido: {key}")

            results_s = calculate_pipeline_stress(**params_stress)

            self.output_labels_stress["sigma_h"].set(f"{results_s['sigma_h'] / 1e6:.3f} MPa")
            self.output_labels_stress["sigma_a_p"].set(f"{results_s['sigma_a_p'] / 1e6:.3f} MPa")
            self.output_labels_stress["sigma_a_T"].set(f"{results_s['sigma_a_T'] / 1e6:.3f} MPa")
            sigma_L_traf_val = results_s['sigma_L_traf']
            self.output_labels_stress["sigma_L_traf"].set(
                f"{sigma_L_traf_val / 1e6:.3f} MPa" if abs(sigma_L_traf_val) > 1e-9 else "0.000 MPa")
            self.output_labels_stress["sigma_a_w"].set(f"{results_s['sigma_a_w'] / 1e6:.3f} MPa")
            self.output_labels_stress["sigma_L_total"].set(f"{results_s['sigma_L_total'] / 1e6:.3f} MPa")
            self.output_labels_stress["sigma_h_total"].set(f"{results_s['sigma_h_total'] / 1e6:.3f} MPa")
            self.output_labels_stress["sigma_VM"].set(f"{results_s['sigma_VM'] / 1e6:.3f} MPa")
            self.output_labels_stress["Ratio_VM_Sy"].set(f"{results_s['Ratio_VM_Sy']:.3f}")

            hist_values = (
                f"{results_s['sigma_L_total'] / 1e6:.3f}", f"{results_s['sigma_h_total'] / 1e6:.3f}",
                f"{results_s['sigma_VM'] / 1e6:.3f}", f"{results_s['Ratio_VM_Sy']:.3f}"
            )
            self.history_tree.insert("", 0, values=hist_values)

            deformation_param_keys = ["D", "t", "E", "nu", "H", "Sy", "gamma_soil_val", "n_s", "alpha_T", "delta_T"]
            for key in deformation_param_keys:
                if key in all_input_params:
                    params_deformation_inputs[key] = all_input_params[key]
                else:
                    raise ValueError(f"Falta el parámetro de deformación requerido: {key}")

            params_deformation_inputs["sigma_L_total"] = results_s['sigma_L_total']
            results_d = calculate_pipeline_deformations(**params_deformation_inputs)

            units_def_map = {
                "P_cr": "MPa", "sigma_v": "MPa", "FS_ext": "",
                "k_soil_modulus": "N/m²", "I_pipe": "m⁴", "A_pipe": "m²",
                "L_eq": "m", "sigma_cr_long": "MPa", "epsilon_total": "%"
            }

            for key, val_raw in results_d.items():
                unit = units_def_map.get(key, "")
                val_to_display = val_raw
                format_str = "{:.3f}"  # Formato por defecto

                if key == "P_cr":
                    val_to_display = val_raw / 1e6 if val_raw is not None and not np.isnan(val_raw) else val_raw
                    format_str = "{:.3e}"
                elif key == "sigma_v":
                    val_to_display = val_raw / 1e6 if val_raw is not None and not np.isnan(val_raw) else val_raw
                elif key == "FS_ext":
                    format_str = "{:.2f}"
                elif key == "k_soil_modulus" or key == "I_pipe" or key == "A_pipe":
                    format_str = "{:.2e}"
                elif key == "L_eq":
                    format_str = "{:.2f}"
                elif key == "sigma_cr_long":
                    val_to_display = val_raw / 1e6 if val_raw is not None and not np.isnan(val_raw) else val_raw
                # epsilon_total ya está en % y usa el formato por defecto .3f

                if val_to_display is not None and not np.isnan(val_to_display) and not np.isinf(val_to_display):
                    self.output_labels_deformation[key].set(format_str.format(val_to_display))
                    self.output_labels_deformation[key + "_display"].set(f"{format_str.format(val_to_display)} {unit}")
                else:
                    self.output_labels_deformation[key].set("--")
                    self.output_labels_deformation[key + "_display"].set(f"-- {unit}")

            self.update_conceptual_diagram()

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", str(ve))
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error durante el cálculo:\n{e}")
            import traceback
            traceback.print_exc()

    def update_conceptual_diagram(self):
        try:
            D_str = self.input_vars.get("D", tk.StringVar(value=str(DEFAULT_PARAMS["D"]))).get()
            H_str = self.input_vars.get("H", tk.StringVar(value=str(DEFAULT_PARAMS["H"]))).get()
            W_traffic_str = self.input_vars.get("W_traffic", tk.StringVar(value=str(DEFAULT_PARAMS["W_traffic"]))).get()

            D = float(D_str)
            H = float(H_str)
            W_traffic = float(W_traffic_str)

            if self.canvas_matplotlib:
                self.canvas_matplotlib.get_tk_widget().destroy()
            if self.fig_traffic:
                plt.close(self.fig_traffic)

            self.fig_traffic = plot_simple_traffic_load(D, H, W_traffic)

            self.canvas_matplotlib = FigureCanvasTkAgg(self.fig_traffic, master=self.traffic_plot_container)
            canvas_widget = self.canvas_matplotlib.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas_matplotlib.draw()

        except ValueError:
            print("Advertencia: No se pudo actualizar el diagrama debido a valores no numéricos o faltantes.")
            for widget in self.traffic_plot_container.winfo_children():
                widget.destroy()
            placeholder = ttk.Label(self.traffic_plot_container, text="(Esperando datos válidos para el diagrama)",
                                    foreground="gray")
            placeholder.pack(expand=True, padx=10, pady=10)
        except Exception as e:
            print(f"Error al actualizar el diagrama simple: {e}")
            for widget in self.traffic_plot_container.winfo_children():
                widget.destroy()
            error_label = ttk.Label(self.traffic_plot_container, text=f"Error al generar diagrama:\n{e}",
                                    foreground="red", wraplength=300)
            error_label.pack(padx=10, pady=10)

    def reset_to_defaults(self):
        print("Restaurando valores por defecto...")
        for key, default_value in DEFAULT_PARAMS.items():
            if key in self.input_vars:
                self.input_vars[key].set(str(default_value))

        self.location_combobox.set(DEFAULT_LOCATION)
        self.on_location_selected()  # Actualiza gamma_soil_val y otros params. de ubicación

        self.vehicle_combobox.set(DEFAULT_VEHICLE_NAME)
        self.on_vehicle_selected()  # Actualiza W_traffic, A_contacto y el diagrama

        for key in self.output_labels_stress:
            self.output_labels_stress[key].set("--")

        units_def_map = {
            "P_cr": "MPa", "sigma_v": "MPa", "FS_ext": "",
            "k_soil_modulus": "N/m²", "I_pipe": "m⁴", "A_pipe": "m²",
            "L_eq": "m", "sigma_cr_long": "MPa", "epsilon_total": "%"
        }
        for key_actual in units_def_map.keys():  # Iterar sobre las llaves base
            if key_actual in self.output_labels_deformation:
                self.output_labels_deformation[key_actual].set("--")
            if key_actual + "_display" in self.output_labels_deformation:
                unit = units_def_map.get(key_actual, "")
                self.output_labels_deformation[key_actual + "_display"].set(f"-- {unit}")

        self.clear_history()
        # self.update_conceptual_diagram() # Ya se llama en on_vehicle_selected
        print("Valores restaurados.")

    def clear_history(self):
        print("Limpiando historial...")
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        print("Historial limpiado.")

    def load_and_display_seismic_map(self, image_path_param):
        max_width = 320
        max_height = 280

        paths_to_try = [image_path_param]
        try:
            # Intentar con la ruta relativa al script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            paths_to_try.append(os.path.join(script_dir, os.path.basename(image_path_param)))
        except NameError:  # __file__ no está definido (ej. en algunos intérpretes interactivos)
            pass
        # Intentar con la ruta relativa al directorio de trabajo actual
        paths_to_try.append(os.path.join(os.getcwd(), os.path.basename(image_path_param)))

        found_path = None
        for p in paths_to_try:
            if os.path.exists(p):
                found_path = p
                break

        if not found_path:
            error_text = f"Imagen no encontrada.\nRutas intentadas:\n- {paths_to_try[0]}"
            if len(paths_to_try) > 1: error_text += f"\n- {paths_to_try[1]}"
            if len(paths_to_try) > 2: error_text += f"\n- {paths_to_try[2]}"
            self.seismic_map_label.config(text=error_text, image='', compound=tk.CENTER, wraplength=max_width - 20,
                                          foreground='red', font=('Arial', 8))
            print(f"Error: {error_text}")
            return

        try:
            img = Image.open(found_path)
            img_width, img_height = img.size

            ratio = min(max_width / img_width, max_height / img_height) if img_width > 0 and img_height > 0 else 1.0
            if ratio < 1.0:
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.seismic_map_photo = ImageTk.PhotoImage(img)
            self.seismic_map_label.config(image=self.seismic_map_photo, text="")
            self.seismic_map_label.image = self.seismic_map_photo
        except Exception as e:
            self.seismic_map_label.config(text=f"Error al cargar imagen:\n{e}", image='', compound=tk.CENTER,
                                          wraplength=max_width - 20, foreground='red')
            print(f"Error al cargar la imagen {found_path}: {e}")


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    try:
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        initial_width = int(screen_width * 0.85)
        initial_height = int(screen_height * 0.80)
        # Asegurar que el tamaño no sea menor que el minsize
        initial_width = max(initial_width, 1000)
        initial_height = max(initial_height, 750)
        root.geometry(f"{initial_width}x{initial_height}")
    except Exception:
        root.geometry("1200x800")

    app = PipelineStressApp(root)
    root.minsize(1000, 750)
    root.mainloop()
