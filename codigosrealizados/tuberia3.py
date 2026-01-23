import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# --- Función de cálculo de esfuerzos ---
def calculate_pipeline_stress(
    D, t, E, nu, alpha_T, Sy,
    H,
    p, delta_T,
    W_traffic, If,
    PGV, C, alpha_seismic
):
    """
    Calcula los esfuerzos en una tubería enterrada bajo cargas combinadas.

    Basado en el modelo analítico simplificado descrito en el documento.

    Args:
        D (float): Diámetro exterior de la tubería (m).
        t (float): Espesor de pared de la tubería (m).
        E (float): Módulo de Young del acero (Pa).
        nu (float): Coeficiente de Poisson del acero (adimensional).
        alpha_T (float): Coeficiente de expansión térmica lineal del acero (°C⁻¹).
        Sy (float): Límite de Fluencia Específico Mínimo (SMYS) del acero (Pa).
        H (float): Profundidad desde la superficie hasta el eje de la tubería (m).
        p (float): Presión interna máxima de operación (Pa).
        delta_T (float): Diferencia de temperatura máxima entre operación e instalación (°C).
        W_traffic (float): Carga por rueda o peso total del vehículo/equipo (N).
        If (float): Factor de impacto (DAF) (adimensional).
        PGV (float): Velocidad Pico del Suelo (m/s).
        C (float): Velocidad aparente de propagación de la onda sísmica (m/s).
        alpha_seismic (float): Factor de relación deformación-velocidad (adimensional).

    Returns:
        dict: Un diccionario que contiene los esfuerzos calculados y el ratio de esfuerzo.
    """

    # --- Ecuaciones del Modelo Analítico ---

    # Esfuerzo Circunferencial por Presión Interna (sigma_h)
    sigma_h = (p * D) / (2 * t)

    # Esfuerzo Axial por Presión Interna (sigma_a_p)
    sigma_a_p = nu * sigma_h

    # Esfuerzo Axial por Cambio de Temperatura (sigma_a_T)
    sigma_a_T = E * alpha_T * delta_T

    # Esfuerzo Longitudinal por Carga de Tráfico (sigma_L_traf)
    Hc = H - D / 2.0 # Profundidad a la corona (m)

    Qd = 0.0
    if Hc > 0:
         # Presión vertical en la corona (Qd) - Usando Boussinesq simplificado para carga puntual
         # Nota: El documento usa W_traffic como carga puntual en el ejemplo
         Qd = (3 * W_traffic) / (2 * np.pi * Hc**2)

    # Carga lineal Wt
    Wt = If * Qd * D # N/m

    # Longitud característica X y factor km (valores de ejemplo del documento)
    X = 2 * Hc # m
    km = 10    # adimensional (intermedio)

    # Módulo de sección aproximado Zpipe
    Zpipe = (np.pi * D**2 * t) / 4.0 # m^3

    sigma_L_traf = 0.0
    if km != 0 and Zpipe != 0:
        # Momento flector por tráfico (simplificado)
        M_traf = (Wt * X**2) / km # Nm
        # Esfuerzo longitudinal por tráfico
        sigma_L_traf = M_traf / Zpipe # Pa


    # Esfuerzo Axial por TGD (sigma_a_w)
    sigma_a_w = 0.0
    if C != 0:
        sigma_a_w = E * alpha_seismic * (PGV / C)


    # Esfuerzo Longitudinal Total (sigma_L)
    sigma_L = sigma_a_p + sigma_a_T + sigma_L_traf + sigma_a_w

    # Esfuerzo Circunferencial Total (sigma_h_total)
    sigma_h_total = sigma_h

    # Esfuerzo Equivalente de Von Mises (sigma_VM)
    sigma_VM = np.sqrt(sigma_L**2 - sigma_L * sigma_h_total + sigma_h_total**2)

    # Ratio de Esfuerzo
    ratio = float('inf') # Evitar división por cero si Sy es cero
    if Sy != 0:
        ratio = sigma_VM / Sy

    results = {
        "sigma_h": sigma_h,
        "sigma_a_p": sigma_a_p,
        "sigma_a_T": sigma_a_T,
        "sigma_L_traf": sigma_L_traf,
        "sigma_a_w": sigma_a_w,
        "sigma_L_total": sigma_L,
        "sigma_h_total": sigma_h_total,
        "sigma_VM": sigma_VM,
        "Ratio_VM_Sy": ratio
    }

    return results

# --- Datos de vehículos pesados (ejemplos para Perú) ---
VEHICLES = {
    "Camión ligero (35.5 kN/rueda)": {"W_traffic": 35500, "A_contacto": 0.1}, # Basado en el ejemplo del documento (HS-20)
    "Camión 2 ejes (carga por eje ~100 kN)": {"W_traffic": 50000, "A_contacto": 0.15},
    "Camión 3 ejes (carga por eje ~150 kN)": {"W_traffic": 50000, "A_contacto": 0.15},
    "Tráiler (carga por eje > 150 kN)": {"W_traffic": 65000, "A_contacto": 0.2},
    "Vehículo Personalizado": {"W_traffic": 35500, "A_contacto": 0.1}
}

# --- Valores por defecto (Caso Miraflores del documento) ---
DEFAULT_PARAMS = {
    "D": 0.61,
    "t": 0.0095,
    "E": 2.07e11,
    "nu": 0.3,
    "alpha_T": 1.2e-5,
    "Sy": 4.48e8,
    "H": 1.5,
    "p": 7e6,
    "delta_T": -15,
    "If": 1.5,
    "PGV": 0.40, # Default Miraflores
    "C": 800,    # Default Miraflores
    "alpha_seismic": 1.0,
    "W_traffic": VEHICLES["Camión ligero (35.5 kN/rueda)"]["W_traffic"],
    "A_contacto": VEHICLES["Camión ligero (35.5 kN/rueda)"]["A_contacto"]
}

# --- Parámetros específicos por ubicación (Tabla 5.5.1 del documento) ---
LOCATION_PARAMS = {
    "Miraflores": {
        "PGV": 0.40,
        "C": 800,
        # Otros parámetros como H, tipo de suelo, etc. podrían añadirse aquí
        # si el modelo los utilizara explícitamente en los cálculos de esfuerzo.
        # En este modelo simplificado, solo PGV y C varían significativamente por distrito.
    },
    "La Molina": {
        "PGV": 0.50,
        "C": 400,
    },
    "Villa El Salvador": {
        "PGV": 0.60,
        "C": 250,
    }
}


# --- Interfaz Gráfica (Tkinter) ---

class PipelineStressApp:
    def __init__(self, root):
        self.root = root
        root.title("Calculadora de Esfuerzos en Tuberías")

        # Configurar estilo
        style = ttk.Style()
        style.configure("TLabel", padding=5, font=('Arial', 10))
        style.configure("TEntry", padding=5, font=('Arial', 10))
        style.configure("TButton", padding=5, font=('Arial', 10, 'bold'))
        style.configure("TCombobox", padding=5, font=('Arial', 10))

        # Frame principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurar expansión
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Diccionario para variables de entrada
        self.input_vars = {}
        self.create_input_fields(main_frame)

        # Sección de resultados
        self.results_label = ttk.Label(main_frame, text="Resultados del Análisis:", font=('Arial', 12, 'bold'))
        self.results_label.grid(row=len(self.input_vars) + 4, column=0, columnspan=3, sticky=tk.W, pady=(15, 5)) # Ajustar fila

        self.output_labels = {}
        self.create_output_labels(main_frame, row_start=len(self.input_vars) + 5) # Ajustar fila

    def create_input_fields(self, parent_frame):
        """Crea las etiquetas y campos de entrada para los parámetros."""
        row = 0

        # Selector de Ubicación
        ttk.Label(parent_frame, text="Ubicación (Cargar Parámetros):").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.location_combobox = ttk.Combobox(parent_frame, values=list(LOCATION_PARAMS.keys()), state="readonly", width=27)
        self.location_combobox.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.location_combobox.set("Miraflores") # Valor por defecto
        self.location_combobox.bind("<<ComboboxSelected>>", self.on_location_selected)
        row += 1


        # Definir parámetros y sus etiquetas/unidades
        parameters_info = {
            "D": ("Diámetro exterior D (m):", "m"),
            "t": ("Espesor de pared t (m):", "m"),
            "E": ("Módulo de Young E (Pa):", "Pa"),
            "nu": ("Coeficiente de Poisson ν:", ""),
            "alpha_T": ("Coeficiente expansión térmica α_T (°C⁻¹):", "°C⁻¹"),
            "Sy": ("Límite de Fluencia Sy (Pa):", "Pa"),
            "H": ("Profundidad al eje H (m):", "m"),
            "p": ("Presión interna p (Pa):", "Pa"),
            "delta_T": ("Cambio de Temperatura ΔT (°C):", "°C"),
            "If": ("Factor de impacto If:", ""),
            "PGV": ("Velocidad Pico del Suelo PGV (m/s):", "m/s"),
            "C": ("Velocidad de Onda C (m/s):", "m/s"),
            "alpha_seismic": ("Factor α (Sísmico):", ""),
            # W_traffic y A_contacto se manejarán por el selector de vehículo
        }

        for key, (label_text, unit) in parameters_info.items():
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            var = tk.StringVar(value=str(DEFAULT_PARAMS[key]))
            entry = ttk.Entry(parent_frame, textvariable=var, width=30)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
            ttk.Label(parent_frame, text=unit).grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
            self.input_vars[key] = var
            row += 1

        # Selector de vehículo
        ttk.Label(parent_frame, text="Vehículo:").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.vehicle_combobox = ttk.Combobox(parent_frame, values=list(VEHICLES.keys()), state="readonly", width=27)
        self.vehicle_combobox.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        self.vehicle_combobox.set("Camión ligero (35.5 kN/rueda)") # Valor por defecto
        self.vehicle_combobox.bind("<<ComboboxSelected>>", self.on_vehicle_selected)
        row += 1

        # Campos para W_traffic y A_contacto (se actualizarán con el selector o se editarán si es personalizado)
        ttk.Label(parent_frame, text="Carga por rueda W_traffic (N):").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.input_vars["W_traffic"] = tk.StringVar(value=str(DEFAULT_PARAMS["W_traffic"]))
        self.entry_w_traffic = ttk.Entry(parent_frame, textvariable=self.input_vars["W_traffic"], width=30)
        self.entry_w_traffic.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        ttk.Label(parent_frame, text="N").grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
        row += 1

        ttk.Label(parent_frame, text="Área de contacto A_contacto (m²):").grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
        self.input_vars["A_contacto"] = tk.StringVar(value=str(DEFAULT_PARAMS["A_contacto"]))
        self.entry_a_contacto = ttk.Entry(parent_frame, textvariable=self.input_vars["A_contacto"], width=30)
        self.entry_a_contacto.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2, padx=5)
        ttk.Label(parent_frame, text="m²").grid(row=row, column=2, sticky=tk.W, pady=2, padx=5)
        row += 1

        # Botón de cálculo
        ttk.Button(parent_frame, text="Calcular Esfuerzos", command=self.run_analysis).grid(row=row, column=0, columnspan=3, pady=10)

        # Asegurar que los campos de tráfico estén deshabilitados inicialmente si no es personalizado
        self.on_vehicle_selected(None) # Llamar al handler para configurar el estado inicial


    def create_output_labels(self, parent_frame, row_start):
        """Crea las etiquetas para mostrar los resultados."""
        results_order = [
            ("sigma_h", "σh (Presión):"),
            ("sigma_a_p", "σa,p (Axial Presión):"),
            ("sigma_a_T", "σa,T (Axial Temperatura):"),
            ("sigma_L_traf", "σL,traf (Longitudinal Tráfico):"),
            ("sigma_a_w", "σa,w (Axial TGD):"),
            ("sigma_L_total", "σL Total:"),
            ("sigma_h_total", "σh Total:"),
            ("sigma_VM", "σVM (Von Mises):"),
            ("Ratio_VM_Sy", "Ratio σVM/Sy:")
        ]

        row = row_start
        for key, label_text in results_order:
            ttk.Label(parent_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2, padx=5)
            label_var = tk.StringVar(value="--")
            ttk.Label(parent_frame, textvariable=label_var, font=('Arial', 10, 'bold')).grid(row=row, column=1, sticky=tk.W, pady=2, padx=5)
            self.output_labels[key] = label_var
            row += 1

    def on_vehicle_selected(self, event):
        """Actualiza los campos de tráfico cuando se selecciona un vehículo."""
        selected_vehicle = self.vehicle_combobox.get()
        if selected_vehicle in VEHICLES:
            traffic_data = VEHICLES[selected_vehicle]
            # Establecer los valores primero
            self.input_vars["W_traffic"].set(str(traffic_data["W_traffic"]))
            self.input_vars["A_contacto"].set(str(traffic_data["A_contacto"]))

            # Habilitar/deshabilitar campos si es "Vehículo Personalizado"
            if selected_vehicle == "Vehículo Personalizado":
                self.entry_w_traffic.config(state="normal")
                self.entry_a_contacto.config(state="normal")
            else:
                # Deshabilitar después de establecer el valor
                self.entry_w_traffic.config(state="disabled")
                self.entry_a_contacto.config(state="disabled")

    def on_location_selected(self, event):
        """Carga los parámetros sísmicos para la ubicación seleccionada."""
        selected_location = self.location_combobox.get()
        if selected_location in LOCATION_PARAMS:
            location_data = LOCATION_PARAMS[selected_location]
            # Actualizar solo los parámetros que varían por ubicación en este modelo simplificado
            if "PGV" in location_data:
                self.input_vars["PGV"].set(str(location_data["PGV"]))
            if "C" in location_data:
                self.input_vars["C"].set(str(location_data["C"]))
            # Si otros parámetros variaran (como H, gamma, etc.), se añadirían aquí.

    def run_analysis(self):
        """Lee los inputs, realiza el cálculo y muestra los resultados."""
        params = {}
        try:
            # Leer todos los valores de los campos de entrada
            # Excluir A_contacto ya que no es un parámetro de la función calculate_pipeline_stress
            param_keys_for_calculation = [key for key in self.input_vars.keys() if key != "A_contacto"]

            for key in param_keys_for_calculation:
                 var = self.input_vars[key]
                 # Convertir a float
                 params[key] = float(var.get())

            # Realizar los cálculos
            results = calculate_pipeline_stress(**params)

            # Mostrar resultados (convertir Pa a MPa y redondear)
            self.output_labels["sigma_h"].set(f"{results['sigma_h']/1e6:.3f} MPa")
            self.output_labels["sigma_a_p"].set(f"{results['sigma_a_p']/1e6:.3f} MPa")
            self.output_labels["sigma_a_T"].set(f"{results['sigma_a_T']/1e6:.3f} MPa")
            self.output_labels["sigma_L_traf"].set(f"{results['sigma_L_traf']/1e6:.3f} MPa")
            self.output_labels["sigma_a_w"].set(f"{results['sigma_a_w']/1e6:.3f} MPa")
            self.output_labels["sigma_L_total"].set(f"{results['sigma_L_total']/1e6:.3f} MPa")
            self.output_labels["sigma_h_total"].set(f"{results['sigma_h_total']/1e6:.3f} MPa")
            self.output_labels["sigma_VM"].set(f"{results['sigma_VM']/1e6:.3f} MPa")
            self.output_labels["Ratio_VM_Sy"].set(f"{results['Ratio_VM_Sy']:.3f}")

        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, ingrese valores numéricos válidos en todos los campos.")
        except Exception as e:
            messagebox.showerror("Error de Cálculo", f"Ocurrió un error durante el cálculo: {e}")


# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineStressApp(root)
    root.mainloop()
