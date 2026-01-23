import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------------------------------------------------------
# Arquitectura y Diseño
# -----------------------------------------------------------------------------
# ACTUANDO COMO CONSULTOR TECNOLÓGICO:
# He elegido `tkinter` para la GUI por ser la librería estándar de Python,
# eliminando dependencias externas. Para la visualización, `matplotlib` es
# el estándar de facto y se integra perfectamente con `tkinter`.
# La aplicación está diseñada como una clase única (`VibrationApp`) para encapsular
# toda la lógica y el estado de la GUI, lo que mejora la mantenibilidad.
# Se ha implementado un manejo de errores robusto con bloques try-except para
# prevenir que la aplicación falle por entradas de usuario inválidas.
# -----------------------------------------------------------------------------

class VibrationApp:
    def __init__(self, root):
        """
        Constructor de la aplicación. Inicializa la GUI y sus componentes.
        """
        self.root = root
        self.root.title("Simulador de Vibración Libre Amortiguada")
        self.root.geometry("1000x800")

        # --- Estilo ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10, "bold"))
        style.configure("TEntry", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))

        # --- Frames para organizar la GUI ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Parámetros de Entrada", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Creación de la figura y el eje para Matplotlib ---
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Diccionario para almacenar las variables de entrada ---
        self.entries = {}

        # --- MEJORA: Valores por defecto para el nuevo problema (Figura 01) ---
        ttk.Label(input_frame, text="Análisis de Amortiguamiento (Gráfica)", style="Header.TLabel").grid(row=0,
                                                                                                         column=0,
                                                                                                         columnspan=2,
                                                                                                         pady=10,
                                                                                                         sticky="w")
        self.add_entry(input_frame, "Amplitud Pico ui (cm)", 1, "6.686352")
        self.add_entry(input_frame, "Amplitud Pico ui+j (cm)", 2, "4.956063")
        self.add_entry(input_frame, "Número de ciclos j", 3, "1")
        self.add_entry(input_frame, "Período Amortiguado Td (s)", 4, "1.0")

        # --- Entradas para el cálculo de la Respuesta ---
        ttk.Label(input_frame, text="Predicción de Respuesta", style="Header.TLabel").grid(row=5, column=0,
                                                                                           columnspan=2, pady=10,
                                                                                           sticky="w")

        # El valor v0 ≈ 44.1 se calcula para que el primer pico sea ~6.68 cm
        self.add_entry(input_frame, "Desplazamiento Inicial u(0) (cm)", 6, "0.0")
        self.add_entry(input_frame, "Velocidad Inicial v(0) (cm/s)", 7, "44.1")
        self.add_entry(input_frame, "Tiempo a evaluar t (s)", 8, "5.0")
        self.add_entry(input_frame, "Tiempo total a graficar (s)", 9, "8.0")

        # --- Botones de Acción ---
        btn_calculate = ttk.Button(input_frame, text="Calcular y Graficar", command=self.run_simulation)
        btn_calculate.grid(row=10, column=0, columnspan=2, pady=20, sticky="ew")

        # --- Frame para los Resultados ---
        results_frame = ttk.LabelFrame(input_frame, text="Resultados Calculados", padding="10")
        results_frame.grid(row=11, column=0, columnspan=2, sticky="ew")
        self.results_labels = {}
        self.add_result_label(results_frame, "Razón Amortiguamiento (ζ)", 0)
        self.add_result_label(results_frame, "Frec. Amortiguada (ωd)", 1)
        self.add_result_label(results_frame, "Frec. Natural (ωn)", 2)
        self.add_result_label(results_frame, "Desplazamiento u(t) (cm)", 3)

        # --- Simulación inicial para mostrar una gráfica al abrir ---
        self.run_simulation()

    def add_entry(self, parent, text, row, default_value):
        """Función auxiliar para crear un Label y un Entry."""
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        var = tk.StringVar(value=default_value)
        entry = ttk.Entry(parent, textvariable=var, width=15)
        entry.grid(row=row, column=1, sticky="e", padx=5, pady=5)
        self.entries[text] = var

    def add_result_label(self, parent, text, row):
        """Función auxiliar para crear las etiquetas de resultados."""
        ttk.Label(parent, text=f"{text}:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        var = tk.StringVar(value="--")
        ttk.Label(parent, textvariable=var, font=("Helvetica", 10, "bold")).grid(row=row, column=1, sticky="w", padx=5,
                                                                                 pady=2)
        self.results_labels[text] = var

    def get_float_from_entry(self, key):
        """Obtiene y convierte el valor de una entrada a float, con manejo de errores."""
        try:
            return float(self.entries[key].get())
        except ValueError:
            messagebox.showerror("Error de Entrada", f"El valor para '{key}' no es un número válido.")
            return None

    def run_simulation(self):
        """
        Orquesta el proceso de cálculo y graficación.
        Maneja las excepciones para asegurar que la app no falle.
        """
        try:
            # 1. Extraer y validar datos de entrada
            params = {
                'ui': self.get_float_from_entry("Amplitud Pico ui (cm)"),
                'uij': self.get_float_from_entry("Amplitud Pico ui+j (cm)"),
                'j': self.get_float_from_entry("Número de ciclos j"),
                'Td': self.get_float_from_entry("Período Amortiguado Td (s)"),
                'u0': self.get_float_from_entry("Desplazamiento Inicial u(0) (cm)"),
                'v0': self.get_float_from_entry("Velocidad Inicial v(0) (cm/s)"),
                't_eval': self.get_float_from_entry("Tiempo a evaluar t (s)"),
                't_total': self.get_float_from_entry("Tiempo total a graficar (s)")
            }

            if any(p is None for p in params.values()):
                return  # Detener si hubo un error de conversión

            if params['j'] <= 0 or params['ui'] <= 0 or params['uij'] <= 0 or params['Td'] <= 0:
                messagebox.showerror("Error de Lógica", "Las amplitudes, ciclos y período deben ser positivos.")
                return

            # 2. Realizar los cálculos físicos
            # Cálculo de la razón de amortiguamiento (ζ)
            zeta = (1 / (2 * np.pi * params['j'])) * np.log(params['ui'] / params['uij'])

            # Cálculo de frecuencias
            wd = (2 * np.pi) / params['Td']  # Frecuencia amortiguada
            wn = wd / np.sqrt(1 - zeta ** 2)  # Frecuencia natural

            # 3. Calcular la respuesta del sistema en el tiempo
            t = np.linspace(0, params['t_total'], 1000)

            # Ecuación completa de la respuesta de desplazamiento u(t)
            term_cos = params['u0'] * np.cos(wd * t)
            term_sin_numerator = params['v0'] + zeta * wn * params['u0']
            term_sin = (term_sin_numerator / wd) * np.sin(wd * t)
            envelope = np.exp(-zeta * wn * t)

            u_t = envelope * (term_cos + term_sin)

            # --- LÓGICA DE DETECCIÓN DE PICOS Y VALLES ---
            peak_indices = []
            valley_indices = []
            for i in range(1, len(u_t) - 1):
                # Condición para picos (máximo local positivo)
                if u_t[i] > u_t[i - 1] and u_t[i] > u_t[i + 1] and u_t[i] > 0:
                    peak_indices.append(i)
                # Condición para valles (mínimo local negativo)
                if u_t[i] < u_t[i - 1] and u_t[i] < u_t[i + 1] and u_t[i] < 0:
                    valley_indices.append(i)

            # Calcular el desplazamiento en el punto específico t_eval
            u_t_eval_cos = params['u0'] * np.cos(wd * params['t_eval'])
            u_t_eval_sin = (term_sin_numerator / wd) * np.sin(wd * params['t_eval'])
            envelope_eval = np.exp(-zeta * wn * params['t_eval'])
            u_t_eval = envelope_eval * (u_t_eval_cos + u_t_eval_sin)

            # 4. Actualizar las etiquetas de resultados
            self.results_labels["Razón Amortiguamiento (ζ)"].set(f"{zeta:.5f}")
            self.results_labels["Frec. Amortiguada (ωd)"].set(f"{wd:.3f} rad/s")
            self.results_labels["Frec. Natural (ωn)"].set(f"{wn:.3f} rad/s")
            self.results_labels["Desplazamiento u(t) (cm)"].set(f"{u_t_eval:.4f}")

            # 5. Actualizar la gráfica
            self.plot_response(t, u_t, envelope, params['t_eval'], u_t_eval, peak_indices, valley_indices)

        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error durante el cálculo:\n{e}")

    def plot_response(self, t, u_t, envelope, t_eval, u_t_eval, peak_indices, valley_indices):
        """
        Limpia y redibuja la gráfica con los nuevos datos, incluyendo anotaciones.
        """
        self.ax.clear()
        self.ax.plot(t, u_t, label="Respuesta del sistema u(t)", color="b")
        self.ax.plot(t, envelope, 'r--', label="Envolvente de decaimiento", alpha=0.7)
        self.ax.plot(t, -envelope, 'r--', alpha=0.7)
        self.ax.plot(t_eval, u_t_eval, 'go', markersize=8, label=f"u({t_eval}s) = {u_t_eval:.3f} cm")

        # --- LÓGICA DE ANOTACIONES ---
        # Se calcula un offset vertical basado en el rango total de la gráfica.
        y_range = np.max(u_t) - np.min(u_t) if len(u_t) > 0 else 2
        text_offset = y_range * 0.05

        # Anotar los picos positivos
        for i in peak_indices:
            peak_x, peak_y = t[i], u_t[i]
            self.ax.annotate(f"{peak_y:.2f}",
                             xy=(peak_x, peak_y),
                             xytext=(peak_x, peak_y + text_offset),
                             ha='center',
                             fontsize=9,
                             fontweight='bold',
                             color='darkgreen')

        # Anotar los valles (picos negativos)
        for i in valley_indices:
            valley_x, valley_y = t[i], u_t[i]
            self.ax.annotate(f"{valley_y:.2f}",
                             xy=(valley_x, valley_y),
                             xytext=(valley_x, valley_y - text_offset),
                             ha='center',
                             fontsize=9,
                             fontweight='bold',
                             color='purple')

        self.ax.set_title("Respuesta de Vibración Libre Amortiguada")
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Desplazamiento (cm)")
        self.ax.legend()
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.axhline(0, color='black', linewidth=0.5)

        self.canvas.draw()


if __name__ == "__main__":
    # --- Punto de entrada de la aplicación ---
    # Se crea la ventana principal y se inicia el bucle de eventos de tkinter.
    root = tk.Tk()
    app = VibrationApp(root)
    root.mainloop()

