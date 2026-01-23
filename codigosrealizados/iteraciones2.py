"""
HERRAMIENTA DE ESCRITORIO PARA ANÁLISIS DE COMPUERTA CON FLUJO MODULAR
Versión: 2.1 (Ajuste Maqueta - Escala Reducida)
Autor: Emanuel - Análisis Estructural & Machine Learning
Fecha: 2025-07-27

Descripción:
Aplicación GUI para análisis hidráulico iterativo de compuertas planas.
Incluye correcciones para aperturas pequeñas (a < 5mm) y logging científico.
Ajustado para escenarios de laboratorio/maqueta (y1 < 1cm).
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import logging
import sys
from datetime import datetime
import os

# Configuración de Matplotlib para Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING (Requisito de Auditoría Científica)
# -----------------------------------------------------------------------------
# Crear directorio de logs si no existe
log_dir = "logs_ingenieria"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"simulacion_compuerta_{timestamp}.log")

# Configurar logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class CompuertaCalculadora:
    """
    Motor de cálculo hidráulico.
    Principio: Conservación de energía (Bernoulli) entre aguas arriba y vena contraída.
    """

    def __init__(self, Q, b, Cc):
        self.Q = float(Q)
        self.b = float(b)
        self.Cc = float(Cc)
        self.g = 9.81
        logger.info(f"Iniciando calculadora: Q={Q}, b={b}, Cc={Cc}")

    def estimar_y1_orificio(self, a):
        """
        Estima y1 usando la aproximación de orificio sumergido.
        Ideal para cuando a << y1 (aperturas pequeñas).
        Q = Cd * A * sqrt(2g * h)
        """
        try:
            Ao = a * self.b
            # Asumimos un Cd inicial conservador para orificio pequeño
            Cd_est = 0.60
            # Despejando h (aprox y1) de la ec. del orificio
            # Q^2 = Cd^2 * Ao^2 * 2g * y1
            y1_est = (self.Q**2) / ((Cd_est * Ao)**2 * 2 * self.g)

            # Factor de seguridad para asegurar que estamos sobre el nivel
            y1_est = max(y1_est, a * 1.5)

            logger.debug(f"Estimación inicial por orificio: {y1_est:.4f} m")
            return y1_est
        except ZeroDivisionError:
            logger.error("Error de división por cero en estimación inicial.")
            return a * 10.0

    def calcular_parametros(self, a, y1):
        """Calcula todos los parámetros hidráulicos para un y1 dado."""
        try:
            y2 = self.Cc * a

            if y1 <= y2:
                # Situación física imposible para flujo modular normal, penalizamos
                return {
                    'Diff': 1e9, # Penalización alta
                    'Error': True,
                    'y2': y2
                }

            Ao = a * self.b

            # Coeficiente de velocidad (corrección por velocidad de aproximación)
            Cv = 1.0 / np.sqrt(1.0 - (y2/y1)**2) if y1 > y2 else 1.0 # Corrección fórmula standard
            # Nota: Tu fórmula original era 1/sqrt(1 + y2/y1), la standard suele ser relacionada a 1-(Ao/A1)^2
            # Mantendré tu fórmula original si es una empírica específica, pero
            # para flujo modular standard, la velocidad teórica es:

            v_teorica = np.sqrt(2 * self.g * (y1 - y2))

            # Usando tu definición de Cd
            # Si tu fórmula de Cv es específica de tu laboratorio, úsala.
            # Si no, Cv suele ser cercano a 1.0 para y1 grande.
            Cv_original = 1 / np.sqrt(1 + y2/y1)
            Cd = self.Cc * Cv_original

            Q_calc = Cd * Ao * v_teorica
            Diff = Q_calc - self.Q

            v2 = Q_calc / (self.b * y2)
            F = v2 / np.sqrt(self.g * y2)

            # Resalto hidráulico (Ec. Belanger)
            if 1 + 8*F**2 >= 0:
                y3 = (y2/2) * (-1 + np.sqrt(1 + 8*F**2))
                L = 6 * (y3 - y2)
            else:
                y3 = 0
                L = 0

            return {
                'a': a, 'y1': y1, 'y2': y2, 'Ao': Ao, 'Cv': Cv_original, 'Cd': Cd,
                'v_teorica': v_teorica, 'Q_calc': Q_calc, 'Diff': Diff,
                'v2': v2, 'F': F, 'y3': y3, 'L': L,
                'y1/a': y1/a if a > 0 else 0,
                '2/3*y1': (2/3)*y1,
                'cumple_y1_a': (y1/a > 1.35) if a > 0 else False,
                'cumple_flujo_modular': ((2/3)*y1 > a),
                'Error': False
            }
        except Exception as e:
            logger.error(f"Error en cálculo de parámetros: {str(e)}")
            return {'Diff': 1e9, 'Error': True}

    def newton_raphson_iteracion(self, a, y1_input_user, tol=1e-8, max_iter=200):
        """
        Solver numérico robusto.
        Estrategia: Híbrida (Estimación Orificio -> Newton Raphson con Relajación).
        """
        logger.info(f"Iniciando solver para a={a:.6f} m")

        y2 = self.Cc * a

        # 1. Determinación Inteligente del Punto de Partida
        # Si el usuario da un valor muy bajo (menor a y2 o a), usamos la estimación de orificio
        if y1_input_user <= y2 * 1.1 or y1_input_user < a:
            y1 = self.estimar_y1_orificio(a)
            logger.info(f"y1 inicial ajustado automáticamente a: {y1:.6f} m")
        else:
            y1 = y1_input_user
            logger.info(f"Usando y1 inicial del usuario: {y1:.6f} m")

        iteraciones = []
        mejor_diff = float('inf')
        mejor_y1 = y1

        convergio = False

        for i in range(max_iter):
            params = self.calcular_parametros(a, y1)

            if params.get('Error'):
                y1 = y1 * 1.5 # Empujamos hacia arriba si hay error matemático
                continue

            # Registro de iteración
            iter_data = {
                'iteracion': i + 1,
                'y1': y1,
                'y2': params['y2'],
                'Cd': params['Cd'],
                'Q_calc': params['Q_calc'],
                'Diff': params['Diff'],
                'F': params['F']
            }
            iteraciones.append(iter_data)

            # Guardar mejor resultado por si acaso no converge
            if abs(params['Diff']) < abs(mejor_diff):
                mejor_diff = params['Diff']
                mejor_y1 = y1

            # Criterio de Convergencia
            # Usamos tolerancia relativa para caudales muy pequeños
            if abs(params['Diff']) < tol or abs(params['Diff']/self.Q) < 1e-4:
                logger.info(f"Convergencia alcanzada en iteración {i+1}. Diff: {params['Diff']:.2e}")
                convergio = True
                return y1, params, iteraciones, True

            # Derivada Numérica (Diferencia Finita Central)
            # El paso 'h' debe ser pequeño pero no causar underflow
            h = max(1e-7, y1 * 1e-4)

            p_plus = self.calcular_parametros(a, y1 + h)
            p_minus = self.calcular_parametros(a, y1 - h)

            if p_plus.get('Error') or p_minus.get('Error'):
                 df = (p_plus['Diff'] - params['Diff']) / h # Fallback a diferencia forward
            else:
                 df = (p_plus['Diff'] - p_minus['Diff']) / (2 * h)

            # Protección contra derivada cero
            if abs(df) < 1e-12:
                logger.warning("Derivada cercana a cero. Aplicando perturbación aleatoria.")
                df = 1e-6 * np.sign(params['Diff'])

            # Paso de Newton
            delta = params['Diff'] / df

            # Factor de Relajación (Learning Rate)
            # Para hidráulica no lineal, a veces paso completo es inestable
            alpha = 1.0

            y1_nuevo = y1 - alpha * delta

            # ---------------------------------------------------------
            # CORRECCIÓN CRÍTICA DE LÍMITES (La solución al problema)
            # ---------------------------------------------------------
            y1_min = y2 * 1.01 # Debe ser estrictamente mayor a y2

            # ELIMINADO EL LIMITE SUPERIOR RESTRICTIVO (y2 * 10)
            # Para a=3.5mm, y1=55mm es > 15 veces y2. El límite debe ser físico (ej. altura del canal)
            y1_max = 100.0 # Metros (virtualmente infinito para este caso)

            if y1_nuevo < y1_min:
                # Si Newton intenta ir bajo el límite físico, usamos bisección hacia el mínimo
                y1_nuevo = (y1 + y1_min) / 2
                logger.debug("Newton intentó violar límite inferior. Corrigiendo.")

            y1 = y1_nuevo

        logger.warning(f"No se alcanzó convergencia absoluta. Mejor Diff: {mejor_diff:.2e}")
        # Recalcular con el mejor valor encontrado
        params_final = self.calcular_parametros(a, mejor_y1)
        return mejor_y1, params_final, iteraciones, False

class CompuertaApp:
    """
    Controlador de la Interfaz Gráfica.
    Maneja la interacción usuario-sistema y visualización.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("EMAIRC VISIÓN - Análisis Hidráulico Modular v2.1 (Maqueta)")
        self.root.geometry("1400x950")
        # Estilo para parecer más "Enterprise"
        style = ttk.Style()
        style.theme_use('clam')

        # Variables AJUSTADAS PARA MAQUETA/ESCALA REDUCIDA
        # Objetivo: y1 < 7mm para a=3.5mm
        # Q reducido de 0.1 L/s a 0.032 L/s aprox.
        self.Q_var = tk.DoubleVar(value=0.000032) # m3/s (32 ml/s)
        self.b_var = tk.DoubleVar(value=0.05)     # m (5 cm ancho canal)
        self.Cc_var = tk.DoubleVar(value=0.61)    # Adimensional
        self.a_var = tk.DoubleVar(value=0.0035)   # 3.5 mm en metros
        self.y1_inicial_var = tk.DoubleVar(value=0.007) # 7 mm (Semilla inicial cercana a meta)
        self.tol_var = tk.StringVar(value="1e-10")

        # Estado
        self.ultimo_resultado = None
        self.ultimas_iteraciones = None

        self.crear_interfaz()
        logger.info("Interfaz gráfica inicializada para escenario Maqueta")

    def crear_interfaz(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Layout de 2 columnas
        left_panel = ttk.Frame(main_frame, width=400, relief="groove", borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_panel = ttk.Frame(main_frame, relief="flat")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.crear_controles(left_panel)
        self.crear_resultados(right_panel)

    def crear_controles(self, parent):
        ttk.Label(parent, text="PANEL DE CONTROL (MAQUETA)", font=("Helvetica", 14, "bold")).pack(pady=15)

        input_frame = ttk.LabelFrame(parent, text="Parámetros de Entrada")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        inputs = [
            ("Caudal Q (m³/s):", self.Q_var),
            ("Ancho b (m):", self.b_var),
            ("Coef. Contracción Cc:", self.Cc_var),
            ("Apertura a (m):", self.a_var),
            ("y1 Inicial (m):", self.y1_inicial_var)
        ]

        for i, (label_text, var) in enumerate(inputs):
            f = ttk.Frame(input_frame)
            f.pack(fill=tk.X, pady=5)
            ttk.Label(f, text=label_text, width=20).pack(side=tk.LEFT, padx=5)
            # Entry validado visualmente para números muy pequeños
            e = ttk.Entry(f, textvariable=var, width=15)
            e.pack(side=tk.RIGHT, padx=5)

        # Combo Tolerancia
        f_tol = ttk.Frame(input_frame)
        f_tol.pack(fill=tk.X, pady=5)
        ttk.Label(f_tol, text="Tolerancia:", width=20).pack(side=tk.LEFT, padx=5)
        ttk.Combobox(f_tol, textvariable=self.tol_var,
                     values=["1e-6", "1e-8", "1e-10", "1e-12"]).pack(side=tk.RIGHT, padx=5)

        # Botones de Acción
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, pady=20)

        btn_calc = ttk.Button(action_frame, text="▶ EJECUTAR ANÁLISIS", command=self.calcular)
        btn_calc.pack(fill=tk.X, pady=5)

        btn_export = ttk.Button(action_frame, text="💾 EXPORTAR DATOS", command=self.exportar_csv)
        btn_export.pack(fill=tk.X, pady=5)

        # Información de Debug
        self.status_var = tk.StringVar(value="Listo para simulación de maqueta.")
        lbl_status = ttk.Label(parent, textvariable=self.status_var,
                              foreground="blue", wraplength=350)
        lbl_status.pack(side=tk.BOTTOM, pady=10)

        # Presets Rápidos
        preset_frame = ttk.LabelFrame(parent, text="Casos de Prueba")
        preset_frame.pack(fill=tk.X, pady=10)

        # Ajustado para tu requerimiento de maqueta
        ttk.Button(preset_frame, text="Caso Maqueta (Objetivo < 7mm)",
                   command=lambda: self.cargar_preset(0.0035, 0.000032)).pack(fill=tk.X, pady=2)

        ttk.Button(preset_frame, text="Caso Laboratorio (5.5 cm)",
                   command=lambda: self.cargar_preset(0.0035, 0.0001057)).pack(fill=tk.X, pady=2)

    def crear_resultados(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Numérico
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Datos Numéricos")

        self.txt_resultados = tk.Text(self.tab_data, font=("Consolas", 10))
        self.txt_resultados.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 2: Gráficos
        self.tab_plots = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plots, text="Gráficos de Convergencia")

        # Placeholder para gráficos
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plots)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def cargar_preset(self, a_val, q_val):
        self.a_var.set(a_val)
        self.Q_var.set(q_val)
        # Ajustamos el valor inicial de y1 según el caudal para ayudar al solver
        if q_val < 0.00005: # Caudal muy bajo (Maqueta)
            self.y1_inicial_var.set(0.007)
            self.status_var.set(f"Preset Maqueta: Q={q_val*1000000:.1f} ml/s")
        else:
            self.y1_inicial_var.set(0.055)
            self.status_var.set(f"Preset Laboratorio: Q={q_val*1000:.2f} L/s")

    def calcular(self):
        try:
            # 1. Obtención de datos (Try/Except para inputs no numéricos)
            Q = self.Q_var.get()
            b = self.b_var.get()
            Cc = self.Cc_var.get()
            a = self.a_var.get()
            y1_ini = self.y1_inicial_var.get()
            tol = float(self.tol_var.get())

            logger.info("Iniciando secuencia de cálculo...")

            # 2. Instancia del motor
            calc = CompuertaCalculadora(Q, b, Cc)

            # 3. Ejecución
            y1_final, params, historial, exito = calc.newton_raphson_iteracion(
                a, y1_ini, tol=tol
            )

            self.ultimo_resultado = params
            self.ultimas_iteraciones = historial

            # 4. Actualización UI
            self.mostrar_resultados_texto(params, historial, exito)
            self.actualizar_graficos(historial)

            if exito:
                self.status_var.set(f"Éxito: y1 = {params['y1']*1000:.2f} mm")
            else:
                self.status_var.set("Advertencia: No convergió totalmente (Revisar logs).")

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", "Por favor verifique que todos los campos sean números válidos.")
            logger.error(f"Input error: {ve}")
        except Exception as e:
            messagebox.showerror("Error Crítico", f"Ocurrió un error inesperado:\n{str(e)}")
            logger.critical(f"Unhandled exception: {e}", exc_info=True)

    def mostrar_resultados_texto(self, params, historial, exito):
        self.txt_resultados.delete(1.0, tk.END)

        header = f"""
============================================================
REPORTE DE ANÁLISIS HIDRÁULICO - EMAIRC VISIÓN
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
============================================================
ESTADO: {"✅ CONVERGENCIA EXITOSA" if exito else "⚠️ APROXIMACIÓN (NO CONVERGIÓ)"}
Iteraciones: {len(historial)}

RESULTADOS FINALES (ESCALA MAQUETA):
-------------------
Tirante Aguas Arriba (y1) : {params['y1']:.6f} m  ({params['y1']*1000:.2f} mm)
Tirante Contraído (y2)    : {params['y2']:.6f} m  ({params['y2']*1000:.2f} mm)
Apertura (a)              : {params['a']:.6f} m   ({params['a']*1000:.2f} mm)
Coeficiente Cd            : {params['Cd']:.6f}
Número de Froude (F)      : {params['F']:.4f}

VERIFICACIÓN DE CAUDAL:
-----------------------
Q Objetivo                : {self.Q_var.get():.8f} m³/s ({self.Q_var.get()*1000000:.1f} ml/s)
Q Calculado               : {params['Q_calc']:.8f} m³/s
Diferencia (Error)        : {params['Diff']:.2e}

CONDICIONES DE FLUJO:
---------------------
Relación y1/a             : {params['y1/a']:.2f} (Debe ser > 1.35) -> {"CUMPLE" if params['cumple_y1_a'] else "NO CUMPLE"}
Modularidad (2/3 y1 > a)  : {params['2/3*y1']:.4f} > {params['a']:.4f} -> {"CUMPLE" if params['cumple_flujo_modular'] else "NO CUMPLE"}

TIRANTE CONJUGADO (Salto Hidráulico):
-------------------------------------
y3 (Secuente)             : {params['y3']:.4f} m
Longitud Salto (L)        : {params['L']:.4f} m
============================================================
"""
        self.txt_resultados.insert(tk.END, header)

        # Tabla de iteraciones
        self.txt_resultados.insert(tk.END, "\nHISTORIAL DE ITERACIONES:\n")
        self.txt_resultados.insert(tk.END, f"{'Iter':<5} {'y1 (m)':<12} {'Diff':<12} {'Cd':<10}\n")
        self.txt_resultados.insert(tk.END, "-"*45 + "\n")

        for item in historial:
            self.txt_resultados.insert(tk.END, f"{item['iteracion']:<5} {item['y1']:.6f}   {item['Diff']:.2e}   {item['Cd']:.4f}\n")

    def actualizar_graficos(self, historial):
        # Limpiar
        self.ax1.clear()
        self.ax2.clear()

        if not historial:
            self.canvas.draw()
            return

        df = pd.DataFrame(historial)

        # Plot 1: Evolución de y1
        self.ax1.plot(df['iteracion'], df['y1']*1000, 'b-o', markersize=4) # Convertido a mm para visualizar mejor en maqueta
        self.ax1.set_ylabel('Tirante y1 (mm)')
        self.ax1.set_title('Convergencia de Tirante y1 (mm)')
        self.ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: Reducción del Error (Escala Log)
        # Añadimos pequeño epsilon para evitar log(0)
        errors = np.abs(df['Diff']) + 1e-15
        self.ax2.plot(df['iteracion'], errors, 'r-o', markersize=4)
        self.ax2.set_ylabel('Error Residual (log)')
        self.ax2.set_xlabel('Iteraciones')
        self.ax2.set_yscale('log')
        self.ax2.grid(True, linestyle='--', alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

    def exportar_csv(self):
        if self.ultimas_iteraciones is None:
            messagebox.showwarning("Sin datos", "Primero realice un cálculo.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            try:
                df = pd.DataFrame(self.ultimas_iteraciones)
                df.to_csv(filename, index=False)
                logger.info(f"Datos exportados a {filename}")
                messagebox.showinfo("Éxito", "Exportación completada.")
            except Exception as e:
                logger.error(f"Error exportando CSV: {e}")
                messagebox.showerror("Error", "No se pudo guardar el archivo.")

def main():
    root = tk.Tk()
    # Icono si existiera, envuelto en try para no romper si falta
    try:
        # root.iconbitmap("icono.ico")
        pass
    except:
        pass

    app = CompuertaApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()