import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import logging
import math
import os
import sys
import datetime

# --- Dependencia para Gráficos ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.patches import Rectangle, Circle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("ADVERTENCIA: Matplotlib no está instalado. Los gráficos no funcionarán.")
    print("Instale con: pip install matplotlib")

# --- Configuración del Logging (Requisito del Usuario) ---
log_file_path = "diseño_albanileria_v3_log_detallado.log"

# --- INICIO DE CORRECCIÓN (UnicodeEncodeError) ---
# 1. Se especifica encoding='utf-8' para el FileHandler.
# 2. Se elimina el StreamHandler(sys.stdout) que causa el crash en consolas Windows (cp1252).
#    El log de la GUI (GuiLogHandler) y el FileHandler son suficientes.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    ]
)
# --- FIN DE CORRECCIÓN ---

logger = logging.getLogger()


class GuiLogHandler(logging.Handler):
    """Handler para redirigir logs al widget ScrolledText."""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        # Usamos un formato más limpio para la GUI
        self.formatter = logging.Formatter('%(levelname)s: %(message)s')

    def emit(self, record):
        msg = self.format(record)

        def append_msg():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)

        # Usar 'after' para programar la actualización de la GUI de forma segura
        self.text_widget.after(0, append_msg)


# --- Módulo de Lógica de Diseño (NTE E.070, E.030, E.060) ---

class CalculadoraIngenieria:
    """Encapsula toda la lógica de cálculo y la generación de la memoria."""

    def __init__(self):
        self.memoria_calculo = []
        self.resultados_graficos = {}
        logger.info("Módulo de Calculadora de Ingeniería inicializado.")

    def log_memoria(self, mensaje, nivel=0, es_titulo=False, cita=None):
        """Agrega un mensaje al log y a la memoria de cálculo."""
        prefijo_mapa = {0: "##", 1: "###", 2: "####", 3: "-"}
        prefijo = prefijo_mapa.get(nivel, "-")

        cita_str = f" [NTE: {cita}]" if cita else ""

        if es_titulo:
            linea_gruesa = "=" * 80
            linea_fina = "-" * 80
            if nivel == 0:
                logger.info(linea_gruesa)
                self.memoria_calculo.append(f"\n{linea_gruesa}\n")
            else:
                logger.info(linea_fina)
                self.memoria_calculo.append(f"\n{linea_fina}\n")

        texto_log = f"{'  ' * nivel}{prefijo} {mensaje}{cita_str}"
        logger.info(texto_log)
        self.memoria_calculo.append(texto_log)

    def calcular_diseno_completo(self, params):
        """
        Orquesta todo el flujo de cálculo y genera la memoria.
        """
        self.memoria_calculo = [f"MEMORIA DE CÁLCULO - DISEÑO DE ALBAÑILERÍA CONFINADA\n",
                                f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                                f"Proyecto: {params['proyecto']}\n",
                                f"Ingeniero: {params['ingeniero']}\n",
                                f"Muro Analizado: {params['muro_id']}\n"]
        self.resultados_graficos = {}

        try:
            # --- PASO 1: ANÁLISIS SÍSMICO ESTÁTICO (NTE E.030) ---
            self.log_memoria("PASO 1: ANÁLISIS SÍSMICO ESTÁTICO (NTE E.030)", nivel=0, es_titulo=True)

            # 1.1: Parámetros de Diseño
            self.log_memoria("1.1. Parámetros Sísmicos de Diseño (E.030)", nivel=1, es_titulo=True)
            Z, U, S = params['Z'], params['U'], params['S']
            R0, Ia, Ip = params['R0'], params['Ia'], params['Ip']
            self.log_memoria(f"Factor de Zona (Z): {Z} (Tabla 1)", nivel=3, cita="E.030 Art. 10 / Tabla 1")
            self.log_memoria(f"Factor de Uso (U): {U} (Tabla 5)", nivel=3, cita="E.030 Art. 15 / Tabla 5")
            self.log_memoria(f"Factor de Suelo (S): {S} (Tabla 3)", nivel=3, cita="E.030 Art. 13 / Tabla 3")
            self.log_memoria(f"Coef. Básico Reducción (R0): {R0} (Tabla 7)", nivel=3, cita="E.030 Art. 18 / Tabla 7")
            self.log_memoria(f"Factor Irregularidad Altura (Ia): {Ia} (Tabla 8)", nivel=3,
                             cita="E.030 Art. 20 / Tabla 8")
            self.log_memoria(f"Factor Irregularidad Planta (Ip): {Ip} (Tabla 9)", nivel=3,
                             cita="E.030 Art. 20 / Tabla 9")

            R = R0 * Ia * Ip
            self.log_memoria(f"Coef. Reducción Sísmica (R = R0 * Ia * Ip): {R0} * {Ia} * {Ip} = {R:.2f}", nivel=2,
                             cita="E.030 Art. 22")

            # 1.2: Período y Factor de Amplificación
            self.log_memoria("1.2. Período y Factor de Amplificación (E.030)", nivel=1, es_titulo=True)
            h_n, C_T = params['h_n'], params['C_T']
            T = h_n / C_T
            self.log_memoria(f"Altura total (h_n): {h_n} m", nivel=3)
            self.log_memoria(f"Coeficiente C_T: {C_T} (Albañilería)", nivel=3, cita="E.030 Art. 28.4.1")
            self.log_memoria(f"Período (T = h_n / C_T): {h_n} / {C_T} = {T:.3f} s", nivel=2, cita="E.030 Art. 28.4.1")

            Tp, Tl = params['Tp'], params['Tl']
            self.log_memoria(f"Período de Suelo (Tp): {Tp} s (Tabla 4)", nivel=3, cita="E.030 Art. 13 / Tabla 4")
            self.log_memoria(f"Período de Suelo (Tl): {Tl} s (Tabla 4)", nivel=3, cita="E.030 Art. 13 / Tabla 4")

            if T < Tp:
                C = 2.5
                self.log_memoria(f"T < Tp ({T:.3f} < {Tp}) -> C = 2.5", nivel=2, cita="E.030 Art. 14")
            elif T < Tl:
                C = 2.5 * (Tp / T)
                self.log_memoria(f"Tp < T < Tl -> C = 2.5 * (Tp / T) = 2.5 * ({Tp} / {T:.3f}) = {C:.3f}", nivel=2,
                                 cita="E.030 Art. 14")
            else:
                C = 2.5 * (Tp * Tl) / (T ** 2)
                self.log_memoria(f"T > Tl -> C = 2.5 * (Tp * Tl) / T² = 2.5 * ({Tp} * {Tl}) / ({T:.3f}²) = {C:.3f}",
                                 nivel=2, cita="E.030 Art. 14")

            # 1.3: Cortante Basal (Fuerzas Sísmicas)
            self.log_memoria("1.3. Cortante Basal del Edificio (E.030)", nivel=1, es_titulo=True)
            P_edificio = params['P_edificio']
            self.log_memoria(f"Peso Total Edificio (P): {P_edificio} kN (Dato de Metrado)", nivel=3,
                             cita="E.030 Art. 26")

            V_base = (Z * U * C * S / R) * P_edificio
            self.log_memoria(
                f"Cortante Basal (V = ZUCS/R * P): ({Z} * {U} * {C:.3f} * {S}) / {R:.2f} * {P_edificio} kN", nivel=3)
            self.log_memoria(f"V_base = {V_base:.2f} kN", nivel=2, cita="E.030 Art. 28.2.1")

            # 1.4: Fuerzas en el Muro (Entrada del Ingeniero)
            self.log_memoria("1.4. Fuerzas en el Muro (Análisis Estructural)", nivel=1, es_titulo=True)
            self.log_memoria(
                "NOTA: Las siguientes fuerzas (Ve, Me) dependen de la distribución de rigideces del edificio.", nivel=3)

            Ve1_muro = params['Ve1_muro']
            Me1_muro = params['Me1_muro']
            Pg_muro = params['Pg_muro']
            self.log_memoria(f"Ve (Cortante Sismo Moderado en Muro): {Ve1_muro} kN (Dato de Análisis)", nivel=3)
            self.log_memoria(f"Me (Momento Sismo Moderado en Muro): {Me1_muro} kN-m (Dato de Análisis)", nivel=3)
            self.log_memoria(f"Pg (Carga Gravitacional en Muro): {Pg_muro} kN (Dato de Metrado)", nivel=3)

            # --- PASO 2: CAPACIDAD DEL MURO (NTE E.070) ---
            self.log_memoria("PASO 2: CAPACIDAD DEL MURO (NTE E.070, Art. 8.5)", nivel=0, es_titulo=True)

            L, h, t_mm = params['L_muro_m'], params['h_muro_m'], params['t_muro_cm'] * 10
            v_m = params['v_m_mpa']

            # 2.1: Factor de Esbeltez (α)
            self.log_memoria("2.1. Factor de Esbeltez (α)", nivel=1, es_titulo=True)
            if Ve1_muro == 0 or Me1_muro == 0:
                alpha = 1.0
                self.log_memoria("Ve o Me es 0. Se asume α = 1.0", nivel=3)
            else:
                alpha = (Ve1_muro * 1000 * L) / (Me1_muro * 1000)  # (kN*m / kNm)
                self.log_memoria(f"α = (Ve * L) / Me = ({Ve1_muro} kN * {L} m) / {Me1_muro} kNm = {alpha:.3f}", nivel=3,
                                 cita="E.070 Art. 8.5.3")

            alpha_limitado = max(1 / 3, min(alpha, 1.0))
            self.log_memoria(f"Se limita α al rango [1/3, 1.0]. α = {alpha_limitado:.3f}", nivel=2,
                             cita="E.070 Art. 8.5.3")

            # 2.2: Resistencia al Corte (Vm1)
            self.log_memoria("2.2. Resistencia al Corte del Muro (Vm1)", nivel=1, es_titulo=True)
            Vm1 = (0.5 * v_m * alpha_limitado * t_mm * (L * 1000)) + (0.23 * (Pg_muro * 1000))  # N
            Vm1_kN = Vm1 / 1000
            self.log_memoria(f"Vm = 0.5*v'm*α*t*L + 0.23*Pg (Unidades Arcilla)", nivel=3, cita="E.070 Art. 8.5.3")
            self.log_memoria(f"Vm = 0.5 * {v_m} * {alpha_limitado:.3f} * {t_mm} * {L * 1000} + 0.23 * {Pg_muro * 1000}",
                             nivel=3)
            self.log_memoria(
                f"Vm = {(0.5 * v_m * alpha_limitado * t_mm * (L * 1000)) / 1000:.2f} kN (Aporte Albañilería) + {(0.23 * (Pg_muro * 1000)) / 1000:.2f} kN (Aporte Carga Axial)",
                nivel=3)
            self.log_memoria(f"Vm = {Vm1_kN:.2f} kN", nivel=2)

            # --- PASO 3: FUERZAS DE DISEÑO (NTE E.070) ---
            self.log_memoria("PASO 3: FUERZAS DE DISEÑO EN CONFINAMIENTOS (NTE E.070, Art. 8.6)", nivel=0,
                             es_titulo=True)

            # 3.1: Factor de Amplificación Sismo Severo
            self.log_memoria("3.1. Factor de Amplificación por Sismo Severo", nivel=1, es_titulo=True)
            if Ve1_muro <= 0:
                factor_amp = 3.0
                self.log_memoria(f"Ve1 es cero. Se asume Factor de Amplificación máximo: 3.0", nivel=3)
            else:
                factor_amp = Vm1_kN / Ve1_muro
                self.log_memoria(f"Factor = Vm1 / Ve1 = {Vm1_kN:.2f} / {Ve1_muro} = {factor_amp:.3f}", nivel=3)

            factor_amp_limitado = max(2.0, min(factor_amp, 3.0))
            self.log_memoria(f"Se limita Factor al rango [2.0, 3.0]. Factor = {factor_amp_limitado:.2f}", nivel=2,
                             cita="E.070 Art. 8.6")

            # 3.2: Fuerzas de Sismo Severo
            self.log_memoria("3.2. Fuerzas de Sismo Severo (Art. 8.6)", nivel=1, es_titulo=True)
            Mu1_kNm = Me1_muro * factor_amp_limitado
            self.log_memoria(f"Mu1 = Me1 * Factor = {Me1_muro} kNm * {factor_amp_limitado:.2f} = {Mu1_kNm:.2f} kNm",
                             nivel=3)
            self.log_memoria(f"Vu1 (Cortante de diseño) = Vm1 = {Vm1_kN:.2f} kN", nivel=3)

            # 3.3: Fuerzas en Elementos (Tabla 11)
            self.log_memoria("3.3. Fuerzas en Elementos (Art. 8.6.3, Tabla 11)", nivel=1, es_titulo=True)
            Pc_col, Lm, Nc = params['Pc_col_kN'], params['Lm_m'], params['Nc_cols']

            M = (Mu1_kNm * 1000 * 1000) - (0.5 * Vm1 * (h * 1000))  # N-mm
            F = M / (L * 1000)  # N
            Vc = 1.5 * Vm1 * (Lm * 1000) / ((L * 1000) * (Nc + 1))  # N
            T = F - (Pc_col * 1000)  # N
            C = F + (Pc_col * 1000)  # N
            Ts = Vm1 * (Lm * 1000) / (2 * (L * 1000))  # N

            self.log_memoria(
                f"M (Momento en base de columnas) = Mu1 - 0.5*Vm1*h = {Mu1_kNm:.2f} - 0.5*{Vm1_kN:.2f}*{h} = {M / 1e6:.2f} kNm",
                nivel=3)
            self.log_memoria(f"F (Fuerza axial por volteo) = M / L = {M / 1e6:.2f} / {L} = {F / 1000:.2f} kN", nivel=3)
            self.log_memoria(f"Vc (Cortante Columna Extrema) = 1.5*Vm1*Lm / (L*(Nc+1)) = {Vc / 1000:.2f} kN", nivel=3)
            self.log_memoria(f"T (Tracción Col.) = F - Pc = {F / 1000:.2f} - {Pc_col} = {T / 1000:.2f} kN", nivel=3)
            self.log_memoria(f"C (Compresión Col.) = F + Pc = {F / 1000:.2f} + {Pc_col} = {C / 1000:.2f} kN", nivel=3)
            self.log_memoria(f"Ts (Tracción Solera) = Vm1*Lm / (2*L) = {Ts / 1000:.2f} kN", nivel=3,
                             cita="E.070 Art. 8.6.3.b")

            # --- PASO 4: DISEÑO DE COLUMNA DE CONFINAMIENTO (NTE E.070, Art. 8.6.3.a) ---
            self.log_memoria("PASO 4: DISEÑO DE COLUMNA DE CONFINAMIENTO (NTE E.070, Art. 8.6.3.a)", nivel=0,
                             es_titulo=True)
            fc_mpa, fy_mpa = params['f_c_mpa'], params['f_y_mpa']
            delta, mu, recub_mm = params['delta'], params['mu'], 20.0
            t_col = t_mm

            # 4.1: Dimensionamiento de Concreto
            self.log_memoria(f"4.1. Dimensionamiento de Concreto (t = {t_col} mm)", nivel=1, es_titulo=True)
            phi_v = 0.85
            A_cf = Vc / (phi_v * 0.2 * fc_mpa)
            self.log_memoria(f"Área por Corte-Fricción A_cf = Vc / (0.85 * 0.2 * f'c) = {A_cf:.0f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.a.1'")

            phi_c = 0.70
            A_s_min_geom_col = 4 * (math.pi * (9.53 ** 2) / 4)  # 4Ø3/8"
            C_design = C if C > 0 else (Pc_col * 1000)  # Usar al menos la carga gravitacional

            A_n_comp = ((C_design / phi_c) - (A_s_min_geom_col * fy_mpa)) / (0.85 * delta * fc_mpa)
            A_n_comp = max(0, A_n_comp)
            self.log_memoria(f"Área Núcleo por Compresión A_n = {A_n_comp:.0f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.a.1")

            b_n = A_n_comp / (t_col - 2 * recub_mm) if (t_col - 2 * recub_mm) > 0 else 0
            b_col_comp = b_n + 2 * recub_mm
            A_c_comp = b_col_comp * t_col
            self.log_memoria(f"Área Columna por Compresión A_c_comp = {A_c_comp:.0f} mm²", nivel=3)

            A_c_min_geom_norma = 15 * (t_col / 10) * 100
            self.log_memoria(f"Área Mínima Geométrica = 15*t_cm*100 = {A_c_min_geom_norma:.0f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.a.1")

            A_c_final_col = max(A_cf, A_c_comp, A_c_min_geom_norma)
            b_col = A_c_final_col / t_col
            b_col_final = math.ceil(b_col / 50) * 50
            if b_col_final < 150: b_col_final = 150  # Mínimo peralte Art. 7.2.5
            A_c_final_col = b_col_final * t_col
            self.log_memoria(
                f"Gobernante A_c = {A_c_final_col:.0f} mm². b = {b_col:.0f} mm -> Se redondea a {b_col_final:.0f} mm",
                nivel=2)
            self.log_memoria(f"Dimensión Final Columna: {t_col:.0f} mm x {b_col_final:.0f} mm", nivel=2,
                             cita="E.070 Art. 7.2.3 & 7.2.5")

            # 4.2: Acero Vertical
            self.log_memoria("4.2. Acero Vertical (As) (Art. 8.6.3.a.2)", nivel=1, es_titulo=True)
            A_sf = Vc / (fy_mpa * mu * phi_v)
            self.log_memoria(f"Acero por Corte-Fricción A_sf = {A_sf:.2f} mm²", nivel=3)
            A_st = max(0, T) / (fy_mpa * 0.90)
            self.log_memoria(f"Acero por Tracción A_st = {A_st:.2f} mm²", nivel=3)
            A_s_calc = A_sf + A_st
            self.log_memoria(f"Acero Calculado (A_sf + A_st) = {A_s_calc:.2f} mm²", nivel=3)
            A_s_min_code_col = (0.1 * fc_mpa * A_c_final_col) / fy_mpa
            self.log_memoria(f"Acero Mínimo Código (0.1*f'c*Ac/fy) = {A_s_min_code_col:.2f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.a.2")
            self.log_memoria(f"Acero Mínimo Geométrico (4Ø3/8\") = {A_s_min_geom_col:.2f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.a.2")
            A_s_final_col = max(A_s_calc, A_s_min_code_col, A_s_min_geom_col)
            self.log_memoria(f"Acero Vertical Requerido (As): {A_s_final_col:.2f} mm²", nivel=2)

            # 4.3: Estribos
            self.log_memoria("4.3. Estribos de Confinamiento (Art. 8.6.3.a.3)", nivel=1, es_titulo=True)
            d_estribo_col = 8.0  # mm (E.070 Art. 7.2.6 / E.060 Art. 7.10.5.1)
            A_v_col = 2 * (math.pi * d_estribo_col ** 2 / 4)  # 2 ramas
            d_col, t_n = b_col_final, t_col - 2 * recub_mm
            A_c, A_n = A_c_final_col, (b_col_final - 2 * recub_mm) * t_n

            s1 = (A_v_col * fy_mpa) / (0.3 * t_n * fc_mpa * (A_c / A_n - 1)) if (A_c / A_n - 1) > 0 else float('inf')
            s2 = (A_v_col * fy_mpa) / (0.12 * t_n * fc_mpa)
            s3 = max(d_col / 4, 50.0)
            s4 = 100.0

            self.log_memoria(
                f"s1 = {s1:.1f} mm; s2 = {s2:.1f} mm; s3 = {s3:.1f} mm; s4 = {s4:.1f} mm (Cálculo Fórmulas 8.6.3-a.3)",
                nivel=3)
            s_conf_col = math.floor(min(s1, s2, s3, s4) / 25) * 25
            Lo_col = max(450.0, 1.5 * d_col)
            self.log_memoria(f"Espaciamiento 's' = {s_conf_col} mm. Longitud 'Lo' = {Lo_col} mm", nivel=2)
            self.log_memoria(
                f"Diseño Estribos: Ø{d_estribo_col}mm: 1@50, R@{s_conf_col} (en Lo={Lo_col}mm), R@250 (central)",
                nivel=2, cita="E.070 Art. 8.6.3.a.3")

            # --- PASO 5: DISEÑO DE VIGA SOLERA (NTE E.070, Art. 8.6.3.b) ---
            self.log_memoria("PASO 5: DISEÑO DE VIGA SOLERA (NTE E.070, Art. 8.6.3.b)", nivel=0, es_titulo=True)
            b_viga, h_viga = t_mm, params['h_solera_mm']

            # 5.1: Acero Longitudinal
            self.log_memoria(f"5.1. Acero Longitudinal (As) (b={b_viga} mm, h={h_viga} mm)", nivel=1, es_titulo=True)
            A_s_tension_viga = Ts / (0.90 * fy_mpa)
            self.log_memoria(f"Acero por Tracción As = Ts / (Φ * fy) = {A_s_tension_viga:.2f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.b")
            A_cs_viga = b_viga * h_viga
            A_s_min_code_viga = (0.1 * fc_mpa * A_cs_viga) / fy_mpa
            self.log_memoria(f"Acero Mínimo Código (0.1*f'c*Acs/fy) = {A_s_min_code_viga:.2f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.b")
            A_s_min_geom_viga = 4 * (math.pi * (9.53 ** 2) / 4)  # 4Ø3/8"
            self.log_memoria(f"Acero Mínimo Geométrico (4Ø3/8\") = {A_s_min_geom_viga:.2f} mm²", nivel=3,
                             cita="E.070 Art. 8.6.3.b")
            A_s_final_viga = max(A_s_tension_viga, A_s_min_code_viga, A_s_min_geom_viga)
            self.log_memoria(f"Acero Longitudinal Requerido (As): {A_s_final_viga:.2f} mm²", nivel=2)

            # 5.2: Estribos
            self.log_memoria("5.2. Estribos (Art. 8.6.3.b)", nivel=1, es_titulo=True)
            d_estribo_viga = 6.0
            detalle_estribos_viga = f"Ø{d_estribo_viga}mm: 1@50, 4@100, R@250 (Prescriptivo)"
            self.log_memoria(detalle_estribos_viga, nivel=2, cita="E.070 Art. 8.6.3.b")

            # --- PASO 6: GUARDAR RESULTADOS PARA GRÁFICOS ---
            self.log_memoria("PASO 6: GENERACIÓN DE GRÁFICOS", nivel=0, es_titulo=True)
            self.resultados_graficos = {
                "columna": {
                    "b": b_col_final, "t": t_col, "As": A_s_final_col, "d_est": d_estribo_col,
                    "s": s_conf_col, "Lo": Lo_col, "s_c": 250.0, "recub": recub_mm
                },
                "viga": {
                    "b": b_viga, "h": h_viga, "As": A_s_final_viga, "detalle": detalle_estribos_viga,
                    "d_est": d_estribo_viga, "recub": 40.0  # E.060 Art. 7.7.1
                }
            }
            self.log_memoria("Cálculo completado exitosamente.", nivel=2)
            return "\n".join(self.memoria_calculo), self.resultados_graficos

        except Exception as e:
            logger.error(f"Error fatal en el cálculo completo: {e}", exc_info=True)
            self.memoria_calculo.append(f"\n\n--- ERROR FATAL EN EL CÁLCULO ---\n{e}")
            return "\n".join(self.memoria_calculo), None


# --- LÓGICA DE GRÁFICOS (Matplotlib) ---

def plot_columna(b, t, A_s_total, d_estribo, s, Lo, s_central, recub=20.0):
    """Genera un gráfico esquemático de la sección de la columna."""
    if not MATPLOTLIB_AVAILABLE: return None

    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)

    # 1. Dibujar Concreto
    ax.add_patch(
        Rectangle((0, 0), b, t, facecolor='#d9d9d9', edgecolor='black', label=f"Columna: {b:.0f} x {t:.0f} mm"))

    # 2. Dibujar Estribo
    b_n = b - 2 * recub
    t_n = t - 2 * recub
    ax.add_patch(Rectangle((recub, recub), b_n, t_n, facecolor='none', edgecolor='#FF0000', linewidth=2,
                           label=f"Estribo Ø{d_estribo}mm"))

    # 3. Dibujar Acero Longitudinal
    d_barra_long = 12.7  # 1/2 pulgada (126.7 mm²)
    area_barra_long = math.pi * (d_barra_long ** 2) / 4
    if A_s_total <= (4 * 71):  # 4x3/8" = 284 mm^2
        d_barra_long = 9.53
        area_barra_long = math.pi * (d_barra_long ** 2) / 4

    n_barras = math.ceil(A_s_total / area_barra_long)
    n_barras = max(n_barras, 4)
    if n_barras % 2 != 0: n_barras += 1  # Hacer par

    A_s_provista = n_barras * area_barra_long
    n_por_lecho_largo = (n_barras // 2)  # Barras en los lados 'b'

    pos = []
    # Esquinas
    x1 = recub + d_estribo + d_barra_long / 2
    y1 = recub + d_estribo + d_barra_long / 2
    x2 = b - x1
    y2 = t - y1

    # Barras intermedias
    if n_por_lecho_largo > 0:
        esp_b = (x2 - x1) / max(1, n_por_lecho_largo - 1) if n_por_lecho_largo > 1 else 0
        for i in range(n_por_lecho_largo):
            x = x1 + i * esp_b
            pos.append((x, y1))  # Lecho inferior
            pos.append((x, y2))  # Lecho superior

    for p in pos:
        ax.add_patch(Circle(p, d_barra_long / 2, facecolor='#333333', edgecolor='black'))

    ax.set_xlabel(f"Dimensión b = {b:.0f} mm")
    ax.set_ylabel(f"Dimensión t = {t:.0f} mm")
    titulo = f"Columna: {n_barras}Ø{d_barra_long:.2f}mm (As={A_s_provista:.0f} mm²)\n"
    titulo += f"Estribos Ø{d_estribo:.0f}mm: 1@50, R@{s:.0f} (en Lo={Lo:.0f}mm), R@{s_central:.0f} (central)"
    ax.set_title(titulo, fontsize=9)
    ax.axis('equal')
    fig.tight_layout()
    return fig


def plot_viga(b, h, A_s_total, detalle_estribos, d_estribo=6.0, recub=40.0):
    """Genera un gráfico esquemático de la sección de la viga solera."""
    if not MATPLOTLIB_AVAILABLE: return None

    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)

    # 1. Concreto
    ax.add_patch(
        Rectangle((0, 0), b, h, facecolor='#d9d9d9', edgecolor='black', label=f"Viga Solera: {b:.0f} x {h:.0f} mm"))

    # 2. Estribo
    b_n = b - 2 * recub
    h_n = h - 2 * recub
    ax.add_patch(Rectangle((recub, recub), b_n, h_n, facecolor='none', edgecolor='#FF0000', linewidth=2,
                           label=f"Estribo Ø{d_estribo}mm"))

    # 3. Acero Longitudinal
    d_barra_long = 9.53  # 3/8" (71 mm²)
    area_barra_long = math.pi * (d_barra_long ** 2) / 4
    if A_s_total > (4 * 71):  # Si 4x3/8" no es suficiente
        d_barra_long = 12.7  # 1/2"
        area_barra_long = math.pi * (d_barra_long ** 2) / 4

    n_barras = math.ceil(A_s_total / area_barra_long)
    n_barras = max(n_barras, 4)
    if n_barras % 2 != 0: n_barras += 1

    A_s_provista = n_barras * area_barra_long
    n_por_lecho = n_barras // 2

    esp_b = (b - 2 * (recub + d_estribo + d_barra_long / 2)) / max(1, n_por_lecho - 1) if n_por_lecho > 1 else 0

    for i in range(n_por_lecho):
        x = (recub + d_estribo + d_barra_long / 2) + i * esp_b
        # Lecho inferior
        ax.add_patch(
            Circle((x, recub + d_estribo + d_barra_long / 2), d_barra_long / 2, facecolor='#333333', edgecolor='black'))
        # Lecho superior
        ax.add_patch(Circle((x, h - recub - d_estribo - d_barra_long / 2), d_barra_long / 2, facecolor='#333333',
                            edgecolor='black'))

    ax.set_xlabel(f"Ancho b = {b:.0f} mm")
    ax.set_ylabel(f"Peralte h = {h:.0f} mm")
    titulo = f"Viga Solera: {n_barras}Ø{d_barra_long:.2f}mm (As={A_s_provista:.0f} mm²)\n{detalle_estribos}"
    ax.set_title(titulo, fontsize=9)
    ax.axis('equal')
    fig.tight_layout()
    return fig


# --- Clases de la GUI (Tkinter) ---

class MainApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Valores de entrada del usuario (NTE E.030 y E.070)
        self.user_data = {
            "Z": 0.35, "U": 1.3, "S": 1.0,
            "f_c_mpa": 17.5,  # 175 kg/cm²
            "t_muro_cm": 13.0,
            "L_muro_m": 2.78,
            "h_muro_m": 2.3,
            "v_m_mpa": 0.8,  # Ajustado para King Kong Industrial (E.070 Tabla 9)
            "f_m_mpa": 6.4,  # Ajustado para King Kong Industrial (E.070 Tabla 9)
        }

        # Valores que ASUMIMOS (El ing. debe ingresar esto de su análisis)
        self.assumed_data = {
            "f_y_mpa": 420.0,
            "Tp": 0.4, "Tl": 3.0,  # Para Suelo S1 (E.030 Tabla 4)
            "R0": 3.0, "Ia": 1.0, "Ip": 1.0,
            "h_n": 6.9, "C_T": 60.0,
            "P_edificio": 5000.0,
            "P_piso1": 1800.0,
            "Pg_muro": 30.0,
            "Pc_col_kN": 15.0,
            "Ve1_muro": 25.0,
            "Me1_muro": 40.0,
            "h_solera_mm": 200.0,
            "Nc_cols": 2,
            "delta": 0.8,
            "mu": 0.8
        }

        self.title("Diseño de Albañilería Confinada (E.070) y Memoria de Cálculo")
        self.geometry("1100x850")
        self.designer = CalculadoraIngenieria()

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[10, 5], font=('Segoe UI', 10, 'bold'))
        style.configure('TLabelframe', padding=10)
        style.configure('TLabelframe.Label', font=('Segoe UI', 11, 'bold'), foreground='#004a99')
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=5)

        self.notebook = ttk.Notebook(self)

        # Pestañas
        self.tab1 = self.crear_tab1(self.notebook)
        self.tab2 = self.crear_tab2(self.notebook)
        self.tab3 = self.crear_tab3(self.notebook)

        self.notebook.add(self.tab1, text='PASO 1: Parámetros del Proyecto (E.030, E.070)')
        self.notebook.add(self.tab2, text='PASO 2: Metrados y Análisis (Entradas)')
        self.notebook.add(self.tab3, text='PASO 3: Resultados y Memoria de Cálculo')

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Cargar datos iniciales del usuario
        self.cargar_datos_iniciales()

    def crear_frame_entradas(self, parent, titulo, entradas):
        """Función helper para crear grids de labels y entries."""
        frame = ttk.LabelFrame(parent, text=titulo)
        frame.pack(side='top', fill='x', padx=10, pady=5)

        widgets = {}
        for i, (text, key, val) in enumerate(entradas):
            ttk.Label(frame, text=text).grid(row=i, column=0, padx=5, pady=4, sticky='e')
            widget = ttk.Entry(frame, width=15)
            widget.grid(row=i, column=1, padx=5, pady=4)
            widget.insert(0, str(val))
            widgets[key] = widget
        return widgets

    def crear_tab1(self, parent):
        """Parámetros de Sitio, Sísmicos y de Materiales"""
        tab = ttk.Frame(parent)

        # --- Datos Generales ---
        entradas_generales = [
            ("Nombre del Proyecto:", "proyecto", "Proyecto GAIATECH (EMAIRC VISIÓN)"),
            ("Ingeniero Responsable:", "ingeniero", "Emanuel Edgar Ancco Guaygua"),
            ("Muro a Diseñar (ID):", "muro_id", "Muro X-1 (Piso 1)"),
        ]
        self.inputs_gen = self.crear_frame_entradas(tab, "Información General", entradas_generales)

        # --- Parámetros E.030 ---
        entradas_e030 = [
            ("Factor de Zona (Z):", "Z", self.user_data['Z']),
            ("Factor de Uso (U):", "U", self.user_data['U']),
            ("Factor de Suelo (S):", "S", self.user_data['S']),
            ("Período Suelo (Tp) [s]:", "Tp", self.assumed_data['Tp']),
            ("Período Suelo (Tl) [s]:", "Tl", self.assumed_data['Tl']),
            ("Sistema Estructural (R0):", "R0", self.assumed_data['R0']),  # Albañilería
            ("Factor Irregularidad Altura (Ia):", "Ia", self.assumed_data['Ia']),
            ("Factor Irregularidad Planta (Ip):", "Ip", self.assumed_data['Ip']),
            ("Altura Total Edificio (h_n) [m]:", "h_n", self.assumed_data['h_n']),
            ("Coef. Período (C_T) (60=Albañilería):", "C_T", self.assumed_data['C_T']),
        ]
        self.inputs_e030 = self.crear_frame_entradas(tab, "Parámetros Sísmicos (NTE E.030)", entradas_e030)

        # --- Parámetros E.070 / E.060 ---
        entradas_mats = [
            ("f'c (Concreto) [MPa]:", "f_c_mpa", self.user_data['f_c_mpa']),
            ("f'y (Acero) [MPa]:", "f_y_mpa", self.assumed_data['f_y_mpa']),
            ("v'm (Corte Albañilería) [MPa]:", "v_m_mpa", self.user_data['v_m_mpa']),
            ("f'm (Compresión Albañilería) [MPa]:", "f_m_mpa", self.user_data['f_m_mpa']),
        ]
        self.inputs_mats = self.crear_frame_entradas(tab, "Propiedades de Materiales (NTE E.060 / E.070)",
                                                     entradas_mats)

        return tab

    def crear_tab2(self, parent):
        """Metrados de Carga y Resultados de Análisis Estructural"""
        tab = ttk.Frame(parent)

        # --- Geometría Muro ---
        entradas_geom = [
            ("Longitud del Muro (L) [m]:", "L_muro_m", self.user_data['L_muro_m']),
            ("Altura Libre Muro (h) [m]:", "h_muro_m", self.user_data['h_muro_m']),
            ("Espesor Muro (t) [cm]:", "t_muro_cm", self.user_data['t_muro_cm']),
            ("Peralte Viga Solera (h_solera) [mm]:", "h_solera_mm", self.assumed_data['h_solera_mm']),
            ("Longitud Paño Mayor (Lm) [m]:", "Lm_m", self.user_data['L_muro_m']),  # Asume Lm=L
            ("Número de Columnas (Nc):", "Nc_cols", self.assumed_data['Nc_cols']),
        ]
        self.inputs_geom = self.crear_frame_entradas(tab, "Geometría del Muro (Piso 1)", entradas_geom)

        # --- Metrados (Entrada Manual) ---
        entradas_metrados = [
            ("Peso Total Edificio (P_edificio) [kN]:", "P_edificio", self.assumed_data['P_edificio']),
            ("Peso Piso 1 (P_piso1) [kN]:", "P_piso1", self.assumed_data['P_piso1']),
            ("Peso Gravitacional Muro (Pg_muro) [kN]:", "Pg_muro", self.assumed_data['Pg_muro']),
            ("Peso Gravitacional Columna (Pc_col) [kN]:", "Pc_col_kN", self.assumed_data['Pc_col_kN']),
        ]
        self.inputs_metrados = self.crear_frame_entradas(tab, "Metrado de Cargas (Entrada Manual, NTE E.020)",
                                                         entradas_metrados)

        # --- Análisis (Entrada Manual) ---
        entradas_analisis = [
            ("Cortante Moderado (Ve1) en Muro [kN]:", "Ve1_muro", self.assumed_data['Ve1_muro']),
            ("Momento Moderado (Me1) en Muro [kN-m]:", "Me1_muro", self.assumed_data['Me1_muro']),
            ("Factor δ (Columna) [0.8 o 1.0]:", "delta", self.assumed_data['delta']),
            ("Factor μ (Fricción) [0.8 o 1.0]:", "mu", self.assumed_data['mu']),
        ]
        self.inputs_analisis = self.crear_frame_entradas(tab, "Fuerzas en el Muro (Entrada de Análisis Estructural)",
                                                         entradas_analisis)

        return tab

    def crear_tab3(self, parent):
        """Resultados, Memoria y Gráficos"""
        tab = ttk.Frame(parent)

        # Botones
        button_frame = ttk.Frame(tab)
        button_frame.pack(side='top', fill='x', padx=10, pady=5)

        calc_button = ttk.Button(button_frame, text="GENERAR DISEÑO Y MEMORIA DE CÁLCULO",
                                 command=self.ejecutar_calculo)
        calc_button.pack(side='left', expand=True, fill='x', padx=5)

        save_button = ttk.Button(button_frame, text="Guardar Memoria (.txt)", command=self.guardar_memoria)
        save_button.pack(side='left', expand=True, fill='x', padx=5)

        # Paneles de salida
        output_pane = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        output_pane.pack(expand=True, fill='both')

        # Panel Izquierdo (Memoria)
        memoria_frame = ttk.LabelFrame(output_pane, text="Memoria de Cálculo Detallada")
        self.memoria_widget = scrolledtext.ScrolledText(memoria_frame, state='disabled', wrap=tk.NONE,
                                                        font=('Consolas', 10))
        self.memoria_widget.pack(expand=True, fill='both', padx=5, pady=5)
        output_pane.add(memoria_frame, weight=50)

        # Panel Derecho (Gráficos)
        graficos_pane = ttk.PanedWindow(output_pane, orient=tk.VERTICAL)
        output_pane.add(graficos_pane, weight=50)

        self.col_graph_frame = ttk.LabelFrame(graficos_pane, text="Gráfico: Columna de Confinamiento")
        self.viga_graph_frame = ttk.LabelFrame(graficos_pane, text="Gráfico: Viga Solera")

        graficos_pane.add(self.col_graph_frame, weight=50)
        graficos_pane.add(self.viga_graph_frame, weight=50)

        # Guardar widgets de canvas para limpiarlos
        self.fig_canvas_col = None
        self.fig_canvas_viga = None

        # --- AÑADIR EL HANDLER DE LOGGING A LA GUI ---
        gui_handler = GuiLogHandler(self.memoria_widget)
        logger.addHandler(gui_handler)

        return tab

    def cargar_datos_iniciales(self):
        """Carga los datos del usuario en los widgets correspondientes. (CORREGIDO)"""
        try:
            self.inputs_e030['Z'].delete(0, tk.END);
            self.inputs_e030['Z'].insert(0, str(self.user_data.get('Z', 0.35)))
            self.inputs_e030['U'].delete(0, tk.END);
            self.inputs_e030['U'].insert(0, str(self.user_data.get('U', 1.3)))
            self.inputs_e030['S'].delete(0, tk.END);
            self.inputs_e030['S'].insert(0, str(self.user_data.get('S', 1.0)))

            self.inputs_mats['f_c_mpa'].delete(0, tk.END);
            self.inputs_mats['f_c_mpa'].insert(0, str(self.user_data.get('f_c_mpa', 17.5)))
            self.inputs_mats['v_m_mpa'].delete(0, tk.END);
            self.inputs_mats['v_m_mpa'].insert(0, str(self.user_data.get('v_m_mpa', 0.8)))
            self.inputs_mats['f_m_mpa'].delete(0, tk.END);
            self.inputs_mats['f_m_mpa'].insert(0, str(self.user_data.get('f_m_mpa', 6.4)))

            self.inputs_geom['t_muro_cm'].delete(0, tk.END);
            self.inputs_geom['t_muro_cm'].insert(0, str(self.user_data.get('t_muro_cm', 13.0)))
            self.inputs_geom['L_muro_m'].delete(0, tk.END);
            self.inputs_geom['L_muro_m'].insert(0, str(self.user_data.get('L_muro_m', 2.78)))
            self.inputs_geom['h_muro_m'].delete(0, tk.END);
            self.inputs_geom['h_muro_m'].insert(0, str(self.user_data.get('h_muro_m', 2.3)))
            self.inputs_geom['Lm_m'].delete(0, tk.END);
            self.inputs_geom['Lm_m'].insert(0, str(self.user_data.get('L_muro_m', 2.78)))  # Asumir Lm=L
        except Exception as e:
            logger.error(f"Error en cargar_datos_iniciales (Corregido): {e}", exc_info=True)

    def recolectar_entradas(self):
        """Recolecta todas las entradas de todas las pestañas."""
        params = {}
        all_inputs = [self.inputs_gen, self.inputs_e030, self.inputs_mats,
                      self.inputs_geom, self.inputs_metrados, self.inputs_analisis]

        for input_dict in all_inputs:
            for key, widget in input_dict.items():
                val = widget.get()
                if key in ['proyecto', 'ingeniero', 'muro_id']:
                    params[key] = val
                else:
                    try:
                        params[key] = float(val)
                    except ValueError:
                        raise ValueError(f"Error: La entrada '{key}' no es un número válido ('{val}').")

        # Validaciones cruzadas
        if params['t_muro_cm'] * 10 != params.get('t_muro_mm', 0):
            params['t_muro_mm'] = params['t_muro_cm'] * 10

        if params['L_muro_m'] != params['Lm_m'] and params['Nc_cols'] == 2:
            params['Lm_m'] = params['L_muro_m']
            self.inputs_geom['Lm_m'].delete(0, tk.END)
            self.inputs_geom['Lm_m'].insert(0, str(params['Lm_m']))

        return params

    def ejecutar_calculo(self):
        try:
            params = self.recolectar_entradas()

            # Limpiar salidas anteriores
            if self.fig_canvas_col: self.fig_canvas_col.get_tk_widget().destroy()
            if self.fig_canvas_viga: self.fig_canvas_viga.get_tk_widget().destroy()
            self.memoria_widget.configure(state='normal')
            self.memoria_widget.delete(1.0, tk.END)

            # Ejecutar cálculo
            memoria, graficos = self.designer.calcular_diseno_completo(params)

            # Mostrar Memoria
            # self.memoria_widget.insert(tk.END, memoria) # Esto es manejado por el GuiLogHandler
            self.memoria_widget.configure(state='disabled')

            # Mostrar Gráficos
            if MATPLOTLIB_AVAILABLE and graficos:
                g_col = graficos['columna']
                fig_col = plot_columna(
                    g_col['b'], g_col['t'], g_col['As'], g_col['d_est'],
                    g_col['s'], g_col['Lo'], g_col['s_c'], g_col['recub']
                )

                self.fig_canvas_col = FigureCanvasTkAgg(fig_col, master=self.col_graph_frame)
                self.fig_canvas_col.draw()
                self.fig_canvas_col.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                g_viga = graficos['viga']
                fig_viga = plot_viga(
                    g_viga['b'], g_viga['h'], g_viga['As'], g_viga['detalle'],
                    g_viga['d_est'], g_viga['recub']
                )

                self.fig_canvas_viga = FigureCanvasTkAgg(fig_viga, master=self.viga_graph_frame)
                self.fig_canvas_viga.draw()
                self.fig_canvas_viga.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            messagebox.showinfo("Cálculo Completo", "El diseño y la memoria de cálculo se han generado exitosamente.")
            self.notebook.select(self.tab3)

        except ValueError as e:
            messagebox.showerror("Error de Entrada", str(e))
            logger.error(f"Error de Validación: {e}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
            logger.error(f"Error en ejecución de cálculo: {e}", exc_info=True)

    def guardar_memoria(self):
        memoria_texto = self.memoria_widget.get(1.0, tk.END)
        if len(memoria_texto) < 200:  # Chequeo simple de que hay contenido
            messagebox.showwarning("Memoria Vacía", "Debe generar un cálculo antes de guardar.")
            return

        filename = filedialog.asksaveasfilename(
            title="Guardar Memoria de Cálculo",
            defaultextension=".txt",
            filetypes=[("Archivos de Texto", "*.txt"), ("Todos los Archivos", "*.*")],
            initialfile=f"Memoria_Calculo_{self.inputs_gen['muro_id'].get().replace(' ', '_')}.txt"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(memoria_texto)
                messagebox.showinfo("Éxito", f"Memoria guardada en:\n{filename}")
                logger.info(f"Memoria guardada en {filename}")
            except Exception as e:
                messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo: {e}")
                logger.error(f"Error al guardar memoria: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)
    except NameError:
        app_dir = os.getcwd()

    logger.info(f"Directorio de trabajo: {app_dir}")
    logger.info(f"Archivo de log: {os.path.join(app_dir, log_file_path)}")

    app = MainApp()
    app.mainloop()