# ==================================================================================================
# == SIMULADOR ESTRUCTURAL Y GESTOR DE PROYECTOS v8.0
# == Conforme a la Norma Técnica de Edificación E.060 (Perú)
# ==
# == Autor: Dr. Consultor en Robótica para Construcción (Versión Mejorada)
# ==
# == Descripción:
# == ... (resto de la descripción sin cambios)
# ==================================================================================================

import sys
import math
import logging
from datetime import datetime
from collections import defaultdict
import numpy as np
import pulp

# --- Dependencias de Terceros ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget, QTextEdit, QFrame,
    QMessageBox, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path

# --- Módulos de la Aplicación ---

# =============================================================================
# MÓDULO DE CONFIGURACIÓN Y CONSTANTES
# =============================================================================
LOG_FILE = f"simulador_estructural_E060_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler(sys.stdout)
                    ])

# Diámetros de barra en cm (NTE E.060)
BAR_DIAMETERS = {'3/8"': 0.95, '1/2"': 1.27, '5/8"': 1.59, '3/4"': 1.91, '1"': 2.54, '1 3/8"': 3.58}
BAR_AREAS = {k: math.pi * (v ** 2) / 4 for k, v in BAR_DIAMETERS.items()}

# Longitudes de empalme (ejemplo, deben ser calculadas según Capítulo 12 de la E.060)
SPLICE_LENGTHS = {'1/2"': 1.10, '5/8"': 1.30, '3/4"': 1.60, '1"': 2.10, '1 3/8"': 2.80}

# =============================================================================
# MÓDULO CORE: LÓGICA DE INGENIERÍA (NTE E.060)
# =============================================================================
try:
    import sectionproperties.pre.library as spl
    # Import CompoundGeometry explicitly if needed for combining list
    from sectionproperties.pre.geometry import CompoundGeometry
    from sectionproperties.pre.pre import Material
    from sectionproperties.analysis.section import Section
    from concreteproperties import (
        Concrete,
        Steel,
        ConcreteSection,
        add_bar_circular_array,
    )
    from concreteproperties.results import MomentInteractionResults
    from concreteproperties.stress_strain_profile import (
        RectangularStressBlock, SteelElasticPlastic, ConcreteLinear
    )

    CONCRETEPROPS_AVAILABLE = True
except ImportError:
    logging.error(
        "Librerías 'sectionproperties' o 'concreteproperties' no encontradas. El análisis de sección no funcionará.")
    CONCRETEPROPS_AVAILABLE = False


class MotorAnalisisE060:
    """
    Encapsula los cálculos de diseño para columnas de concreto armado
    siguiendo la Norma Técnica de Edificación E.060 de Perú.
    """

    def __init__(self, b, h, fc, fy, rec, acero_config):
        self.b = b
        self.h = h
        self.fc = fc
        self.fy = fy
        self.rec = rec
        self.acero_config = acero_config

        self.beta1 = self._calcular_beta1()
        self.Es = 2.0e6  # kg/cm^2
        self.ecu = 0.003

        logging.info(f"Motor de Análisis E.060 inicializado para columna {b}x{h}.")
        logging.info(f" -> Parámetros: f'c={fc}, fy={fy}, beta1={self.beta1:.3f}")

    def _calcular_beta1(self):
        """Calcula el factor beta1 según NTE E.060 (10.2.7.3)."""
        if self.fc <= 280:
            return 0.85
        else:
            val = 0.85 - 0.05 * (self.fc - 280) / 70
            return max(0.65, val)

    def generar_diagrama_interaccion(self):
        """
        Genera el diagrama de interacción P-M nominal y de diseño (con phi variable).
        """
        if not CONCRETEPROPS_AVAILABLE:
            raise RuntimeError("Librería 'concreteproperties' no está instalada.")

        # 1. Definir materiales para concreteproperties (unidades en MPa y mm)
        fc_mpa = self.fc / 10.197
        fy_mpa = self.fy / 10.197
        Es_mpa = self.Es / 10.197

        # Calcular Ec y crear perfil de servicio
        ec_mpa = 4700 * math.sqrt(fc_mpa)
        perfil_concreto_svc = ConcreteLinear(elastic_modulus=ec_mpa)

        # Definir el perfil de esfuerzos del concreto (Bloque Rectangular de Whitney)
        perfil_concreto_ult = RectangularStressBlock(
            compressive_strength=fc_mpa,
            alpha=0.85,
            gamma=self.beta1,
            ultimate_strain=self.ecu,
        )

        # Crear el material 'Concrete'
        try:
            concreto = Concrete(
                name="Concreto",
                density=2.4e-9,
                stress_strain_profile=perfil_concreto_svc,
                ultimate_stress_strain_profile=perfil_concreto_ult,
                flexural_tensile_strength=0,
                colour='#C0C0C0'
            )
        except Exception as e:
            logging.error(f"Error al inicializar 'Concrete' con la API moderna: {e}")
            raise RuntimeError(
                "No se pudo inicializar 'Concrete'. "
                f"Verifique la versión de 'concreteproperties'. Error: {e}"
            ) from e

        # Definir el perfil de esfuerzos del acero (Bilineal Elasto-plástico)
        perfil_acero = SteelElasticPlastic(
            yield_strength=fy_mpa,
            elastic_modulus=Es_mpa,
            fracture_strain=0.05,
        )

        # Crear el material 'Steel'
        acero = Steel(
            name="Acero de Refuerzo",
            density=7.85e-9,
            stress_strain_profile=perfil_acero,
            colour='#505050'
        )

        # 2. Crear la geometría del concreto (unidades en mm)
        b_mm, h_mm = self.b * 10, self.h * 10
        concrete_geom = spl.rectangular_section(d=h_mm, b=b_mm, material=concreto) # Renombrada para claridad

        # 3. Añadir el acero de refuerzo
        d_long_mm = BAR_DIAMETERS[self.acero_config['d_long']] * 10
        d_est_mm = BAR_DIAMETERS[self.acero_config['d_est']] * 10
        rec_mm = self.rec * 10

        cover = rec_mm + d_est_mm + d_long_mm / 2

        nx = self.acero_config['nx']
        ny = self.acero_config['ny']

        if nx < 2 or ny < 2:
            raise ValueError("El número de barras en cada dirección (nx, ny) debe ser al menos 2.")

        puntos_acero = []

        # Añadir barras de las caras superior e inferior
        x_coords = np.linspace(-b_mm / 2 + cover, b_mm / 2 - cover, nx)
        for x in x_coords:
            puntos_acero.append((x, h_mm / 2 - cover))
            puntos_acero.append((x, -h_mm / 2 - cover))

        # Añadir barras de las caras laterales (intermedias)
        if ny > 2:
            y_coords = np.linspace(-h_mm / 2 + cover, h_mm / 2 - cover, ny)[1:-1]
            for y in y_coords:
                puntos_acero.append((-b_mm / 2 + cover, y))
                puntos_acero.append((b_mm / 2 - cover, y))

        self.acero_config['puntos_acero_calculados'] = puntos_acero

        # --- CAMBIO: Crear lista de geometrías de barras y combinar al final ---
        n_bar_segments = 16
        bar_geometries = [] # Lista para guardar las geometrías de las barras individuales

        for pt in puntos_acero:
            # 1. Crear la geometría de la barra (círculo)
            single_bar_geom = spl.circular_section(
                d=d_long_mm,
                n=n_bar_segments,
                material=acero
            )
            # 2. Mover la barra a su posición (usando .shift())
            shifted_bar_geom = single_bar_geom.shift(x=pt[0], y=pt[1])

            # 3. Añadir la geometría de la barra movida a la lista
            bar_geometries.append(shifted_bar_geom)

        # 4. Combinar la geometría del concreto con TODAS las barras a la vez
        #    'concrete_geom' es solo el rectángulo de concreto.
        #    El operador '+' puede tomar una lista de geometrías.
        final_compound_geom = concrete_geom + bar_geometries

        # 5. Crear la malla sobre la geometría compuesta final
        final_compound_geom.create_mesh(mesh_sizes=[b_mm * h_mm / 20])

        # 6. Crear el ConcreteSection con la geometría mallada y compuesta
        conc_sec = ConcreteSection(final_compound_geom)
        # --- FIN DE SECCIÓN CORREGIDA ---


        # 4. Calcular el diagrama de interacción nominal
        results_nominal = conc_sec.moment_interaction_diagram(theta=0, n_points=64)

        # 5. Aplicar factor phi variable (NTE E.060 - 9.3.2)
        pn_nom_ton = np.array(results_nominal.p_n) / 1000 / 9.81  # de N a Ton-f
        mn_nom_ton_m = np.array(results_nominal.m_n) / 1e6 / 9.81  # de N-mm a Ton-m

        phi_vals = []
        ety = self.fy / self.Es

        # Aproximación: Usaremos la carga axial para determinar la zona
        po = 0.85 * self.fc * (
                self.b * self.h - len(puntos_acero) * BAR_AREAS[self.acero_config['d_long']]) + \
             len(puntos_acero) * BAR_AREAS[self.acero_config['d_long']] * self.fy
        po_ton = po / 1000

        # Pn_max (columnas con estribos)
        pn_max_ton = 0.80 * po_ton
        phi_pn_max_ton = 0.65 * pn_max_ton

        for pn in pn_nom_ton:
            if pn < 0:  # Tracción
                phi = 0.90
            elif pn < 0.1 * self.fc * self.b * self.h / 1000:  # Flexión casi pura
                phi = 0.90
            elif pn > pn_max_ton:  # Zona de compresión alta
                phi = 0.65
            else:
                # Interpolación lineal simplificada
                phi = 0.90 - 0.25 * (pn - 0.1 * self.fc * self.b * self.h / 1000) / \
                      (pn_max_ton - 0.1 * self.fc * self.b * self.h / 1000)
                phi = max(0.65, min(0.90, phi))
            phi_vals.append(phi)

        phi_vals = np.array(phi_vals)
        pn_dis_ton = pn_nom_ton * phi_vals
        mn_dis_ton_m = mn_nom_ton_m * phi_vals

        # Aplicar el límite superior de carga axial (NTE E.060 - 10.3.6)
        pn_nom_ton[pn_nom_ton > pn_max_ton] = pn_max_ton
        pn_dis_ton[pn_dis_ton > phi_pn_max_ton] = phi_pn_max_ton

        return {
            "pn_nom": pn_nom_ton, "mn_nom": mn_nom_ton_m,
            "pn_dis": pn_dis_ton, "mn_dis": mn_dis_ton_m,
            "phi_vals": phi_vals, "conc_sec": conc_sec
        }

    def diseno_por_corte(self, mn_nom_extremos, altura_libre):
        """
        Realiza el diseño a cortante por capacidad según NTE E.060 - Capítulo 21.
        """
        # ... (resto del método sin cambios) ...
        mpr_sup, mpr_inf = (1.25 * mn for mn in mn_nom_extremos)
        vu_ton = (mpr_sup + mpr_inf) / altura_libre
        vu_kg = vu_ton * 1000
        vc_kg = 0.53 * math.sqrt(self.fc) * self.b * self.h
        phi_vc_kg = 0.85 * vc_kg

        if vu_kg <= phi_vc_kg / 2:
            return {"requiere_refuerzo": False, "vu_ton": vu_ton, "phi_vc_ton": phi_vc_kg / 1000}

        vs_req_kg = (vu_kg / 0.85) - vc_kg
        if vs_req_kg < 0: vs_req_kg = 0

        area_estribo = BAR_AREAS[self.acero_config['d_est']]
        n_ramas = self.acero_config['nx']
        av = n_ramas * area_estribo
        s_cm = (av * self.fy * self.h) / vs_req_kg if vs_req_kg > 0 else 50

        lo = max(self.h, altura_libre * 100 / 6, 45)
        d_long_cm = BAR_DIAMETERS[self.acero_config['d_long']]
        d_est_cm = BAR_DIAMETERS[self.acero_config['d_est']]
        s_max_confinamiento = min(self.h / 4, 8 * d_long_cm, 24 * d_est_cm, 30)
        s_max_central = self.h / 2
        s_diseno_confinamiento = min(s_cm, s_max_confinamiento)
        s_diseno_central = min(s_cm, s_max_central)

        return {
            "requiere_refuerzo": True, "vu_ton": vu_ton, "phi_vc_ton": phi_vc_kg / 1000,
            "vs_req_ton": vs_req_kg / 1000, "s_calculado_cm": s_cm,
            "s_confinamiento_cm": s_diseno_confinamiento, "s_central_cm": s_diseno_central,
            "longitud_confinamiento_cm": lo
        }

# =============================================================================
# MÓDULO DE OPTIMIZACIÓN
# =============================================================================
class OptimizadorCorte:
    # ... (sin cambios) ...
    def __init__(self, stock_length=9.0):
        self.stock_length = stock_length
        logging.info(f"Optimizador de Corte inicializado con barras de {stock_length}m.")

    def _generate_patterns(self, piece_lengths):
        patterns = []
        sorted_pieces = sorted(list(set(p for p in piece_lengths if p <= self.stock_length)), reverse=True)
        if not sorted_pieces: return []
        for piece in sorted_pieces:
            patterns.append([piece] * int(self.stock_length / piece))
        for i in range(len(sorted_pieces)):
            current_pattern, remaining_space = [], self.stock_length
            for j in range(i, len(sorted_pieces)):
                piece = sorted_pieces[j]
                while remaining_space >= piece:
                    current_pattern.append(piece)
                    remaining_space -= piece
            if current_pattern:
                patterns.append(current_pattern)
        return [list(p) for p in set(tuple(sorted(p)) for p in patterns)]

    def solve_csp_with_ilp(self, demand):
        unique_piece_lengths = list(demand.keys())
        patterns = self._generate_patterns(unique_piece_lengths)
        if not patterns: return "No Patterns", 0, None
        model = pulp.LpProblem("Optimizacion_Corte_Acero", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("Patron", range(len(patterns)), lowBound=0, cat='Integer')
        model += pulp.lpSum(x[i] for i in range(len(patterns))), "Total_Barras_Utilizadas"
        piece_counts_in_pattern = [defaultdict(int) for _ in patterns]
        for i, p in enumerate(patterns):
            for piece_len in p:
                piece_counts_in_pattern[i][piece_len] += 1
        for piece_len, required_qty in demand.items():
            model += pulp.lpSum(piece_counts_in_pattern[i][piece_len] * x[i] for i in range(len(patterns))) >= required_qty, f"Constraint_{piece_len}"
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[model.status]
        if status == 'Optimal':
            plan = {i: {'patron': patterns[i], 'cantidad': int(round(pulp.value(x[i])))} for i in range(len(patterns)) if pulp.value(x[i]) > 0.1}
            return status, pulp.value(model.objective), plan
        return status, 0, None

# =============================================================================
# MÓDULO DE INTERFAZ GRÁFICA (PySide6)
# =============================================================================
class MatplotlibCanvas(FigureCanvas):
    # ... (sin cambios) ...
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

class DesignerWidget(QWidget):
    # ... (sin cambios) ...
    def __init__(self):
        super().__init__()
        self.last_analysis_results = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        controls_panel = QFrame()
        controls_panel.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls_panel)
        geom_frame = QFrame()
        geom_frame.setFrameShape(QFrame.Shape.StyledPanel)
        geom_layout = QGridLayout(geom_frame)
        geom_layout.addWidget(QLabel("<b>Geometría de la Columna</b>"), 0, 0, 1, 2)
        geom_layout.addWidget(QLabel("Ancho, b (cm):"), 1, 0)
        self.b_input = QLineEdit("30")
        geom_layout.addWidget(self.b_input, 1, 1)
        geom_layout.addWidget(QLabel("Peralte, h (cm):"), 2, 0)
        self.h_input = QLineEdit("50")
        geom_layout.addWidget(self.h_input, 2, 1)
        geom_layout.addWidget(QLabel("Altura Libre, Hn (m):"), 3, 0)
        self.hn_input = QLineEdit("3.0")
        geom_layout.addWidget(self.hn_input, 3, 1)
        geom_layout.addWidget(QLabel("Recubrimiento, rec (cm):"), 4, 0)
        self.rec_input = QLineEdit("4")
        geom_layout.addWidget(self.rec_input, 4, 1)
        mat_frame = QFrame()
        mat_frame.setFrameShape(QFrame.Shape.StyledPanel)
        mat_layout = QGridLayout(mat_frame)
        mat_layout.addWidget(QLabel("<b>Materiales</b>"), 0, 0, 1, 2)
        mat_layout.addWidget(QLabel("f'c (kg/cm²):"), 1, 0)
        self.fc_input = QLineEdit("280")
        mat_layout.addWidget(self.fc_input, 1, 1)
        mat_layout.addWidget(QLabel("fy (kg/cm²):"), 2, 0)
        self.fy_input = QLineEdit("4200")
        mat_layout.addWidget(self.fy_input, 2, 1)
        reinf_frame = QFrame()
        reinf_frame.setFrameShape(QFrame.Shape.StyledPanel)
        reinf_layout = QGridLayout(reinf_frame)
        reinf_layout.addWidget(QLabel("<b>Acero de Refuerzo</b>"), 0, 0, 1, 2)
        reinf_layout.addWidget(QLabel("Ø Long.:"), 1, 0)
        self.d_long_combo = QComboBox()
        self.d_long_combo.addItems(BAR_DIAMETERS.keys())
        self.d_long_combo.setCurrentText("5/8\"")
        reinf_layout.addWidget(self.d_long_combo, 1, 1)
        reinf_layout.addWidget(QLabel("Ø Estribo:"), 2, 0)
        self.d_est_combo = QComboBox()
        self.d_est_combo.addItems(BAR_DIAMETERS.keys())
        self.d_est_combo.setCurrentText("3/8\"")
        reinf_layout.addWidget(self.d_est_combo, 2, 1)
        reinf_layout.addWidget(QLabel("Barras en dir. X (nx):"), 3, 0)
        self.nx_input = QLineEdit("4")
        reinf_layout.addWidget(self.nx_input, 3, 1)
        reinf_layout.addWidget(QLabel("Barras en dir. Y (ny):"), 4, 0)
        self.ny_input = QLineEdit("2")
        reinf_layout.addWidget(self.ny_input, 4, 1)
        demand_frame = QFrame()
        demand_frame.setFrameShape(QFrame.Shape.StyledPanel)
        demand_layout = QGridLayout(demand_frame)
        demand_layout.addWidget(QLabel("<b>Cargas de Demanda (Diseño Último)</b>"), 0, 0, 1, 2)
        demand_layout.addWidget(QLabel("Carga Axial, Pu (Ton):"), 1, 0)
        self.pu_input = QLineEdit("120")
        demand_layout.addWidget(self.pu_input, 1, 1)
        demand_layout.addWidget(QLabel("Momento Flector, Mu (Ton-m):"), 2, 0)
        self.mu_input = QLineEdit("35")
        demand_layout.addWidget(self.mu_input, 2, 1)
        seismic_frame = QFrame()
        seismic_frame.setFrameShape(QFrame.Shape.StyledPanel)
        seismic_layout = QGridLayout(seismic_frame)
        seismic_layout.addWidget(QLabel("<b>Verificación Sísmica (E.060 21.6.2)</b>"), 0, 0, 1, 2)
        seismic_layout.addWidget(QLabel("Σ Mnv (Ton-m):"), 1, 0)
        self.mnv_input = QLineEdit("55")
        seismic_layout.addWidget(self.mnv_input, 1, 1)
        self.analyze_button = QPushButton("1. Analizar Flexo-Compresión")
        self.analyze_button.clicked.connect(self.run_flexo_compression_analysis)
        self.shear_button = QPushButton("2. Diseñar por Cortante")
        self.shear_button.clicked.connect(self.run_shear_design)
        self.shear_button.setEnabled(False)
        controls_layout.addWidget(geom_frame)
        controls_layout.addWidget(mat_frame)
        controls_layout.addWidget(reinf_frame)
        controls_layout.addWidget(demand_frame)
        controls_layout.addWidget(seismic_frame)
        controls_layout.addWidget(self.analyze_button)
        controls_layout.addWidget(self.shear_button)
        controls_layout.addStretch()
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        self.interaction_canvas = MatplotlibCanvas(self, width=8, height=6, dpi=100)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        results_layout.addWidget(self.interaction_canvas)
        results_layout.addWidget(self.results_text)
        main_layout.addWidget(controls_panel)
        main_layout.addWidget(results_panel)

    def _get_inputs(self):
        try:
            inputs = {
                "b": float(self.b_input.text()), "h": float(self.h_input.text()),
                "hn": float(self.hn_input.text()), "rec": float(self.rec_input.text()),
                "fc": float(self.fc_input.text()), "fy": float(self.fy_input.text()),
                "acero_config": {
                    "d_long": self.d_long_combo.currentText(), "d_est": self.d_est_combo.currentText(),
                    "nx": int(self.nx_input.text()), "ny": int(self.ny_input.text()),
                },
                "pu": float(self.pu_input.text()), "mu": float(self.mu_input.text()),
                "sum_mnv": float(self.mnv_input.text()),
            }
            return inputs
        except ValueError as e:
            QMessageBox.critical(self, "Error de Entrada", f"Valor inválido en los campos de entrada: {e}")
            return None

    @Slot()
    def run_flexo_compression_analysis(self):
        inputs = self._get_inputs()
        if not inputs: return
        try:
            motor = MotorAnalisisE060(b=inputs['b'], h=inputs['h'], fc=inputs['fc'], fy=inputs['fy'], rec=inputs['rec'], acero_config=inputs['acero_config'])
            self.last_analysis_results = motor.generar_diagrama_interaccion()
            self.last_analysis_results['motor'] = motor
            self.last_analysis_results['inputs'] = inputs
            self.plot_interaction_diagram()
            self.update_results_text_flexion()
            self.shear_button.setEnabled(True)
        except Exception as e:
            logging.error(f"Error en análisis de flexo-compresión: {e}", exc_info=True)
            QMessageBox.critical(self, "Error de Análisis", f"Ocurrió un error durante el análisis:\n{e}")
            self.shear_button.setEnabled(False)

    @Slot()
    def run_shear_design(self):
        if not self.last_analysis_results:
            QMessageBox.warning(self, "Acción Requerida", "Primero debe ejecutar el análisis de flexo-compresión.")
            return
        try:
            motor = self.last_analysis_results['motor']
            inputs = self.last_analysis_results['inputs']
            mn_max = np.max(self.last_analysis_results['mn_nom'])
            shear_results = motor.diseno_por_corte(mn_nom_extremos=(mn_max, mn_max), altura_libre=inputs['hn'])
            self.last_analysis_results['shear_results'] = shear_results
            self.update_results_text_shear()
        except Exception as e:
            logging.error(f"Error en diseño por cortante: {e}", exc_info=True)
            QMessageBox.critical(self, "Error de Diseño", f"Ocurrió un error durante el diseño por cortante:\n{e}")

    def plot_interaction_diagram(self):
        res = self.last_analysis_results
        inputs = res['inputs']
        ax = self.interaction_canvas.axes
        ax.clear()
        ax.plot(res['mn_nom'], res['pn_nom'], 'k--', label='Capacidad Nominal ($P_n$-$M_n$)')
        ax.plot(-res['mn_nom'], res['pn_nom'], 'k--')
        ax.plot(res['mn_dis'], res['pn_dis'], 'r-', linewidth=2, label='Capacidad de Diseño ($\phi P_n$-$\phi M_n$)')
        ax.plot(-res['mn_dis'], res['pn_dis'], 'r-', linewidth=2)
        ax.plot(inputs['mu'], inputs['pu'], 'bo', markersize=10, label=f"Demanda ($P_u, M_u$)")
        design_path = Path(np.column_stack([np.concatenate([res['mn_dis'], -res['mn_dis'][::-1]]), np.concatenate([res['pn_dis'], res['pn_dis'][::-1]])]))
        is_safe = design_path.contains_point((inputs['mu'], inputs['pu']))
        if is_safe: ax.set_title('Diagrama de Interacción P-M: DISEÑO CONFORME', color='green', fontsize=12, weight='bold')
        else: ax.set_title('Diagrama de Interacción P-M: DISEÑO NO CONFORME', color='red', fontsize=12, weight='bold')
        ax.set_xlabel('Momento Flector (Ton-m)')
        ax.set_ylabel('Carga Axial (Ton)')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        self.interaction_canvas.draw()
        self.last_analysis_results['is_safe_flexion'] = is_safe

    def update_results_text_flexion(self):
        res = self.last_analysis_results
        inputs = res['inputs']
        mn_max_columna = np.max(res['mn_nom'])
        sum_mnc = 2 * mn_max_columna
        sum_mnv_req = 1.2 * inputs['sum_mnv']
        check_cfvd = sum_mnc >= sum_mnv_req
        total_barras = len(self.last_analysis_results['motor'].acero_config['puntos_acero_calculados'])
        area_acero = total_barras * BAR_AREAS[inputs['acero_config']['d_long']]
        area_gruesa = inputs['b'] * inputs['h']
        cuantia = area_acero / area_gruesa
        check_cuantia = 0.01 <= cuantia <= 0.06
        text = "--- ANÁLISIS DE FLEXO-COMPRESIÓN ---\n"
        text += f"Estado: {'CUMPLE' if res['is_safe_flexion'] else 'NO CUMPLE'}\n\n"
        text += "--- VERIFICACIÓN SÍSMICA (E.060 Cap. 21) ---\n"
        text += "1. Columna Fuerte - Viga Débil (21.6.2.2):\n"
        text += f"   ΣMnc = 2 * {mn_max_columna:.2f} = {sum_mnc:.2f} Ton-m\n"
        text += f"   1.2 * ΣMnv = 1.2 * {inputs['sum_mnv']:.2f} = {sum_mnv_req:.2f} Ton-m\n"
        text += f"   Estado: {'CUMPLE' if check_cfvd else 'NO CUMPLE - Aumentar resistencia de la columna'}\n\n"
        text += "2. Cuantía de Acero Longitudinal (21.6.3.1):\n"
        text += f"   Número total de barras: {total_barras}\n"
        text += f"   Área de Acero: {area_acero:.2f} cm²\n"
        text += f"   Cuantía (ρg): {cuantia:.4f} ({cuantia * 100:.2f}%)\n"
        text += f"   Límites: 0.01 ≤ ρg ≤ 0.06\n"
        text += f"   Estado: {'CUMPLE' if check_cuantia else 'NO CUMPLE - Ajustar cantidad de acero'}\n\n"
        self.results_text.setText(text)

    def update_results_text_shear(self):
        current_text = self.results_text.toPlainText()
        shear_res = self.last_analysis_results['shear_results']
        text = "\n--- DISEÑO POR CORTANTE (POR CAPACIDAD) ---\n"
        text += f"Cortante de Diseño por Capacidad (Vu): {shear_res['vu_ton']:.2f} Ton\n"
        text += f"Resistencia al Cortante del Concreto (φVc): {shear_res['phi_vc_ton']:.2f} Ton\n"
        if not shear_res['requiere_refuerzo']:
            text += "No se requiere refuerzo por cortante por cálculo, pero se debe colocar el mínimo normativo.\n"
        else:
            text += f"Cortante a resistir por el acero (Vs): {shear_res['vs_req_ton']:.2f} Ton\n"
            text += f"Espaciamiento calculado (s): {shear_res['s_calculado_cm']:.1f} cm\n\n"
            text += "DISTRIBUCIÓN DE ESTRIBOS RECOMENDADA:\n"
            text += f"Longitud de Confinamiento (lo): {shear_res['longitud_confinamiento_cm']:.1f} cm en cada extremo.\n"
            s_conf = math.floor(shear_res['s_confinamiento_cm'] / 2.5) * 2.5
            s_cent = math.floor(shear_res['s_central_cm'] / 5) * 5
            text += f"  - Zona de Confinamiento: 1 @ 5, resto @ {s_conf:.1f} cm\n"
            text += f"  - Zona Central: @ {s_cent:.1f} cm\n"
        self.results_text.setText(current_text + text)


class MainWindow(QMainWindow):
    # ... (sin cambios) ...
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador Estructural y Gestor de Proyectos v8.0 (NTE E.060)")
        self.setGeometry(100, 100, 1600, 900)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.designer_tab = DesignerWidget()
        self.tabs.addTab(self.designer_tab, "Diseñador de Columnas (E.060)")
        self.tabs.addTab(QWidget(), "Optimizador de Corte de Acero")
        self.tabs.addTab(QWidget(), "Seguimiento de Avances en Obra")
        logging.info("Aplicación iniciada con éxito.")

# =============================================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# =============================================================================
if __name__ == "__main__":
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())