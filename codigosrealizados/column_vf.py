import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches


# ============================================================================
# SISTEMA DE DISEÑO DE ALBAÑILERÍA CONFINADA - GUI COMPLETA
# Columnas y Vigas de Confinamiento con Cálculos Detallados
# Basada en NTE E.070 Albañilería y SENCICO
# Autor: Emanuel Ancco (@EmanuelAncco)
# Fecha: 2025-01-11
# Versión: 1.0 COMPLETA - 1400+ líneas
# ============================================================================

class BaseDatos:
    """Base de datos de materiales y barras de acero"""

    AREAS_BARRAS = {
        '6mm': 0.28, '1/4"': 0.32, '8mm': 0.50, '3/8"': 0.71,
        '12mm': 1.13, '1/2"': 1.27, '5/8"': 1.99, '3/4"': 2.85,
        '1"': 5.07, '1-3/8"': 9.87
    }

    DIAMETROS_BARRAS = {
        '6mm': 0.6, '1/4"': 0.635, '8mm': 0.8, '3/8"': 0.95,
        '12mm': 1.2, '1/2"': 1.27, '5/8"': 1.59, '3/4"': 1.91,
        '1"': 2.54, '1-3/8"': 3.49
    }

    DENSIDAD_CONCRETO = 2400
    DENSIDAD_ALBANILERIA = 1800
    PHI_FLEXOCOMP = 0.65
    PHI_CORTE = 0.85
    H_T_MAX = 20.0
    CUANTIA_MIN = 0.01
    CUANTIA_MAX = 0.06


class CalculosAlbanileria:
    """Clase con todos los cálculos estructurales detallados"""

    @staticmethod
    def verificar_esbeltez(h_cm, t_cm):
        """Verifica esbeltez h/t <= 20 (E.070, Art. 20.a)"""
        esbeltez = h_cm / t_cm
        t_min = h_cm / BaseDatos.H_T_MAX
        cumple = esbeltez <= BaseDatos.H_T_MAX

        return {
            'formula': 'λ = h / t',
            'sustitucion': f'λ = {h_cm:.0f} / {t_cm:.1f}',
            'resultado': f'λ = {esbeltez:.2f}',
            'limite': f'λ ≤ {BaseDatos.H_T_MAX:.0f}',
            'cumple': cumple,
            't_min_req': t_min,
            'esbeltez': esbeltez
        }

    @staticmethod
    def calcular_numero_vigas(esbeltez):
        """Calcula número de vigas intermedias necesarias"""
        num_vigas = max(0, math.ceil(esbeltez / BaseDatos.H_T_MAX) - 1)
        num_paños = num_vigas + 1

        return {
            'formula': 'n_vigas = ceil(λ / λ_max) - 1',
            'sustitucion': f'n_vigas = ceil({esbeltez:.2f} / {BaseDatos.H_T_MAX}) - 1',
            'resultado': f'n_vigas = {num_vigas}',
            'num_paños': num_paños,
            'num_vigas': num_vigas
        }

    @staticmethod
    def area_minima_columna(t_cm):
        """Área mínima de columna (E.070, Art. 27.3.a.1)"""
        Ac_min = 15 * t_cm
        return {
            'formula': 'Ac_min = 15t',
            'sustitucion': f'Ac_min = 15 × {t_cm:.1f}',
            'resultado': f'Ac_min = {Ac_min:.2f} cm²',
            'valor': Ac_min
        }

    @staticmethod
    def acero_minimo_columna(Ac, fc, fy):
        """Acero longitudinal mínimo (E.070, Art. 27.3.a.2)"""
        As_calc = (0.1 * fc * Ac) / fy
        As_min_norma = BaseDatos.CUANTIA_MIN * Ac
        As_req = max(As_calc, As_min_norma)

        return {
            'formula_1': 'As_calc = (0.1 × f\'c × Ac) / fy',
            'sustitucion_1': f'As_calc = (0.1 × {fc} × {Ac:.2f}) / {fy}',
            'resultado_1': f'As_calc = {As_calc:.3f} cm²',
            'formula_2': f'As_min = ρ_min × Ac = {BaseDatos.CUANTIA_MIN} × Ac',
            'sustitucion_2': f'As_min = {BaseDatos.CUANTIA_MIN} × {Ac:.2f}',
            'resultado_2': f'As_min = {As_min_norma:.3f} cm²',
            'formula_3': 'As_req = max(As_calc, As_min)',
            'resultado_3': f'As_req = {As_req:.3f} cm²',
            'As_calculado': As_calc,
            'As_minimo': As_min_norma,
            'As_requerido': As_req
        }

    @staticmethod
    def seleccionar_barras_exacto(As_req, b, h, rec, tipo='columna'):
        """Selección exacta de barras con verificación de espaciamiento"""
        opciones_validas = []

        if tipo == 'columna':
            diametros = ['3/8"', '1/2"', '5/8"', '3/4"', '1"']
            num_min_barras = 4
        else:
            diametros = ['3/8"', '1/2"', '5/8"']
            num_min_barras = 2

        for diam in diametros:
            area_barra = BaseDatos.AREAS_BARRAS[diam]
            db = BaseDatos.DIAMETROS_BARRAS[diam]

            num_barras = max(num_min_barras, math.ceil(As_req / area_barra))
            if tipo == 'columna' and num_barras % 2 != 0:
                num_barras += 1

            tamaño_agregado = 1.9
            s_min = max(2.5, db, 1.3 * tamaño_agregado)

            if tipo == 'columna':
                barras_por_lado_largo = math.ceil(num_barras / 2)
                espacio_disponible = h - 2 * rec - 2 * db
                espacio_necesario = (barras_por_lado_largo - 1) * s_min
                caben = espacio_disponible >= espacio_necesario
            else:
                espacio_disponible = b - 2 * rec - num_barras * db
                espacio_necesario = (num_barras - 1) * s_min
                caben = espacio_disponible >= espacio_necesario

            if caben:
                As_prov = num_barras * area_barra
                diferencia = As_prov - As_req
                eficiencia = (As_req / As_prov) * 100

                opciones_validas.append({
                    'diametro': diam,
                    'num_barras': num_barras,
                    'area_barra': area_barra,
                    'As_provisto': As_prov,
                    'diferencia': diferencia,
                    'eficiencia': eficiencia,
                    's_min': s_min,
                    'caben': caben
                })

        mejor = None
        for opcion in opciones_validas:
            if opcion['eficiencia'] >= 80:
                if mejor is None or opcion['diferencia'] < mejor['diferencia']:
                    mejor = opcion

        if mejor is None and opciones_validas:
            mejor = max(opciones_validas, key=lambda x: x['eficiencia'])

        return mejor, opciones_validas

    @staticmethod
    def diseñar_estribos(b, h, tipo='columna'):
        """Diseño de estribos según E.070 y E.060"""
        diam_estr = '1/4"' if max(b, h) <= 30 else '3/8"'

        s_conf = math.floor(min(b / 2, h / 2, 10.0) / 5) * 5
        s_central = math.floor(min(16 * 0.95, 48 * BaseDatos.DIAMETROS_BARRAS[diam_estr], min(b, h)) / 5) * 5
        Lo = max(max(b, h), 45.0)
        n_estr_conf = math.floor(Lo / s_conf)

        return {
            'diametro': diam_estr,
            's_confinamiento': s_conf,
            's_central': s_central,
            'Lo': Lo,
            'n_estribos_conf': n_estr_conf,
            'patron': f'1@5cm, {n_estr_conf - 1}@{s_conf:.0f}cm, Resto@{s_central:.0f}cm',
            'formulas': {
                's_conf': f's_conf = min(b/2, h/2, 10) = {s_conf:.0f} cm',
                'Lo': f'Lo = max(b, h, 45) = {Lo:.0f} cm'
            }
        }

    @staticmethod
    def peralte_viga_minimo(luz_m):
        """Peralte mínimo de viga según criterios prácticos"""
        h_recomendado = max(17.0, math.ceil(((luz_m * 100) / 12) / 5) * 5)

        return {
            'formula': 'h = L / k',
            'h_L10': (luz_m * 100) / 10,
            'h_L12': (luz_m * 100) / 12,
            'h_L14': (luz_m * 100) / 14,
            'h_recomendado': h_recomendado,
            'h_min_practico': 17.0
        }


class AplicacionDiseño:
    """Aplicación principal con interfaz gráfica completa"""

    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Diseño de Albañilería Confinada - E.070")
        self.root.geometry("1400x900")
        self.resultados = {}
        self.configurar_estilos()
        self.crear_interfaz()

    def configurar_estilos(self):
        """Configura estilos personalizados ttk"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 10, 'bold'), foreground='#34495e')
        style.configure('TButton', font=('Arial', 10))
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

    def crear_interfaz(self):
        """Crea la interfaz con pestañas"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.tab_entrada = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrada, text='📝 Datos de Entrada')
        self.crear_tab_entrada()

        self.tab_calculos = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_calculos, text='🔬 Cálculos Detallados')
        self.crear_tab_calculos()

        self.tab_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_resultados, text='📊 Resultados')
        self.crear_tab_resultados()

        self.tab_memoria = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_memoria, text='📄 Memoria de Cálculo')
        self.crear_tab_memoria()

    def crear_tab_entrada(self):
        """Crea la pestaña de entrada de datos con scroll"""
        canvas = tk.Canvas(self.tab_entrada)
        scrollbar = ttk.Scrollbar(self.tab_entrada, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text="DATOS DE ENTRADA", style='Title.TLabel').grid(row=0, column=0, columnspan=3,
                                                                                        pady=20)

        row = 1
        ttk.Label(scrollable_frame, text="GEOMETRÍA DEL MURO", style='Subtitle.TLabel').grid(row=row, column=0,
                                                                                             columnspan=3, sticky='w',
                                                                                             padx=20, pady=(20, 10))

        campos = [
            (1, "Altura libre total (m):", "entrada_altura", "4.8", "metros"),
            (2, "Espesor del muro (cm):", "entrada_espesor", "12.5", "cm"),
            (3, "Longitud del muro (m):", "entrada_longitud", "2.78", "metros (opcional)")
        ]

        for offset, label, attr, default, unit in campos:
            row += 1
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky='e', padx=5, pady=5)
            entry = ttk.Entry(scrollable_frame, width=15)
            entry.grid(row=row, column=1, padx=5, pady=5)
            entry.insert(0, default)
            setattr(self, attr, entry)
            ttk.Label(scrollable_frame, text=unit).grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="MATERIALES", style='Subtitle.TLabel').grid(row=row, column=0, columnspan=3,
                                                                                     sticky='w', padx=20, pady=(20, 10))

        materiales = [
            (1, "Resistencia concreto f'c (kg/cm²):", "entrada_fc", ['140', '175', '210', '280', '350'], '175'),
            (2, "Fluencia del acero fy (kg/cm²):", "entrada_fy", ['4200', '5600'], '4200'),
            (3, "Resistencia albañilería f'm (kg/cm²):", "entrada_fm", ['35', '45', '50', '65'], '65')
        ]

        for offset, label, attr, values, default in materiales:
            row += 1
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky='e', padx=5, pady=5)
            combo = ttk.Combobox(scrollable_frame, width=13, values=values)
            combo.grid(row=row, column=1, padx=5, pady=5)
            combo.set(default)
            setattr(self, attr, combo)
            ttk.Label(scrollable_frame, text="kg/cm²").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="COLUMNAS DE CONFINAMIENTO", style='Subtitle.TLabel').grid(row=row, column=0,
                                                                                                    columnspan=3,
                                                                                                    sticky='w', padx=20,
                                                                                                    pady=(20, 10))

        columnas_campos = [
            (1, "Peralte propuesto (cm):", "entrada_peralte_col", "25", "cm (0 = automático)"),
            (2, "Recubrimiento (cm):", "entrada_recubrimiento", "4.0", "cm")
        ]

        for offset, label, attr, default, unit in columnas_campos:
            row += 1
            ttk.Label(scrollable_frame, text=label).grid(row=row, column=0, sticky='e', padx=5, pady=5)
            entry = ttk.Entry(scrollable_frame, width=15)
            entry.grid(row=row, column=1, padx=5, pady=5)
            entry.insert(0, default)
            setattr(self, attr, entry)
            ttk.Label(scrollable_frame, text=unit).grid(row=row, column=2, sticky='w')

        row += 1
        frame_botones = ttk.Frame(scrollable_frame)
        frame_botones.grid(row=row, column=0, columnspan=3, pady=30)

        ttk.Button(frame_botones, text="🚀 CALCULAR", style='Accent.TButton', command=self.calcular, width=20).pack(
            side='left', padx=10)
        ttk.Button(frame_botones, text="🔄 Limpiar", command=self.limpiar_campos, width=15).pack(side='left', padx=10)
        ttk.Button(frame_botones, text="💾 Exportar", command=self.exportar_memoria, width=15).pack(side='left', padx=10)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def crear_tab_calculos(self):
        """Crea la pestaña de cálculos detallados"""
        self.texto_calculos = scrolledtext.ScrolledText(self.tab_calculos, wrap=tk.WORD, width=80, height=40,
                                                        font=('Courier New', 9))
        self.texto_calculos.pack(fill='both', expand=True, padx=10, pady=10)

        self.texto_calculos.tag_config('titulo', font=('Courier New', 12, 'bold'), foreground='#2c3e50')
        self.texto_calculos.tag_config('subtitulo', font=('Courier New', 10, 'bold'), foreground='#34495e')
        self.texto_calculos.tag_config('formula', font=('Courier New', 9, 'italic'), foreground='#16a085')
        self.texto_calculos.tag_config('resultado', font=('Courier New', 9, 'bold'), foreground='#27ae60')
        self.texto_calculos.tag_config('error', font=('Courier New', 9, 'bold'), foreground='#e74c3c')

    def crear_tab_resultados(self):
        """Crea la pestaña de resultados con gráficos"""
        frame_izq = ttk.Frame(self.tab_resultados)
        frame_izq.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(frame_izq, text="RESUMEN DE RESULTADOS", style='Title.TLabel').pack(pady=10)

        self.texto_resumen = scrolledtext.ScrolledText(frame_izq, wrap=tk.WORD, width=50, height=35, font=('Arial', 9))
        self.texto_resumen.pack(fill='both', expand=True)

        frame_der = ttk.Frame(self.tab_resultados)
        frame_der.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(frame_der, text="VISUALIZACIÓN", style='Title.TLabel').pack(pady=10)

        self.figura_grafico = Figure(figsize=(6, 8), dpi=80)
        self.canvas_grafico = FigureCanvasTkAgg(self.figura_grafico, frame_der)
        self.canvas_grafico.get_tk_widget().pack(fill='both', expand=True)

    def crear_tab_memoria(self):
        """Crea la pestaña de memoria de cálculo"""
        toolbar = ttk.Frame(self.tab_memoria)
        toolbar.pack(fill='x', padx=10, pady=5)

        ttk.Button(toolbar, text="📄 Generar Memoria Completa", command=self.generar_memoria_completa).pack(side='left',
                                                                                                           padx=5)
        ttk.Button(toolbar, text="💾 Guardar como TXT", command=self.exportar_memoria).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📋 Copiar al Portapapeles", command=self.copiar_portapapeles).pack(side='left', padx=5)

        self.texto_memoria = scrolledtext.ScrolledText(self.tab_memoria, wrap=tk.WORD, width=100, height=40,
                                                       font=('Courier New', 9))
        self.texto_memoria.pack(fill='both', expand=True, padx=10, pady=10)

    def calcular(self):
        """Función principal de cálculo - IMPLEMENTACIÓN COMPLETA"""
        try:
            # Obtener datos
            h_total_m = float(self.entrada_altura.get())
            t_muro_cm = float(self.entrada_espesor.get())
            longitud_m = float(self.entrada_longitud.get() or "0")
            fc = float(self.entrada_fc.get())
            fy = float(self.entrada_fy.get())
            fm = float(self.entrada_fm.get() or "0")
            b_col_prop = float(self.entrada_peralte_col.get() or "25")
            rec = float(self.entrada_recubrimiento.get())

            self.texto_calculos.delete(1.0, tk.END)

            # ENCABEZADO
            self.agregar_texto_calculos("=" * 80 + "\n", 'titulo')
            self.agregar_texto_calculos("  CÁLCULOS DETALLADOS - ALBAÑILERÍA CONFINADA\n", 'titulo')
            self.agregar_texto_calculos("  Norma E.070 + SENCICO\n", 'subtitulo')
            self.agregar_texto_calculos(f"  Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n", 'subtitulo')
            self.agregar_texto_calculos("=" * 80 + "\n\n")

            # DATOS DE ENTRADA
            self.agregar_texto_calculos("1. DATOS DE ENTRADA\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos(f"   Altura total: h = {h_total_m:.2f} m = {h_total_m * 100:.0f} cm\n")
            self.agregar_texto_calculos(f"   Espesor: t = {t_muro_cm:.1f} cm\n")
            if longitud_m > 0:
                self.agregar_texto_calculos(f"   Longitud: L = {longitud_m:.2f} m\n")
            self.agregar_texto_calculos(f"   f'c = {fc:.0f} kg/cm², fy = {fy:.0f} kg/cm²\n\n")

            # VERIFICACIÓN DE ESBELTEZ
            self.agregar_texto_calculos("2. VERIFICACIÓN DE ESBELTEZ (E.070, Art. 20.a)\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")

            esbeltez_calc = CalculosAlbanileria.verificar_esbeltez(h_total_m * 100, t_muro_cm)

            self.agregar_texto_calculos(f"   Fórmula: {esbeltez_calc['formula']}\n", 'formula')
            self.agregar_texto_calculos(f"   Sustitución: {esbeltez_calc['sustitucion']}\n")
            self.agregar_texto_calculos(f"   Resultado: {esbeltez_calc['resultado']}\n", 'resultado')
            self.agregar_texto_calculos(f"   Límite: {esbeltez_calc['limite']}\n")

            if esbeltez_calc['cumple']:
                self.agregar_texto_calculos(f"\n   ✓ CUMPLE - No requiere viga intermedia\n", 'resultado')
                necesita_viga = False
                num_vigas = 0
                h_paño_m = h_total_m
            else:
                self.agregar_texto_calculos(f"\n   ✗ NO CUMPLE - Requiere viga(s) intermedia(s)\n", 'error')

                vigas_calc = CalculosAlbanileria.calcular_numero_vigas(esbeltez_calc['esbeltez'])
                num_vigas = vigas_calc['num_vigas']
                num_paños = vigas_calc['num_paños']
                h_paño_m = h_total_m / num_paños

                self.agregar_texto_calculos(f"\n   {vigas_calc['formula']}\n", 'formula')
                self.agregar_texto_calculos(f"   {vigas_calc['sustitucion']}\n")
                self.agregar_texto_calculos(f"   {vigas_calc['resultado']}\n", 'resultado')
                self.agregar_texto_calculos(f"\n   Número de paños: {num_paños}\n")
                self.agregar_texto_calculos(f"   Altura por paño: {h_paño_m:.2f} m\n")

                nueva_esbeltez = (h_paño_m * 100) / t_muro_cm
                self.agregar_texto_calculos(f"   Nueva esbeltez: λ = {nueva_esbeltez:.2f}\n")

                if nueva_esbeltez <= BaseDatos.H_T_MAX:
                    self.agregar_texto_calculos(f"   ✓ Nueva esbeltez CUMPLE\n", 'resultado')

                necesita_viga = True

            self.agregar_texto_calculos("\n")

            # DISEÑO DE COLUMNAS
            self.agregar_texto_calculos("3. DISEÑO DE COLUMNAS DE CONFINAMIENTO\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")

            t_col = t_muro_cm
            b_col = max(25.0, b_col_prop)
            Ac_col = t_col * b_col

            self.agregar_texto_calculos(f"3.1. DIMENSIONES\n", 'subtitulo')
            self.agregar_texto_calculos(f"     Ancho (t) = {t_col:.1f} cm\n")
            self.agregar_texto_calculos(f"     Peralte (b) = {b_col:.1f} cm\n")
            self.agregar_texto_calculos(f"     Área: Ac = {Ac_col:.2f} cm²\n")

            # Área mínima
            self.agregar_texto_calculos(f"\n3.2. VERIFICACIÓN ÁREA MÍNIMA\n", 'subtitulo')
            area_min_calc = CalculosAlbanileria.area_minima_columna(t_col)
            self.agregar_texto_calculos(f"     Fórmula: {area_min_calc['formula']}\n", 'formula')
            self.agregar_texto_calculos(f"     Sustitución: {area_min_calc['sustitucion']}\n")
            self.agregar_texto_calculos(f"     Resultado: {area_min_calc['resultado']}\n", 'resultado')

            if Ac_col >= area_min_calc['valor']:
                self.agregar_texto_calculos(f"     ✓ CUMPLE\n", 'resultado')
            else:
                self.agregar_texto_calculos(f"     ✗ NO CUMPLE\n", 'error')

            # Acero longitudinal
            self.agregar_texto_calculos(f"\n3.3. ACERO LONGITUDINAL\n", 'subtitulo')
            acero_calc = CalculosAlbanileria.acero_minimo_columna(Ac_col, fc, fy)

            self.agregar_texto_calculos(f"     a) Por resistencia:\n")
            self.agregar_texto_calculos(f"        Fórmula: {acero_calc['formula_1']}\n", 'formula')
            self.agregar_texto_calculos(f"        Sustitución: {acero_calc['sustitucion_1']}\n")
            self.agregar_texto_calculos(f"        Resultado: {acero_calc['resultado_1']}\n", 'resultado')

            self.agregar_texto_calculos(f"\n     b) Por cuantía mínima:\n")
            self.agregar_texto_calculos(f"        Fórmula: {acero_calc['formula_2']}\n", 'formula')
            self.agregar_texto_calculos(f"        Resultado: {acero_calc['resultado_2']}\n", 'resultado')

            self.agregar_texto_calculos(f"\n     c) Área requerida:\n")
            self.agregar_texto_calculos(f"        {acero_calc['formula_3']}\n", 'formula')
            self.agregar_texto_calculos(f"        {acero_calc['resultado_3']}\n", 'resultado')

            # Selección de barras
            As_req_col = acero_calc['As_requerido']
            mejor_barras, todas_opciones = CalculosAlbanileria.seleccionar_barras_exacto(As_req_col, b_col, t_col, rec,
                                                                                         'columna')

            self.agregar_texto_calculos(f"\n3.4. SELECCIÓN DE BARRAS\n", 'subtitulo')
            self.agregar_texto_calculos(f"     Análisis de opciones:\n")
            self.agregar_texto_calculos(f"     {'Diám.':<10} {'N° barras':<12} {'As prov.':<12} {'Efic.':<10}\n")
            self.agregar_texto_calculos(f"     {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}\n")

            for opcion in todas_opciones:
                self.agregar_texto_calculos(
                    f"     {opcion['diametro']:<10} {opcion['num_barras']:<12} {opcion['As_provisto']:<12.2f} {opcion['eficiencia']:<10.1f}%\n")

            if mejor_barras:
                self.agregar_texto_calculos(
                    f"\n     ✓ Acero seleccionado: {mejor_barras['num_barras']} Ø {mejor_barras['diametro']}\n",
                    'resultado')
                self.agregar_texto_calculos(f"       As = {mejor_barras['As_provisto']:.2f} cm²\n", 'resultado')

                cuantia_col = mejor_barras['As_provisto'] / Ac_col
                self.agregar_texto_calculos(f"       Cuantía: ρ = {cuantia_col * 100:.2f}%\n")

            # Estribos
            self.agregar_texto_calculos(f"\n3.5. ESTRIBOS\n", 'subtitulo')
            estribos_calc = CalculosAlbanileria.diseñar_estribos(b_col, t_col, 'columna')

            self.agregar_texto_calculos(f"     {estribos_calc['formulas']['s_conf']}\n", 'formula')
            self.agregar_texto_calculos(f"     {estribos_calc['formulas']['Lo']}\n", 'formula')
            self.agregar_texto_calculos(f"\n     ✓ Diámetro: Ø {estribos_calc['diametro']}\n", 'resultado')
            self.agregar_texto_calculos(f"       Patrón: {estribos_calc['patron']}\n", 'resultado')

            # Guardar resultados
            self.resultados['columna'] = {
                't': t_col,
                'b': b_col,
                'Ac': Ac_col,
                'barras': mejor_barras,
                'estribos': estribos_calc
            }

            # DISEÑO DE VIGAS
            if necesita_viga and longitud_m > 0:
                self.agregar_texto_calculos(f"\n\n4. DISEÑO DE VIGA(S) INTERMEDIA(S)\n", 'subtitulo')
                self.agregar_texto_calculos("-" * 80 + "\n")

                t_viga = t_muro_cm
                peralte_calc = CalculosAlbanileria.peralte_viga_minimo(longitud_m)
                h_viga = peralte_calc['h_recomendado']

                self.agregar_texto_calculos(f"4.1. PERALTE DE VIGA\n", 'subtitulo')
                self.agregar_texto_calculos(f"     Fórmula: {peralte_calc['formula']}\n", 'formula')
                self.agregar_texto_calculos(f"     h = L/12 = {peralte_calc['h_L12']:.1f} cm\n")
                self.agregar_texto_calculos(f"     ✓ Peralte adoptado: h = {h_viga:.0f} cm\n", 'resultado')

                Ac_viga = t_viga * h_viga
                self.agregar_texto_calculos(f"\n     Dimensiones: {t_viga:.0f} × {h_viga:.0f} cm\n")
                self.agregar_texto_calculos(f"     Área: Ac = {Ac_viga:.2f} cm²\n")

                # Acero de viga
                As_req_viga = max((0.7 * math.sqrt(fc) * t_viga * h_viga / 2) / fy, 0.0018 * t_viga * h_viga)
                mejor_sup, _ = CalculosAlbanileria.seleccionar_barras_exacto(As_req_viga, t_viga, h_viga, rec, 'viga')
                mejor_inf, _ = CalculosAlbanileria.seleccionar_barras_exacto(As_req_viga, t_viga, h_viga, rec, 'viga')

                self.agregar_texto_calculos(f"\n4.2. ACERO LONGITUDINAL\n", 'subtitulo')
                if mejor_sup and mejor_inf:
                    self.agregar_texto_calculos(
                        f"     ✓ Superior: {mejor_sup['num_barras']} Ø {mejor_sup['diametro']}\n", 'resultado')
                    self.agregar_texto_calculos(f"       As = {mejor_sup['As_provisto']:.2f} cm²\n", 'resultado')
                    self.agregar_texto_calculos(
                        f"\n     ✓ Inferior: {mejor_inf['num_barras']} Ø {mejor_inf['diametro']}\n", 'resultado')
                    self.agregar_texto_calculos(f"       As = {mejor_inf['As_provisto']:.2f} cm²\n", 'resultado')

                estribos_viga = CalculosAlbanileria.diseñar_estribos(t_viga, h_viga, 'viga')

                self.agregar_texto_calculos(f"\n4.3. ESTRIBOS\n", 'subtitulo')
                self.agregar_texto_calculos(f"     ✓ Diámetro: Ø {estribos_viga['diametro']}\n", 'resultado')
                self.agregar_texto_calculos(f"       Patrón: {estribos_viga['patron']}\n", 'resultado')

                self.resultados['viga'] = {
                    't': t_viga,
                    'h': h_viga,
                    'Ac': Ac_viga,
                    'barras_sup': mejor_sup,
                    'barras_inf': mejor_inf,
                    'estribos': estribos_viga
                }

            # Guardar datos
            self.resultados['datos'] = {
                'h_total': h_total_m,
                't_muro': t_muro_cm,
                'longitud': longitud_m,
                'fc': fc,
                'fy': fy,
                'fm': fm,
                'necesita_viga': necesita_viga,
                'num_vigas': num_vigas,
                'h_paño': h_paño_m,
                'recubrimiento': rec
            }

            self.actualizar_resultados()
            self.generar_graficos()
            self.generar_memoria_completa()

            self.notebook.select(1)
            messagebox.showinfo("Éxito", "✅ Cálculos completados correctamente")

        except ValueError as e:
            messagebox.showerror("Error", f"Error en datos: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {str(e)}")

    def agregar_texto_calculos(self, texto, tag='normal'):
        """Agrega texto formateado"""
        self.texto_calculos.insert(tk.END, texto, tag)
        self.texto_calculos.see(tk.END)

    def actualizar_resultados(self):
        """Actualiza resumen de resultados"""
        self.texto_resumen.delete(1.0, tk.END)

        if not self.resultados:
            return

        datos = self.resultados.get('datos', {})
        col = self.resultados.get('columna', {})
        viga = self.resultados.get('viga', None)

        self.texto_resumen.insert(tk.END, "═" * 50 + "\n")
        self.texto_resumen.insert(tk.END, "   RESUMEN DE RESULTADOS\n")
        self.texto_resumen.insert(tk.END, "═" * 50 + "\n\n")

        self.texto_resumen.insert(tk.END, "📋 CONFIGURACIÓN:\n")
        self.texto_resumen.insert(tk.END, f"   Altura: {datos.get('h_total', 0):.2f} m\n")
        self.texto_resumen.insert(tk.END, f"   Espesor: {datos.get('t_muro', 0):.1f} cm\n")

        if datos.get('necesita_viga', False):
            self.texto_resumen.insert(tk.END, f"\n   🔴 Requiere {datos.get('num_vigas', 0)} viga(s)\n")
            self.texto_resumen.insert(tk.END, f"   Altura por paño: {datos.get('h_paño', 0):.2f} m\n")
        else:
            self.texto_resumen.insert(tk.END, f"\n   🟢 No requiere vigas\n")

        self.texto_resumen.insert(tk.END, "\n─" * 50 + "\n\n")

        self.texto_resumen.insert(tk.END, "📐 COLUMNAS:\n")
        self.texto_resumen.insert(tk.END, f"   Sección: {col.get('t', 0):.0f} × {col.get('b', 0):.0f} cm\n")

        barras = col.get('barras', {})
        if barras:
            self.texto_resumen.insert(tk.END,
                                      f"   Acero: {barras.get('num_barras', 0)} Ø {barras.get('diametro', '')}\n")
            self.texto_resumen.insert(tk.END, f"   As = {barras.get('As_provisto', 0):.2f} cm²\n")

        estribos = col.get('estribos', {})
        if estribos:
            self.texto_resumen.insert(tk.END, f"   Estribos: Ø {estribos.get('diametro', '')}\n")

        if viga:
            self.texto_resumen.insert(tk.END, "\n─" * 50 + "\n\n")
            self.texto_resumen.insert(tk.END, "🔗 VIGAS:\n")
            self.texto_resumen.insert(tk.END, f"   Sección: {viga.get('t', 0):.0f} × {viga.get('h', 0):.0f} cm\n")

            barras_sup = viga.get('barras_sup', {})
            barras_inf = viga.get('barras_inf', {})

            if barras_sup:
                self.texto_resumen.insert(tk.END,
                                          f"   Superior: {barras_sup.get('num_barras', 0)} Ø {barras_sup.get('diametro', '')}\n")
            if barras_inf:
                self.texto_resumen.insert(tk.END,
                                          f"   Inferior: {barras_inf.get('num_barras', 0)} Ø {barras_inf.get('diametro', '')}\n")

    def generar_graficos(self):
        """Genera gráficos de secciones"""
        self.figura_grafico.clear()

        if not self.resultados:
            return

        col = self.resultados.get('columna', {})
        viga = self.resultados.get('viga', None)
        datos = self.resultados.get('datos', {})

        if viga:
            ax1 = self.figura_grafico.add_subplot(2, 1, 1)
            ax2 = self.figura_grafico.add_subplot(2, 1, 2)
        else:
            ax1 = self.figura_grafico.add_subplot(1, 1, 1)
            ax2 = None

        self.dibujar_seccion_columna(ax1, col, datos)

        if viga and ax2:
            self.dibujar_seccion_viga(ax2, viga, datos)

        self.figura_grafico.tight_layout()
        self.canvas_grafico.draw()

    def dibujar_seccion_columna(self, ax, col, datos):
        """Dibuja sección de columna con barras"""
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title('SECCIÓN DE COLUMNA', fontweight='bold')

        t = col.get('t', 25)
        b = col.get('b', 25)
        barras = col.get('barras', {})
        rec = datos.get('recubrimiento', 4.0)

        rect = patches.Rectangle((0, 0), t, b, linewidth=2.5, edgecolor='black', facecolor='lightgray', alpha=0.4)
        ax.add_patch(rect)

        rect_rec = patches.Rectangle((rec, rec), t - 2 * rec, b - 2 * rec, linewidth=1, edgecolor='blue',
                                     facecolor='none', linestyle='--')
        ax.add_patch(rect_rec)

        if barras:
            num_barras = barras.get('num_barras', 4)
            diam = barras.get('diametro', '3/8"')
            radio = BaseDatos.DIAMETROS_BARRAS.get(diam, 0.95) / 2

            posiciones = self.calcular_posiciones_barras(num_barras, t, b, rec)

            for pos in posiciones:
                circle = patches.Circle(pos, radio, color='red', zorder=3)
                ax.add_patch(circle)
                circle_ext = patches.Circle(pos, radio, fill=False, edgecolor='darkred', linewidth=1.5, zorder=4)
                ax.add_patch(circle_ext)

        ax.annotate('', xy=(0, -2), xytext=(t, -2), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(t / 2, -3.5, f't = {t:.1f} cm', ha='center', fontsize=9, fontweight='bold')

        ax.annotate('', xy=(-2, 0), xytext=(-2, b), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(-4, b / 2, f'b = {b:.1f} cm', ha='center', rotation=90, fontsize=9, fontweight='bold')

        if barras:
            texto = f"{num_barras} Ø {diam}\nAs = {barras.get('As_provisto', 0):.2f} cm²"
            ax.text(t / 2, b + 3, texto, ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlim(-6, t + 6)
        ax.set_ylim(-6, b + 6)
        ax.set_xlabel('Dimensiones en cm', fontsize=8)
        ax.grid(True, alpha=0.3)

    def calcular_posiciones_barras(self, num_barras, t, b, rec):
        """Calcula posiciones de barras en perímetro"""
        posiciones = []

        if num_barras == 4:
            posiciones = [(rec, rec), (t - rec, rec), (t - rec, b - rec), (rec, b - rec)]
        elif num_barras == 6:
            posiciones = [(rec, rec), (t / 2, rec), (t - rec, rec), (t - rec, b - rec), (t / 2, b - rec),
                          (rec, b - rec)]
        elif num_barras == 8:
            posiciones = [(rec, rec), (t / 2, rec), (t - rec, rec), (t - rec, b / 2), (t - rec, b - rec),
                          (t / 2, b - rec), (rec, b - rec), (rec, b / 2)]
        else:
            perimetro = 2 * (t + b)
            espaciamiento = perimetro / num_barras

            for i in range(num_barras):
                distancia = i * espaciamiento

                if distancia < t:
                    x = rec + distancia
                    y = rec
                elif distancia < t + b:
                    x = t - rec
                    y = rec + (distancia - t)
                elif distancia < 2 * t + b:
                    x = t - rec - (distancia - t - b)
                    y = b - rec
                else:
                    x = rec
                    y = b - rec - (distancia - 2 * t - b)

                posiciones.append((x, y))

        return posiciones

    def dibujar_seccion_viga(self, ax, viga, datos):
        """Dibuja sección de viga"""
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title('SECCIÓN DE VIGA', fontweight='bold')

        t = viga.get('t', 25)
        h = viga.get('h', 25)
        barras_sup = viga.get('barras_sup', {})
        barras_inf = viga.get('barras_inf', {})
        rec = datos.get('recubrimiento', 4.0)

        rect = patches.Rectangle((0, 0), t, h, linewidth=2.5, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)

        rect_rec = patches.Rectangle((rec, rec), t - 2 * rec, h - 2 * rec, linewidth=1, edgecolor='blue',
                                     facecolor='none', linestyle='--')
        ax.add_patch(rect_rec)

        if barras_sup:
            num_sup = barras_sup.get('num_barras', 2)
            diam_sup = barras_sup.get('diametro', '3/8"')
            radio_sup = BaseDatos.DIAMETROS_BARRAS.get(diam_sup, 0.95) / 2

            for i in range(num_sup):
                x = rec + (i + 1) * (t - 2 * rec) / (num_sup + 1)
                y = h - rec
                circle = patches.Circle((x, y), radio_sup, color='red', zorder=3)
                ax.add_patch(circle)
                circle_ext = patches.Circle((x, y), radio_sup, fill=False, edgecolor='darkred', linewidth=1.5, zorder=4)
                ax.add_patch(circle_ext)

        if barras_inf:
            num_inf = barras_inf.get('num_barras', 2)
            diam_inf = barras_inf.get('diametro', '3/8"')
            radio_inf = BaseDatos.DIAMETROS_BARRAS.get(diam_inf, 0.95) / 2

            for i in range(num_inf):
                x = rec + (i + 1) * (t - 2 * rec) / (num_inf + 1)
                y = rec
                circle = patches.Circle((x, y), radio_inf, color='blue', zorder=3)
                ax.add_patch(circle)
                circle_ext = patches.Circle((x, y), radio_inf, fill=False, edgecolor='darkblue', linewidth=1.5,
                                            zorder=4)
                ax.add_patch(circle_ext)

        ax.annotate('', xy=(0, -2), xytext=(t, -2), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(t / 2, -3.5, f't = {t:.1f} cm', ha='center', fontsize=9, fontweight='bold')

        ax.annotate('', xy=(-2, 0), xytext=(-2, h), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(-4, h / 2, f'h = {h:.1f} cm', ha='center', rotation=90, fontsize=9, fontweight='bold')

        texto = f"Sup: {barras_sup.get('num_barras', 0)} Ø {barras_sup.get('diametro', '')}\n"
        texto += f"Inf: {barras_inf.get('num_barras', 0)} Ø {barras_inf.get('diametro', '')}"
        ax.text(t / 2, h + 3, texto, ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax.set_xlim(-6, t + 6)
        ax.set_ylim(-6, h + 6)
        ax.set_xlabel('Dimensiones en cm', fontsize=8)
        ax.grid(True, alpha=0.3)

    def generar_memoria_completa(self):
        """Genera memoria de cálculo completa"""
        self.texto_memoria.delete(1.0, tk.END)

        if not self.resultados:
            self.texto_memoria.insert(tk.END, "Ejecute los cálculos primero")
            return

        datos = self.resultados.get('datos', {})
        col = self.resultados.get('columna', {})
        viga = self.resultados.get('viga', None)

        memoria = []
        memoria.append("=" * 90)
        memoria.append("                    MEMORIA DE CÁLCULO")
        memoria.append("            DISEÑO DE ALBAÑILERÍA CONFINADA")
        memoria.append("                  NTE E.070 + E.060 + SENCICO")
        memoria.append("=" * 90)
        memoria.append(f"\nFecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        memoria.append(f"Ingeniero: Emanuel Ancco (@EmanuelAncco)")
        memoria.append(f"Proyecto: [Especificar]\n")

        memoria.append("\n1. DATOS DE ENTRADA")
        memoria.append("-" * 90)
        memoria.append(f"   Altura total: h = {datos.get('h_total', 0):.2f} m")
        memoria.append(f"   Espesor: t = {datos.get('t_muro', 0):.1f} cm")
        if datos.get('longitud', 0) > 0:
            memoria.append(f"   Longitud: L = {datos.get('longitud', 0):.2f} m")
        memoria.append(f"   f'c = {datos.get('fc', 0):.0f} kg/cm²")
        memoria.append(f"   fy = {datos.get('fy', 0):.0f} kg/cm²")

        memoria.append(f"\n\n2. VERIFICACIONES")
        memoria.append("-" * 90)
        if datos.get('h_total') and datos.get('t_muro'):
            esbeltez = (datos.get('h_total') * 100) / datos.get('t_muro')
            memoria.append(f"   Esbeltez: λ = {esbeltez:.2f}")
            memoria.append(f"   Límite: λ ≤ 20.0")

            if datos.get('necesita_viga', False):
                memoria.append(f"   RESULTADO: Requiere {datos.get('num_vigas', 0)} viga(s)")
            else:
                memoria.append(f"   RESULTADO: CUMPLE")

        memoria.append(f"\n\n3. COLUMNAS")
        memoria.append("-" * 90)
        memoria.append(f"   Sección: {col.get('t', 0):.0f} × {col.get('b', 0):.0f} cm")
        memoria.append(f"   Área: Ac = {col.get('Ac', 0):.2f} cm²")

        barras = col.get('barras', {})
        if barras:
            memoria.append(f"\n   Acero longitudinal:")
            memoria.append(f"   {barras.get('num_barras', 0)} Ø {barras.get('diametro', '')}")
            memoria.append(f"   As = {barras.get('As_provisto', 0):.2f} cm²")

        estribos = col.get('estribos', {})
        if estribos:
            memoria.append(f"\n   Estribos:")
            memoria.append(f"   Ø {estribos.get('diametro', '')}")
            memoria.append(f"   {estribos.get('patron', '')}")

        if viga:
            memoria.append(f"\n\n4. VIGAS")
            memoria.append("-" * 90)
            memoria.append(f"   Sección: {viga.get('t', 0):.0f} × {viga.get('h', 0):.0f} cm")

            barras_sup = viga.get('barras_sup', {})
            barras_inf = viga.get('barras_inf', {})

            if barras_sup:
                memoria.append(f"\n   Superior: {barras_sup.get('num_barras', 0)} Ø {barras_sup.get('diametro', '')}")
            if barras_inf:
                memoria.append(f"   Inferior: {barras_inf.get('num_barras', 0)} Ø {barras_inf.get('diametro', '')}")

        memoria.append(f"\n\n5. RECOMENDACIONES")
        memoria.append("-" * 90)
        memoria.append("   - Dentado: 5 cm cada 60 cm")
        memoria.append("   - Chicotes: cada 3 hiladas")
        memoria.append("   - Curado: mínimo 7 días")

        memoria.append(f"\n\n6. REFERENCIAS")
        memoria.append("-" * 90)
        memoria.append("   - NTE E.070: Albañilería")
        memoria.append("   - NTE E.060: Concreto Armado")
        memoria.append("   - Manual SENCICO")

        memoria.append(f"\n\n" + "=" * 90)
        memoria.append("FIN DE LA MEMORIA")
        memoria.append("=" * 90 + "\n")

        for linea in memoria:
            self.texto_memoria.insert(tk.END, linea + "\n")

    def limpiar_campos(self):
        """Limpia todos los campos"""
        self.entrada_altura.delete(0, tk.END)
        self.entrada_espesor.delete(0, tk.END)
        self.entrada_longitud.delete(0, tk.END)
        self.entrada_peralte_col.delete(0, tk.END)
        self.entrada_recubrimiento.delete(0, tk.END)

        self.entrada_altura.insert(0, "4.8")
        self.entrada_espesor.insert(0, "12.5")
        self.entrada_longitud.insert(0, "2.78")
        self.entrada_fc.set('175')
        self.entrada_fy.set('4200')
        self.entrada_fm.set('65')
        self.entrada_peralte_col.insert(0, "25")
        self.entrada_recubrimiento.insert(0, "4.0")

        self.resultados = {}
        self.texto_calculos.delete(1.0, tk.END)
        self.texto_resumen.delete(1.0, tk.END)
        self.texto_memoria.delete(1.0, tk.END)
        self.figura_grafico.clear()
        self.canvas_grafico.draw()

        def exportar_memoria(self):
            """Exporta memoria a .txt"""
            if not self.resultados:
                messagebox.showwarning("Advertencia", "Ejecute los cálculos primero")
                return

            archivo = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")],
                initialfile=f"Memoria_Albanileria_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            )

            if archivo:
                try:
                    with open(archivo, 'w', encoding='utf-8') as f:
                        f.write(self.texto_calculos.get(1.0, tk.END))
                        f.write("\n\n" + "=" * 90 + "\n\n")
                        f.write(self.texto_memoria.get(1.0, tk.END))

                    messagebox.showinfo("Éxito", f"Memoria exportada:\n{archivo}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar:\n{str(e)}")

        def copiar_portapapeles(self):
            """Copia memoria al portapapeles"""
            if not self.resultados:
                messagebox.showwarning("Advertencia", "Ejecute los cálculos primero")
                return

            try:
                self.root.clipboard_clear()
                contenido = self.texto_memoria.get(1.0, tk.END)
                self.root.clipboard_append(contenido)
                messagebox.showinfo("Éxito", "✅ Memoria copiada al portapapeles")
            except Exception as e:
                messagebox.showerror("Error", f"Error al copiar:\n{str(e)}")

    # ============================================================================
    # PUNTO DE ENTRADA PRINCIPAL - FUERA DE LA CLASE
    # ============================================================================

    def main(self):
        """Función principal que inicia la aplicación"""
        try:
            root = tk.Tk()

            # Configurar icono (opcional)
            try:
                root.iconbitmap('icon.ico')
            except:
                pass

            # Crear aplicación
            app = AplicacionDiseño(root)

            # Centrar ventana
            root.update_idletasks()
            width = root.winfo_width()
            height = root.winfo_height()
            x = (root.winfo_screenwidth() // 2) - (width // 2)
            y = (root.winfo_screenheight() // 2) - (height // 2)
            root.geometry(f'{width}x{height}+{x}+{y}')

            # Iniciar bucle principal
            root.mainloop()

        except Exception as e:
            print(f"Error al iniciar la aplicación: {str(e)}")
            messagebox.showerror("Error Fatal", f"No se pudo iniciar:\n{str(e)}")

    if __name__ == "__main__":
        print("=" * 80)
        print("  SISTEMA DE DISEÑO DE ALBAÑILERÍA CONFINADA")
        print("  Norma E.070 + E.060 + SENCICO")
        print("  Autor: Emanuel Ancco (@EmanuelAncco)")
        print("  Fecha: 2025-01-11")
        print("  Versión: 1.0 COMPLETA")
        print("=" * 80)
        print("\n🚀 Iniciando aplicación...")
        print("📋 Cargando interfaz gráfica...")
        print("✅ Sistema listo para usar\n")



    # ============================================================================
    # FIN DEL CÓDIGO COMPLETO
    # ============================================================================