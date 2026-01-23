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
# ============================================================================
# Autor: Emanuel Ancco (@EmanuelAncco)
# Fecha: 2025-01-11
# Versión: 1.0 FINAL
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
    """Clase con todos los cálculos estructurales"""

    @staticmethod
    def verificar_esbeltez(h_cm, t_cm):
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
        Ac_min = 15 * t_cm
        return {
            'formula': 'Ac_min = 15t',
            'sustitucion': f'Ac_min = 15 × {t_cm:.1f}',
            'resultado': f'Ac_min = {Ac_min:.2f} cm²',
            'valor': Ac_min
        }

    @staticmethod
    def acero_minimo_columna(Ac, fc, fy):
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
        diam_estr = '1/4"' if max(b, h) <= 30 else '3/8"'

        s_conf = math.floor(min(b / 2, h / 2, 10.0) / 5) * 5
        s_central = math.floor(min(16 * 0.95, 48 * BaseDatos.DIAMETROS_BARRAS[diam_estr], min(b, h)) / 5) * 5
        Lo = max(max(b, h), 45.0)
        n_estr_conf = math.floor(Lo / s_conf) if s_conf > 0 else 1

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
    """Aplicación principal con interfaz gráfica"""

    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Diseño de Albañilería Confinada - E.070")
        self.root.geometry("1400x900")
        self.resultados = {}
        self.configurar_estilos()
        self.crear_interfaz()

    def configurar_estilos(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 10, 'bold'), foreground='#34495e')
        style.configure('TButton', font=('Arial', 10))
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

    def crear_interfaz(self):
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

        row += 1
        ttk.Label(scrollable_frame, text="Altura libre total (m):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.entrada_altura = ttk.Entry(scrollable_frame, width=15)
        self.entrada_altura.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_altura.insert(0, "4.8")
        ttk.Label(scrollable_frame, text="metros").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="Espesor del muro (cm):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.entrada_espesor = ttk.Entry(scrollable_frame, width=15)
        self.entrada_espesor.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_espesor.insert(0, "12.5")
        ttk.Label(scrollable_frame, text="cm").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="Longitud del muro (m):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.entrada_longitud = ttk.Entry(scrollable_frame, width=15)
        self.entrada_longitud.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_longitud.insert(0, "2.78")
        ttk.Label(scrollable_frame, text="metros (opcional)").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="MATERIALES", style='Subtitle.TLabel').grid(row=row, column=0, columnspan=3,
                                                                                     sticky='w', padx=20, pady=(20, 10))

        row += 1
        ttk.Label(scrollable_frame, text="Resistencia concreto f'c (kg/cm²):").grid(row=row, column=0, sticky='e',
                                                                                    padx=5, pady=5)
        self.entrada_fc = ttk.Combobox(scrollable_frame, width=13, values=['140', '175', '210', '280', '350'])
        self.entrada_fc.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_fc.set('175')
        ttk.Label(scrollable_frame, text="kg/cm²").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="Fluencia del acero fy (kg/cm²):").grid(row=row, column=0, sticky='e', padx=5,
                                                                                 pady=5)
        self.entrada_fy = ttk.Combobox(scrollable_frame, width=13, values=['4200', '5600'])
        self.entrada_fy.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_fy.set('4200')
        ttk.Label(scrollable_frame, text="kg/cm²").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="Resistencia albañilería f'm (kg/cm²):").grid(row=row, column=0, sticky='e',
                                                                                       padx=5, pady=5)
        self.entrada_fm = ttk.Combobox(scrollable_frame, width=13, values=['35', '45', '50', '65'])
        self.entrada_fm.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_fm.set('65')
        ttk.Label(scrollable_frame, text="kg/cm²").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="COLUMNAS DE CONFINAMIENTO", style='Subtitle.TLabel').grid(row=row, column=0,
                                                                                                    columnspan=3,
                                                                                                    sticky='w', padx=20,
                                                                                                    pady=(20, 10))

        row += 1
        ttk.Label(scrollable_frame, text="Peralte propuesto (cm):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.entrada_peralte_col = ttk.Entry(scrollable_frame, width=15)
        self.entrada_peralte_col.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_peralte_col.insert(0, "25")
        ttk.Label(scrollable_frame, text="cm").grid(row=row, column=2, sticky='w')

        row += 1
        ttk.Label(scrollable_frame, text="Recubrimiento (cm):").grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.entrada_recubrimiento = ttk.Entry(scrollable_frame, width=15)
        self.entrada_recubrimiento.grid(row=row, column=1, padx=5, pady=5)
        self.entrada_recubrimiento.insert(0, "4.0")
        ttk.Label(scrollable_frame, text="cm").grid(row=row, column=2, sticky='w')

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
        self.texto_calculos = scrolledtext.ScrolledText(self.tab_calculos, wrap=tk.WORD, width=80, height=40,
                                                        font=('Courier New', 9))
        self.texto_calculos.pack(fill='both', expand=True, padx=10, pady=10)

        self.texto_calculos.tag_config('titulo', font=('Courier New', 12, 'bold'), foreground='#2c3e50')
        self.texto_calculos.tag_config('subtitulo', font=('Courier New', 10, 'bold'), foreground='#34495e')
        self.texto_calculos.tag_config('formula', font=('Courier New', 9, 'italic'), foreground='#16a085')
        self.texto_calculos.tag_config('resultado', font=('Courier New', 9, 'bold'), foreground='#27ae60')
        self.texto_calculos.tag_config('error', font=('Courier New', 9, 'bold'), foreground='#e74c3c')

    def crear_tab_resultados(self):
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
        toolbar = ttk.Frame(self.tab_memoria)
        toolbar.pack(fill='x', padx=10, pady=5)

        ttk.Button(toolbar, text="📄 Generar Memoria", command=self.generar_memoria_completa).pack(side='left', padx=5)
        ttk.Button(toolbar, text="💾 Exportar TXT", command=self.exportar_memoria).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📋 Copiar", command=self.copiar_portapapeles).pack(side='left', padx=5)

        self.texto_memoria = scrolledtext.ScrolledText(self.tab_memoria, wrap=tk.WORD, width=100, height=40,
                                                       font=('Courier New', 9))
        self.texto_memoria.pack(fill='both', expand=True, padx=10, pady=10)

    def calcular(self):
        try:
            h_total_m = float(self.entrada_altura.get())
            t_muro_cm = float(self.entrada_espesor.get())
            longitud_m = float(self.entrada_longitud.get() or "0")
            fc = float(self.entrada_fc.get())
            fy = float(self.entrada_fy.get())
            fm = float(self.entrada_fm.get() or "0")
            b_col_prop = float(self.entrada_peralte_col.get() or "25")
            rec = float(self.entrada_recubrimiento.get())

            self.texto_calculos.delete(1.0, tk.END)

            # ========== ENCABEZADO ==========
            self.agregar_texto_calculos("=" * 80 + "\n", 'titulo')
            self.agregar_texto_calculos("  CÁLCULOS DETALLADOS - ALBAÑILERÍA CONFINADA\n", 'titulo')
            self.agregar_texto_calculos("  Norma E.070 - Albañilería + E.060 - Concreto + SENCICO\n", 'subtitulo')
            self.agregar_texto_calculos(f"  Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n", 'subtitulo')
            self.agregar_texto_calculos(f"  Ingeniero: Emanuel Ancco\n", 'subtitulo')
            self.agregar_texto_calculos("=" * 80 + "\n\n")

            # ========== DATOS DE ENTRADA ==========
            self.agregar_texto_calculos("1. DATOS DE ENTRADA\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos(f"   Altura total muro:     h = {h_total_m:.2f} m = {h_total_m * 100:.0f} cm\n")
            self.agregar_texto_calculos(f"   Espesor de muro:       t = {t_muro_cm:.1f} cm\n")
            if longitud_m > 0:
                self.agregar_texto_calculos(f"   Longitud de muro:      L = {longitud_m:.2f} m\n")
            self.agregar_texto_calculos(f"   Resistencia concreto:  f'c = {fc:.0f} kg/cm²\n")
            self.agregar_texto_calculos(f"   Fluencia acero:        fy = {fy:.0f} kg/cm²\n")
            if fm > 0:
                self.agregar_texto_calculos(f"   Resistencia ladrillo:  f'm = {fm:.0f} kg/cm²\n")
            self.agregar_texto_calculos(f"   Peralte propuesto col: b = {b_col_prop:.1f} cm\n")
            self.agregar_texto_calculos(f"   Recubrimiento:         r = {rec:.1f} cm\n")
            self.agregar_texto_calculos("\n")

            # ========== VERIFICACIÓN DE ESBELTEZ ==========
            self.agregar_texto_calculos("2. VERIFICACIÓN DE ESBELTEZ (E.070, Art. 20.a)\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos("   La norma E.070 establece que la relación altura libre a espesor\n")
            self.agregar_texto_calculos("   del muro no debe ser mayor a 20:\n\n")

            esbeltez_calc = CalculosAlbanileria.verificar_esbeltez(h_total_m * 100, t_muro_cm)

            self.agregar_texto_calculos(f"   Fórmula:      {esbeltez_calc['formula']}\n", 'formula')
            self.agregar_texto_calculos(f"   Sustitución:  {esbeltez_calc['sustitucion']}\n")
            self.agregar_texto_calculos(f"   Resultado:    {esbeltez_calc['resultado']}\n", 'resultado')
            self.agregar_texto_calculos(f"   Límite:       {esbeltez_calc['limite']}\n")
            self.agregar_texto_calculos(f"   Espesor mínimo requerido: t_min = {esbeltez_calc['t_min_req']:.2f} cm\n")

            if esbeltez_calc['cumple']:
                self.agregar_texto_calculos(f"\n   ✓ CUMPLE - No se requiere viga intermedia\n", 'resultado')
                self.agregar_texto_calculos(f"   El muro puede construirse sin vigas soleras intermedias.\n")
                necesita_viga = False
                num_vigas = 0
                h_paño_m = h_total_m
            else:
                self.agregar_texto_calculos(f"\n   ✗ NO CUMPLE - Se requiere viga(s) intermedia(s)\n", 'error')
                self.agregar_texto_calculos(f"   Es necesario colocar vigas soleras intermedias para reducir\n")
                self.agregar_texto_calculos(f"   la altura libre de cada paño de albañilería.\n\n")

                vigas_calc = CalculosAlbanileria.calcular_numero_vigas(esbeltez_calc['esbeltez'])
                num_vigas = vigas_calc['num_vigas']
                num_paños = vigas_calc['num_paños']
                h_paño_m = h_total_m / num_paños

                self.agregar_texto_calculos(f"   Cálculo de número de vigas necesarias:\n")
                self.agregar_texto_calculos(f"   {vigas_calc['formula']}\n", 'formula')
                self.agregar_texto_calculos(f"   {vigas_calc['sustitucion']}\n")
                self.agregar_texto_calculos(f"   {vigas_calc['resultado']}\n", 'resultado')
                self.agregar_texto_calculos(f"\n   Número de paños:        {num_paños}\n")
                self.agregar_texto_calculos(f"   Altura por paño:        {h_paño_m:.2f} m = {h_paño_m * 100:.0f} cm\n")

                nueva_esbeltez = (h_paño_m * 100) / t_muro_cm
                self.agregar_texto_calculos(f"   Nueva esbeltez:         λ = {nueva_esbeltez:.2f}\n")

                if nueva_esbeltez <= BaseDatos.H_T_MAX:
                    self.agregar_texto_calculos(f"   ✓ Nueva esbeltez CUMPLE con el límite\n", 'resultado')
                else:
                    self.agregar_texto_calculos(f"   ⚠ ADVERTENCIA: Verificar espesor del muro\n", 'error')

                necesita_viga = True

            self.agregar_texto_calculos("\n")

            # ========== DISEÑO DE COLUMNAS ==========
            self.agregar_texto_calculos("3. DISEÑO DE COLUMNAS DE CONFINAMIENTO\n", 'subtitulo')
            self.agregar_texto_calculos("=" * 80 + "\n")
            self.agregar_texto_calculos("   Las columnas de confinamiento son elementos estructurales de\n")
            self.agregar_texto_calculos("   concreto armado que se construyen en los extremos y esquinas\n")
            self.agregar_texto_calculos("   del muro para proporcionar rigidez y resistencia lateral.\n\n")

            # === DIMENSIONES ===
            self.agregar_texto_calculos("3.1. DIMENSIONES DE LA COLUMNA\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")

            t_col = t_muro_cm
            b_col = b_col_prop if b_col_prop > 0 else 25.0
            Ac_col = t_col * b_col

            self.agregar_texto_calculos(f"   Ancho de columna (igual al espesor del muro):\n")
            self.agregar_texto_calculos(f"   t = t_muro = {t_col:.1f} cm\n\n")
            self.agregar_texto_calculos(f"   Peralte de columna (propuesto):\n")
            self.agregar_texto_calculos(f"   b = {b_col:.1f} cm\n\n")
            self.agregar_texto_calculos(f"   Área de la sección transversal:\n")
            self.agregar_texto_calculos(f"   Fórmula:      Ac = t × b\n", 'formula')
            self.agregar_texto_calculos(f"   Sustitución:  Ac = {t_col:.1f} × {b_col:.1f}\n")
            self.agregar_texto_calculos(f"   Resultado:    Ac = {Ac_col:.2f} cm²\n", 'resultado')

            # === ÁREA MÍNIMA ===
            self.agregar_texto_calculos(f"\n3.2. VERIFICACIÓN ÁREA MÍNIMA (E.070, Art. 27.3.a.1)\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos(f"   La norma E.070 establece que el área mínima de la columna debe ser:\n")

            area_min_calc = CalculosAlbanileria.area_minima_columna(t_col)

            self.agregar_texto_calculos(f"\n   Fórmula:      {area_min_calc['formula']}\n", 'formula')
            self.agregar_texto_calculos(f"   Sustitución:  {area_min_calc['sustitucion']}\n")
            self.agregar_texto_calculos(f"   Resultado:    {area_min_calc['resultado']}\n", 'resultado')
            self.agregar_texto_calculos(f"\n   Comparación:\n")
            self.agregar_texto_calculos(f"   Ac provisto = {Ac_col:.2f} cm²\n")
            self.agregar_texto_calculos(f"   Ac mínimo   = {area_min_calc['valor']:.2f} cm²\n")

            if Ac_col >= area_min_calc['valor']:
                self.agregar_texto_calculos(f"\n   ✓ CUMPLE - El área provista es mayor al mínimo requerido\n",
                                            'resultado')
            else:
                self.agregar_texto_calculos(f"\n   ✗ NO CUMPLE - Aumentar dimensiones de la columna\n", 'error')
                self.agregar_texto_calculos(
                    f"   Peralte mínimo sugerido: b_min = {area_min_calc['valor'] / t_col:.1f} cm\n")

            # === ACERO LONGITUDINAL ===
            self.agregar_texto_calculos(f"\n3.3. ACERO LONGITUDINAL (E.070, Art. 27.3.a.2)\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos(f"   El acero longitudinal debe cumplir con dos criterios:\n")
            self.agregar_texto_calculos(f"   a) Por resistencia según E.070\n")
            self.agregar_texto_calculos(f"   b) Por cuantía mínima según E.060\n\n")

            acero_calc = CalculosAlbanileria.acero_minimo_columna(Ac_col, fc, fy)

            self.agregar_texto_calculos(f"   a) Por resistencia (E.070, Art. 27.3.a.2):\n")
            self.agregar_texto_calculos(f"      La norma establece:\n")
            self.agregar_texto_calculos(f"      Fórmula:      {acero_calc['formula_1']}\n", 'formula')
            self.agregar_texto_calculos(f"      Sustitución:  {acero_calc['sustitucion_1']}\n")
            self.agregar_texto_calculos(f"      Resultado:    {acero_calc['resultado_1']}\n", 'resultado')

            self.agregar_texto_calculos(f"\n   b) Por cuantía mínima (E.060, 10.9.1):\n")
            self.agregar_texto_calculos(
                f"      La cuantía mínima de acero longitudinal es ρ_min = {BaseDatos.CUANTIA_MIN}\n")
            self.agregar_texto_calculos(f"      Fórmula:      {acero_calc['formula_2']}\n", 'formula')
            self.agregar_texto_calculos(f"      Sustitución:  {acero_calc['sustitucion_2']}\n")
            self.agregar_texto_calculos(f"      Resultado:    {acero_calc['resultado_2']}\n", 'resultado')

            self.agregar_texto_calculos(f"\n   c) Área de acero requerida:\n")
            self.agregar_texto_calculos(f"      Se toma el mayor de ambos valores:\n")
            self.agregar_texto_calculos(f"      {acero_calc['formula_3']}\n", 'formula')
            self.agregar_texto_calculos(f"      {acero_calc['resultado_3']}\n", 'resultado')

            # === SELECCIÓN DE BARRAS ===
            As_req_col = acero_calc['As_requerido']
            mejor_barras, todas_opciones = CalculosAlbanileria.seleccionar_barras_exacto(As_req_col, b_col, t_col, rec,
                                                                                         'columna')

            self.agregar_texto_calculos(f"\n3.4. SELECCIÓN DE BARRAS LONGITUDINALES\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos(f"   Criterios de diseño (E.060, Art. 7.6.1):\n")
            self.agregar_texto_calculos(f"   - Separación libre mínima: s_min = max(2.5 cm, db, 1.3×agregado)\n",
                                        'formula')
            self.agregar_texto_calculos(f"   - Número mínimo: 4 barras en columnas\n")
            self.agregar_texto_calculos(f"   - Distribución: número par de barras en el perímetro\n")
            self.agregar_texto_calculos(f"   - Recubrimiento: {rec:.1f} cm\n\n")

            self.agregar_texto_calculos(f"   Análisis de opciones de refuerzo:\n\n")
            self.agregar_texto_calculos(
                f"   {'Diámetro':<10} {'N° barras':<12} {'As prov.(cm²)':<15} {'Eficiencia':<12} {'¿Caben?':<10}\n")
            self.agregar_texto_calculos(f"   {'-' * 10} {'-' * 12} {'-' * 15} {'-' * 12} {'-' * 10}\n")

            for opcion in todas_opciones:
                caben_str = "✓ Sí" if opcion['caben'] else "✗ No"
                self.agregar_texto_calculos(
                    f"   {opcion['diametro']:<10} {opcion['num_barras']:<12} "
                    f"{opcion['As_provisto']:<15.2f} {opcion['eficiencia']:<12.1f}% {caben_str:<10}\n"
                )

            if mejor_barras:
                self.agregar_texto_calculos(f"\n   ✓ ACERO SELECCIONADO:\n", 'resultado')
                self.agregar_texto_calculos(
                    f"     Configuración: {mejor_barras['num_barras']} Ø {mejor_barras['diametro']}\n", 'resultado')
                self.agregar_texto_calculos(f"     Área provista: As = {mejor_barras['As_provisto']:.2f} cm²\n",
                                            'resultado')
                self.agregar_texto_calculos(f"     Área requerida: As_req = {As_req_col:.2f} cm²\n")
                self.agregar_texto_calculos(f"     Eficiencia: {mejor_barras['eficiencia']:.1f}%\n", 'resultado')

                cuantia_col = mejor_barras['As_provisto'] / Ac_col
                self.agregar_texto_calculos(
                    f"     Cuantía de acero: ρ = As/Ac = {cuantia_col:.4f} = {cuantia_col * 100:.2f}%\n")

                if BaseDatos.CUANTIA_MIN <= cuantia_col <= BaseDatos.CUANTIA_MAX:
                    self.agregar_texto_calculos(
                        f"     ✓ Cuantía dentro de límites ({BaseDatos.CUANTIA_MIN} ≤ ρ ≤ {BaseDatos.CUANTIA_MAX})\n",
                        'resultado')
                else:
                    self.agregar_texto_calculos(f"     ⚠ Cuantía fuera de límites recomendados\n", 'error')
            else:
                self.agregar_texto_calculos(f"\n   ✗ NO se encontró configuración válida\n", 'error')
                self.agregar_texto_calculos(f"   Recomendación: Aumentar dimensiones de la columna\n")

            # === ESTRIBOS ===
            self.agregar_texto_calculos(f"\n3.5. ESTRIBOS DE CONFINAMIENTO (E.060, Cap. 21 + E.070)\n", 'subtitulo')
            self.agregar_texto_calculos("-" * 80 + "\n")
            self.agregar_texto_calculos(f"   Los estribos proporcionan confinamiento al núcleo de concreto\n")
            self.agregar_texto_calculos(f"   y evitan el pandeo de las barras longitudinales.\n\n")

            estribos_calc = CalculosAlbanileria.diseñar_estribos(b_col, t_col, 'columna')

            self.agregar_texto_calculos(f"   Espaciamiento en zona de confinamiento:\n")
            self.agregar_texto_calculos(f"   {estribos_calc['formulas']['s_conf']}\n", 'formula')
            self.agregar_texto_calculos(f"\n   Longitud de confinamiento desde cada extremo:\n")
            self.agregar_texto_calculos(f"   {estribos_calc['formulas']['Lo']}\n", 'formula')
            self.agregar_texto_calculos(f"\n   Espaciamiento en zona central:\n")
            self.agregar_texto_calculos(f"   s_central = min(16db_long, 48db_estribo, b_min)\n", 'formula')
            self.agregar_texto_calculos(f"   s_central = {estribos_calc['s_central']:.0f} cm\n", 'resultado')

            self.agregar_texto_calculos(f"\n   ✓ CONFIGURACIÓN DE ESTRIBOS:\n", 'resultado')
            self.agregar_texto_calculos(f"     Diámetro: Ø {estribos_calc['diametro']}\n", 'resultado')
            self.agregar_texto_calculos(f"     Patrón de distribución:\n", 'resultado')
            self.agregar_texto_calculos(f"     {estribos_calc['patron']}\n", 'resultado')
            self.agregar_texto_calculos(f"\n     Donde:\n")
            self.agregar_texto_calculos(f"     - Primer estribo a 5 cm del extremo\n")
            self.agregar_texto_calculos(
                f"     - Zona confinada: {estribos_calc['n_estribos_conf']} estribos @ {estribos_calc['s_confinamiento']:.0f} cm\n")
            self.agregar_texto_calculos(f"     - Zona central: @ {estribos_calc['s_central']:.0f} cm\n")

            # Guardar resultados de columnas
            self.resultados['columna'] = {
                't': t_col,
                'b': b_col,
                'Ac': Ac_col,
                'barras': mejor_barras,
                'estribos': estribos_calc
            }

            # ========== DISEÑO DE VIGAS (si es necesario) ==========
            if necesita_viga and longitud_m > 0:
                self.agregar_texto_calculos(f"\n\n4. DISEÑO DE VIGA(S) DE CONFINAMIENTO INTERMEDIA(S)\n", 'subtitulo')
                self.agregar_texto_calculos("=" * 80 + "\n")
                self.agregar_texto_calculos("   Las vigas soleras intermedias son necesarias para reducir\n")
                self.agregar_texto_calculos("   la esbeltez de los paños de albañilería.\n\n")

                t_viga = t_muro_cm
                peralte_calc = CalculosAlbanileria.peralte_viga_minimo(longitud_m)
                h_viga = peralte_calc['h_recomendado']

                self.agregar_texto_calculos(f"4.1. PERALTE DE VIGA\n", 'subtitulo')
                self.agregar_texto_calculos("-" * 80 + "\n")
                self.agregar_texto_calculos(f"   Criterio de peralte mínimo: {peralte_calc['formula']}\n", 'formula')
                self.agregar_texto_calculos(f"   Para luz L = {longitud_m:.2f} m:\n\n")
                self.agregar_texto_calculos(f"   h = L/10 = {peralte_calc['h_L10']:.1f} cm (muy conservador)\n")
                self.agregar_texto_calculos(
                    f"   h = L/12 = {peralte_calc['h_L12']:.1f} cm (conservador - RECOMENDADO)\n")
                self.agregar_texto_calculos(f"   h = L/14 = {peralte_calc['h_L14']:.1f} cm (estándar)\n")
                self.agregar_texto_calculos(f"\n   Peralte mínimo práctico: {peralte_calc['h_min_practico']:.0f} cm\n")
                self.agregar_texto_calculos(f"\n   ✓ Peralte adoptado: h = {h_viga:.0f} cm\n", 'resultado')

                Ac_viga = t_viga * h_viga
                self.agregar_texto_calculos(f"\n   Dimensiones finales de la viga:\n")
                self.agregar_texto_calculos(f"   t × h = {t_viga:.0f} cm × {h_viga:.0f} cm\n")
                self.agregar_texto_calculos(f"   Área: Ac = {Ac_viga:.2f} cm²\n")

                # Acero longitudinal de viga
                self.agregar_texto_calculos(f"\n4.2. ACERO LONGITUDINAL (E.070, Art. 27.3.b)\n", 'subtitulo')
                self.agregar_texto_calculos("-" * 80 + "\n")
                self.agregar_texto_calculos(f"   Criterio mínimo según E.070:\n")
                self.agregar_texto_calculos(f"   As_min = max(0.7√f'c × bw × d / fy, 0.0018 × bw × d)\n", 'formula')

                As_req_viga = max((0.7 * math.sqrt(fc) * t_viga * h_viga / 2) / fy, 0.0018 * t_viga * h_viga)
                self.agregar_texto_calculos(f"\n   As_requerido = {As_req_viga:.2f} cm²\n")

                mejor_sup, _ = CalculosAlbanileria.seleccionar_barras_exacto(As_req_viga, t_viga, h_viga, rec, 'viga')
                mejor_inf, _ = CalculosAlbanileria.seleccionar_barras_exacto(As_req_viga, t_viga, h_viga, rec, 'viga')

                if mejor_sup and mejor_inf:
                    self.agregar_texto_calculos(
                        f"\n   ✓ Acero superior: {mejor_sup['num_barras']} Ø {mejor_sup['diametro']}\n", 'resultado')
                    self.agregar_texto_calculos(f"     As_sup = {mejor_sup['As_provisto']:.2f} cm²\n", 'resultado')

                    self.agregar_texto_calculos(
                        f"\n   ✓ Acero inferior: {mejor_inf['num_barras']} Ø {mejor_inf['diametro']}\n", 'resultado')
                    self.agregar_texto_calculos(f"     As_inf = {mejor_inf['As_provisto']:.2f} cm²\n", 'resultado')

                # Estribos de viga
                estribos_viga = CalculosAlbanileria.diseñar_estribos(t_viga, h_viga, 'viga')

                self.agregar_texto_calculos(f"\n4.3. ESTRIBOS DE LA VIGA\n", 'subtitulo')
                self.agregar_texto_calculos("-" * 80 + "\n")
                self.agregar_texto_calculos(f"   ✓ Diámetro: Ø {estribos_viga['diametro']}\n", 'resultado')
                self.agregar_texto_calculos(f"     Patrón: {estribos_viga['patron']}\n", 'resultado')

                self.resultados['viga'] = {
                    't': t_viga,
                    'h': h_viga,
                    'Ac': Ac_viga,
                    'barras_sup': mejor_sup,
                    'barras_inf': mejor_inf,
                    'estribos': estribos_viga
                }

            # Guardar datos generales
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
        self.texto_calculos.insert(tk.END, texto, tag)
        self.texto_calculos.see(tk.END)

    def actualizar_resultados(self):
        self.texto_resumen.delete(1.0, tk.END)

        if not self.resultados:
            return

        datos = self.resultados.get('datos', {})
        col = self.resultados.get('columna', {})
        viga = self.resultados.get('viga', None)

        self.texto_resumen.insert(tk.END, "═" * 50 + "\n")
        self.texto_resumen.insert(tk.END, "   RESUMEN\n")
        self.texto_resumen.insert(tk.END, "═" * 50 + "\n\n")

        self.texto_resumen.insert(tk.END, "📋 MURO:\n")
        self.texto_resumen.insert(tk.END, f"   h = {datos.get('h_total', 0):.2f} m\n")
        self.texto_resumen.insert(tk.END, f"   t = {datos.get('t_muro', 0):.1f} cm\n\n")

        self.texto_resumen.insert(tk.END, "📐 COLUMNAS:\n")
        self.texto_resumen.insert(tk.END, f"   {col.get('t', 0):.0f} × {col.get('b', 0):.0f} cm\n")

        barras = col.get('barras', {})
        if barras:
            self.texto_resumen.insert(tk.END, f"   {barras.get('num_barras', 0)} Ø {barras.get('diametro', '')}\n")

        if viga:
            self.texto_resumen.insert(tk.END, "\n🔗 VIGAS:\n")
            self.texto_resumen.insert(tk.END, f"   {viga.get('t', 0):.0f} × {viga.get('h', 0):.0f} cm\n")

    def generar_graficos(self):
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
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title('COLUMNA', fontweight='bold')

        t = col.get('t', 25)
        b = col.get('b', 25)
        barras = col.get('barras', {})
        rec = datos.get('recubrimiento', 4.0)

        rect = patches.Rectangle((0, 0), t, b, linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.4)
        ax.add_patch(rect)

        # ✅ AGREGAR ESTAS LÍNEAS AQUÍ
        num_barras = 0
        diam = '3/8"'

        if barras:
            num_barras = barras.get('num_barras', 4)
            diam = barras.get('diametro', '3/8"')
            radio = BaseDatos.DIAMETROS_BARRAS.get(diam, 0.95) / 2

            posiciones = self.calcular_posiciones_barras(num_barras, t, b, rec)

            for pos in posiciones:
                circle = patches.Circle(pos, radio, color='red', zorder=3)
                ax.add_patch(circle)

        # ✅ CAMBIAR ESTA LÍNEA
        if num_barras > 0:  # En lugar de usar directamente num_barras
            ax.text(t / 2, b + 2, f"{num_barras} Ø {diam}", ha='center', fontsize=9)

        ax.set_xlim(-5, t + 5)
        ax.set_ylim(-5, b + 5)
        ax.grid(True, alpha=0.3)

    def calcular_posiciones_barras(self, num_barras, t, b, rec):
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
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title('VIGA', fontweight='bold')

        t = viga.get('t', 25)
        h = viga.get('h', 25)
        barras_sup = viga.get('barras_sup', {})
        barras_inf = viga.get('barras_inf', {})
        rec = datos.get('recubrimiento', 4.0)

        rect = patches.Rectangle((0, 0), t, h, linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)

        if barras_sup:
            num_sup = barras_sup.get('num_barras', 2)
            diam_sup = barras_sup.get('diametro', '3/8"')
            radio_sup = BaseDatos.DIAMETROS_BARRAS.get(diam_sup, 0.95) / 2

            for i in range(num_sup):
                x = rec + (i + 1) * (t - 2 * rec) / (num_sup + 1)
                y = h - rec
                circle = patches.Circle((x, y), radio_sup, color='red', zorder=3)
                ax.add_patch(circle)

        if barras_inf:
            num_inf = barras_inf.get('num_barras', 2)
            diam_inf = barras_inf.get('diametro', '3/8"')
            radio_inf = BaseDatos.DIAMETROS_BARRAS.get(diam_inf, 0.95) / 2

            for i in range(num_inf):
                x = rec + (i + 1) * (t - 2 * rec) / (num_inf + 1)
                y = rec
                circle = patches.Circle((x, y), radio_inf, color='blue', zorder=3)
                ax.add_patch(circle)

        ax.set_xlim(-5, t + 5)
        ax.set_ylim(-5, h + 5)
        ax.grid(True, alpha=0.3)

    def generar_memoria_completa(self):
        self.texto_memoria.delete(1.0, tk.END)

        if not self.resultados:
            return

        datos = self.resultados.get('datos', {})
        col = self.resultados.get('columna', {})
        viga = self.resultados.get('viga', None)

        memoria = []
        memoria.append("=" * 90)
        memoria.append("MEMORIA DE CÁLCULO - ALBAÑILERÍA CONFINADA")
        memoria.append("=" * 90)
        memoria.append(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        memoria.append(f"Ingeniero: Emanuel Ancco\n")

        memoria.append("1. DATOS")
        memoria.append(f"   h = {datos.get('h_total', 0):.2f} m")
        memoria.append(f"   t = {datos.get('t_muro', 0):.1f} cm")
        memoria.append(f"   f'c = {datos.get('fc', 0):.0f} kg/cm²\n")

        memoria.append("2. COLUMNAS")
        memoria.append(f"   Sección: {col.get('t', 0):.0f} × {col.get('b', 0):.0f} cm")

        barras = col.get('barras', {})
        if barras:
            memoria.append(f"   Acero: {barras.get('num_barras', 0)} Ø {barras.get('diametro', '')}")

        if viga:
            memoria.append("\n3. VIGAS")
            memoria.append(f"   Sección: {viga.get('t', 0):.0f} × {viga.get('h', 0):.0f} cm")

        memoria.append("\n" + "=" * 90)

        for linea in memoria:
            self.texto_memoria.insert(tk.END, linea + "\n")

    def limpiar_campos(self):
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
        if not self.resultados:
            messagebox.showwarning("Advertencia", "Ejecute los cálculos primero")
            return

        archivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos", "*.*")],
            initialfile=f"Memoria_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        )

        if archivo:
            try:
                with open(archivo, 'w', encoding='utf-8') as f:
                    f.write(self.texto_calculos.get(1.0, tk.END))
                    f.write("\n\n" + "=" * 90 + "\n\n")
                    f.write(self.texto_memoria.get(1.0, tk.END))

                messagebox.showinfo("Éxito", f"Exportado:\n{archivo}")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")

    def copiar_portapapeles(self):
        if not self.resultados:
            messagebox.showwarning("Advertencia", "Ejecute los cálculos primero")
            return

        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.texto_memoria.get(1.0, tk.END))
            messagebox.showinfo("Éxito", "✅ Copiado al portapapeles")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")


# ============================================================================
# FUNCIÓN PRINCIPAL - FUERA DE LA CLASE
# ============================================================================

def main():
    """Inicia la aplicación"""
    try:
        root = tk.Tk()
        app = AplicacionDiseño(root)

        # Centrar ventana
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')

        root.mainloop()

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        messagebox.showerror("Error", f"No se pudo iniciar:\n{str(e)}")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  SISTEMA DE DISEÑO DE ALBAÑILERÍA CONFINADA")
    print("  NTE E.070 + E.060 + SENCICO")
    print("  Autor: Emanuel Ancco (@EmanuelAncco)")
    print("  Versión: 1.0 FINAL")
    print("=" * 80)
    print("\n🚀 Iniciando aplicación...")
    print("✅ Listo\n")

    main()

# ============================================================================
# FIN DEL CÓDIGO
# ============================================================================