import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv

MM_TO_M = 1e-3
MPA_TO_KN_M2 = 1.0 # 1 MPa = 1 N/mm^2 = 1e6 N/m^2 = 1e3 kN/m^2

# =============================
# Utilidades de barras
# =============================
BAR_DIAMETERS_MM = [6, 8, 10, 12, 14, 16, 18, 20, 22, 25]
BAR_AREA_MM2 = {d: math.pi*(d**2)/4 for d in BAR_DIAMETERS_MM}


def format_mm(x):
    return f"{x:.0f} mm"


def format_mm2(x):
    return f"{x:.0f} mm²"


def format_kN(x):
    return f"{x:.2f} kN"


def format_m(x):
    return f"{x:.3f} m"


def choose_bars(required_mm2, min_bars=2, prefer_diam_mm=12):
    """Selecciona un arreglo básico de barras que cumpla As >= requerido.
    Estrategia simple: intenta con el diámetro preferido y aumenta número de barras;
    si no cumple, incrementa diámetro.
    """
    for d in BAR_DIAMETERS_MM:
        n = max(min_bars, math.ceil(required_mm2 / BAR_AREA_MM2[d]))
        As = n * BAR_AREA_MM2[d]
        if As >= required_mm2:
            return n, d, As
    # Si no alcanzó, devuelve con el mayor diámetro
    d = BAR_DIAMETERS_MM[-1]
    n = max(min_bars, math.ceil(required_mm2 / BAR_AREA_MM2[d]))
    As = n * BAR_AREA_MM2[d]
    return n, d, As


# =============================
# Diseño: Columna de Confinamiento (Albañilería Confinada)
# Basado en NTE E.070 (Cap. 7 y 8) y detalles de confinamiento de NTE E.060
# =============================

def diseno_columna_confinamiento(params):
    """
    Calcula cuantías mínimas y detallado para columnas de confinamiento de albañilería.

    Entradas (en unidades prácticas):
    - t_muro_mm: espesor efectivo del muro (NTE E.070 §7.2.3)
    - b_mm, h_mm: dimensiones propuestas de la columna (se recomienda b=h=t_muro, E.070 §7.2.3)
    - L_libre_m: altura libre entre arriostres (para longitud de zona confinada, E.060 §21.4.5)
    - fc_MPa: f'c del concreto del elemento de confinamiento (≥17.5 MPa, E.070 §7.2.1.f)
    - fy_MPa: fy del acero (varillas corrugadas)
    - phi_m: factor de reducción de resistencia para E.070 (tensión/corte-fricción) — usar 0.85 por defecto (E.070 8.6.3)
    - mu_friccion: 0.8 para juntas sin tratamiento; 1.0 para junta intencionalmente rugosa (E.070 8.6.3)
    - Vu_kN: esfuerzo cortante de diseño que debe transferir la columna a través de corte-fricción (E.070 8.6.3)
    - Tu_kN: tracción de diseño en la columna (E.070 8.6.3)
    - db_long_mm: diámetro tentativo de barras longitudinales
    - db_st_mm: diámetro de estribo

    Devuelve: texto de reporte y un dict con resultados clave.
    """
    t = params['t_muro_mm']
    b = params['b_mm']
    h = params['h_mm']
    L_libre = params['L_libre_m']
    fc = params['fc_MPa']
    fy = params['fy_MPa']
    phi_m = params.get('phi_m', 0.85)
    mu = params.get('mu_friccion', 0.8)
    Vu = params['Vu_kN']
    Tu = params['Tu_kN']
    db_long = params['db_long_mm']
    db_st = params['db_st_mm']

    reporte = []
    out = {}

    reporte.append("DISEÑO DE COLUMNA DE CONFINAMIENTO (NTE E.070 – Albañilería Confinada)\n")

    # Chequeos geométricos mínimos
    # E.070 §7.2.3: espesor mínimo de columnas igual al espesor efectivo del muro
    cumple_espesor = (b >= t) and (h >= 150)  # §7.2.5 peralte mínimo 15 cm
    reporte.append(f"1) Geometría propuesta: b = {format_mm(b)}, h = {format_mm(h)}, t_muro = {format_mm(t)}")
    if b < t:
        reporte.append("   - Incumple E.070 §7.2.3: b debe ser ≥ espesor efectivo del muro.")
    else:
        reporte.append("   - Cumple E.070 §7.2.3: b ≥ t_muro.")
    if h < 150:
        reporte.append("   - Incumple E.070 §7.2.5: peralte mínimo h ≥ 150 mm.")
    else:
        reporte.append("   - Cumple E.070 §7.2.5: h ≥ 150 mm.")

    # Resistencia mínima del concreto del elemento de confinamiento (E.070 §7.2.1.f)
    if fc < 17.5:
        reporte.append("   - Advertencia: E.070 §7.2.1.f exige f'c ≥ 17.5 MPa en elementos de confinamiento.")
    else:
        reporte.append("   - f'c cumple E.070 §7.2.1.f (≥ 17.5 MPa).")

    # Área del núcleo (aprox) para información
    Ac = b * h  # mm²

    # Refuerzo vertical requerido por fricción de corte y por tracción (E.070 §8.6.3-a.2)
    # A_sf = V_f / (phi * mu * fy)  y  A_st = T / (phi * fy)
    Asf = (Vu * 1e3) / (phi_m * mu * fy)  # kN -> kN * 1e3 / (MPa) -> mm², pues 1 MPa = 1 N/mm² y 1 kN = 1e3 N
    Ast = (Tu * 1e3) / (phi_m * fy)
    As_req = Asf + Ast

    # Mínimo: al menos 4 barras (núcleo confinado, E.070 §8.6.3-a.2)
    n_barras_min = 4

    # Selección de barras longitudinales
    n_long, d_long, As_prov = choose_bars(As_req, min_bars=n_barras_min)

    reporte.append("\n2) Refuerzo longitudinal (E.070 §8.6.3-a.2)")
    reporte.append(f"   - A_sf = V_u/(φ·μ·f_y) = {format_mm2(Asf)}")
    reporte.append(f"   - A_st = T_u/(φ·f_y) = {format_mm2(Ast)}")
    reporte.append(f"   → A_s,req = A_sf + A_st = {format_mm2(As_req)}")
    reporte.append(f"   - Se disponen {n_long} barras Ø{d_long} mm → A_s,prov = {format_mm2(As_prov)} (≥ {format_mm2(As_req)})")
    if n_long < 4:
        reporte.append("   - Ajustado a mínimo de 4 barras (núcleo confinado, E.070 §8.6.3-a.2).")

    # Estribos: criterios E.060 §21.4.5 para columnas (zona confinada en extremos)
    # So ≤ min(8·db_long, 0.5·menor dimensión, 100 mm)
    menor_dim = min(b, h)
    So_max = min(8*db_long, 0.5*menor_dim, 100)
    # Longitud de zona confinada Lo ≥ max(L_libre/6, mayor_dimensión, 500 mm)
    mayor_dim = max(b, h)
    Lo_min = max(L_libre*1000/6.0, mayor_dim, 500)

    # Estribos fuera de Lo: E.060 §7.10.5.2
    s_fuera_max = min(16*db_long, 48*db_st, menor_dim)

    # Detalle adicional: primer estribo ≤ 100 mm desde el nudo (E.060 §21.4.5)

    reporte.append("\n3) Refuerzo transversal (E.060 §21.4.5 y §7.10.5)")
    reporte.append(f"   - Zona confinada en extremos: Lo ≥ max(Ln/6, dim_mayor, 500 mm) = {format_mm(Lo_min)}")
    reporte.append(f"   - Espaciamiento en zona confinada: So ≤ min(8·db_long, 0.5·dim_menor, 100 mm) = {format_mm(So_max)}")
    reporte.append(f"   - Primer estribo a ≤ 100 mm del nudo.")
    reporte.append(f"   - Fuera de Lo: s ≤ min(16·db_long, 48·db_estribo, dim_menor) = {format_mm(s_fuera_max)}")

    # Especificaciones de ganchos de estribo (E.070 §8.6.3-a.3 permite 135° o 180° en columnas de confinamiento)
    reporte.append("   - En columnas de confinamiento: estribos cerrados con gancho a 135°/180° (E.070 §8.6.3-a.3).")

    out.update(dict(Asf_mm2=Asf, Ast_mm2=Ast, As_req_mm2=As_req, As_prov_mm2=As_prov,
                    n_long=n_long, d_long_mm=d_long, So_max_mm=So_max, Lo_min_mm=Lo_min,
                    s_fuera_max_mm=s_fuera_max))

    return "\n".join(reporte), out


# =============================
# Diseño: Viga Solera (Albañilería Confinada)
# Basado en NTE E.070 (Cap. 7 y 8) + confinamiento E.060 §21.4.4
# =============================

def diseno_viga_solera(params):
    """
    Diseño básico de viga solera para:
      - Geometría mínima (E.070 §7.2.3 y §7.2.4)
      - Tracción pura (E.070 §8.6.3.b: solera del primer nivel a tracción T_s)
      - Confinamiento de estribos en extremos (E.060 §21.4.4)

    Entradas:
    - t_muro_mm: espesor efectivo del muro (mínimo de ancho de solera, §7.2.3)
    - bw_mm, h_mm: sección propuesta de solera
    - h_losa_mm: espesor de losa (h_solera ≥ h_losa, §7.2.4)
    - fc_MPa, fy_MPa
    - phi_t: φ para tracción (usar 0.85 como referencia de seguridad)
    - Ts_kN: tracción de diseño que debe ser tomada por la solera (E.070 §8.6.3-b)
    - db_long_mm, db_st_mm

    Salidas: reporte y resultados clave.
    """
    t = params['t_muro_mm']
    bw = params['bw_mm']
    h = params['h_mm']
    h_losa = params['h_losa_mm']
    fc = params['fc_MPa']
    fy = params['fy_MPa']
    phi_t = params.get('phi_t', 0.85)
    Ts = params['Ts_kN']
    db_long = params['db_long_mm']
    db_st = params['db_st_mm']

    reporte = []
    out = {}

    reporte.append("DISEÑO DE VIGA SOLERA (NTE E.070 – Albañilería Confinada)\n")

    # Chequeos geométricos mínimos (E.070 §7.2.3 y §7.2.4)
    reporte.append(f"1) Geometría propuesta: b_w = {format_mm(bw)}, h = {format_mm(h)}, t_muro = {format_mm(t)}, h_losa = {format_mm(h_losa)}")
    if bw < t:
        reporte.append("   - Incumple E.070 §7.2.3: ancho mínimo b_w ≥ espesor efectivo del muro.")
    else:
        reporte.append("   - Cumple E.070 §7.2.3: b_w ≥ t_muro.")
    if h < h_losa:
        reporte.append("   - Incumple E.070 §7.2.4: peralte mínimo de solera h ≥ espesor de losa.")
    else:
        reporte.append("   - Cumple E.070 §7.2.4: h ≥ h_losa.")

    if fc < 17.5:
        reporte.append("   - Advertencia: E.070 §7.2.1.f exige f'c ≥ 17.5 MPa en elementos de confinamiento.")
    else:
        reporte.append("   - f'c cumple E.070 §7.2.1.f (≥ 17.5 MPa).")

    # Tracción pura (E.070 §8.6.3-b: solera del primer nivel)
    As_req = (Ts * 1e3) / (phi_t * fy)
    n_long, d_long, As_prov = choose_bars(As_req, min_bars=2)

    reporte.append("\n2) Refuerzo longitudinal por tracción (E.070 §8.6.3-b)")
    reporte.append(f"   - A_s,req = T_s/(φ·f_y) = {format_mm2(As_req)}")
    reporte.append(f"   - Provisión: {n_long} barras Ø{d_long} mm → A_s,prov = {format_mm2(As_prov)}")

    # Confinamiento de estribos en extremos (E.060 §21.4.4)
    d_util = h - 40  # aprox. d (mm) con recubrimientos y estribo típico; puede ajustarse por proyecto
    # Espaciamiento en zona confinada en extremos
    s_end_1 = max(0.25*d_util, 150)  # "no es necesario menor a 150 mm" (E.060 §21.4.4)
    s_end_2 = 10*db_long
    s_end_3 = 24*db_st
    s_end_4 = 300
    s_end = min(s_end_1, s_end_2, s_end_3, s_end_4)

    reporte.append("\n3) Refuerzo transversal en extremos (E.060 §21.4.4)")
    reporte.append(f"   - s_end ≤ min(d/4 (pero ≥150 mm), 10·db_long, 24·db_estribo, 300 mm) = {format_mm(s_end)}")
    reporte.append("   - Primer estribo a ≤ 100 mm de la cara del apoyo.")
    reporte.append("   - A lo largo del elemento: s ≤ 0.5·d y lo que exija el diseño a cortante (E.060 §21.4.4.5).")

    out.update(dict(As_req_mm2=As_req, As_prov_mm2=As_prov, n_long=n_long, d_long_mm=d_long,
                    s_end_mm=s_end, d_util_mm=d_util))

    return "\n".join(reporte), out


# =============================
# Interfaz Tkinter
# =============================

class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()

    def create_widgets(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        self.col_tab = ttk.Frame(nb)
        self.sol_tab = ttk.Frame(nb)
        self.muros_tab = ttk.Frame(nb)
        nb.add(self.col_tab, text="Columna de Confinamiento")
        nb.add(self.sol_tab, text="Viga Solera")
        nb.add(self.muros_tab, text="Predimensionamiento de Muros")

        # ---- Columna ----
        self._build_columna_tab()
        # ---- Solera ----
        self._build_solera_tab()
        # ---- Muros (predimensionamiento) ----
        self._build_muros_tab()

    def _build_columna_tab(self):
        f = self.col_tab
        pad = {'padx': 6, 'pady': 4}

        # Entradas
        row = 0
        ttk.Label(f, text="Espesor de muro t (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.col_t = ttk.Entry(f); self.col_t.insert(0, "140")
        self.col_t.grid(row=row, column=1, **pad)

        row += 1
        ttk.Label(f, text="b (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.col_b = ttk.Entry(f); self.col_b.insert(0, "140")
        self.col_b.grid(row=row, column=1, **pad)

        ttk.Label(f, text="h (mm)").grid(row=row, column=2, sticky='e', **pad)
        self.col_h = ttk.Entry(f); self.col_h.insert(0, "200")
        self.col_h.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="Altura libre Ln (m)").grid(row=row, column=0, sticky='e', **pad)
        self.col_Ln = ttk.Entry(f); self.col_Ln.insert(0, "2.50")
        self.col_Ln.grid(row=row, column=1, **pad)

        ttk.Label(f, text="f'c (MPa)").grid(row=row, column=2, sticky='e', **pad)
        self.col_fc = ttk.Entry(f); self.col_fc.insert(0, "21")
        self.col_fc.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="f_y (MPa)").grid(row=row, column=0, sticky='e', **pad)
        self.col_fy = ttk.Entry(f); self.col_fy.insert(0, "420")
        self.col_fy.grid(row=row, column=1, **pad)

        ttk.Label(f, text="φ (E.070, tensión/corte)" ).grid(row=row, column=2, sticky='e', **pad)
        self.col_phi = ttk.Entry(f); self.col_phi.insert(0, "0.85")
        self.col_phi.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="Coef. fricción μ" ).grid(row=row, column=0, sticky='e', **pad)
        self.col_mu = ttk.Entry(f); self.col_mu.insert(0, "0.8")
        self.col_mu.grid(row=row, column=1, **pad)

        ttk.Label(f, text="V_u (kN)").grid(row=row, column=2, sticky='e', **pad)
        self.col_Vu = ttk.Entry(f); self.col_Vu.insert(0, "60")
        self.col_Vu.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="T_u (kN)").grid(row=row, column=0, sticky='e', **pad)
        self.col_Tu = ttk.Entry(f); self.col_Tu.insert(0, "30")
        self.col_Tu.grid(row=row, column=1, **pad)

        ttk.Label(f, text="Ø barras long. (mm)").grid(row=row, column=2, sticky='e', **pad)
        self.col_db_long = ttk.Entry(f); self.col_db_long.insert(0, "12")
        self.col_db_long.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="Ø estribo (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.col_db_st = ttk.Entry(f); self.col_db_st.insert(0, "8")
        self.col_db_st.grid(row=row, column=1, **pad)

        ttk.Button(f, text="Calcular", command=self.run_columna).grid(row=row, column=3, sticky='e', **pad)

        # Salida
        row += 1
        self.col_text = tk.Text(f, height=22, width=110)
        self.col_text.grid(row=row, column=0, columnspan=4, **pad)

        row += 1
        ttk.Button(f, text="Exportar informe (TXT)", command=lambda: self.export_text(self.col_text)).grid(row=row, column=3, sticky='e', **pad)

    def run_columna(self):
        try:
            params = dict(
                t_muro_mm = float(self.col_t.get()),
                b_mm = float(self.col_b.get()),
                h_mm = float(self.col_h.get()),
                L_libre_m = float(self.col_Ln.get()),
                fc_MPa = float(self.col_fc.get()),
                fy_MPa = float(self.col_fy.get()),
                phi_m = float(self.col_phi.get()),
                mu_friccion = float(self.col_mu.get()),
                Vu_kN = float(self.col_Vu.get()),
                Tu_kN = float(self.col_Tu.get()),
                db_long_mm = float(self.col_db_long.get()),
                db_st_mm = float(self.col_db_st.get()),
            )
        except Exception as e:
            messagebox.showerror("Error de entrada", f"Verifique los datos:\n{e}")
            return

        reporte, out = diseno_columna_confinamiento(params)
        self.col_text.delete('1.0', tk.END)
        self.col_text.insert(tk.END, reporte + "\n")
        self.col_text.insert(tk.END, "\n— RECOMENDACIÓN DE ARMADO —\n")
        self.col_text.insert(tk.END, f"  Barras longitudinales: {out['n_long']} Ø{out['d_long_mm']:.0f} mm  (A_s,prov = {out['As_prov_mm2']:.0f} mm²)\n")
        self.col_text.insert(tk.END, f"  Estribos zona confinada: Ø{float(self.col_db_st.get()):.0f} mm @ ≤ {out['So_max_mm']:.0f} mm en Lo = {out['Lo_min_mm']:.0f} mm desde el nudo\n")
        self.col_text.insert(tk.END, f"  Estribos fuera de Lo: Ø{float(self.col_db_st.get()):.0f} mm @ ≤ {out['s_fuera_max_mm']:.0f} mm\n")

    def _build_solera_tab(self):
        f = self.sol_tab
        pad = {'padx': 6, 'pady': 4}

        row = 0
        ttk.Label(f, text="Espesor de muro t (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.sol_t = ttk.Entry(f); self.sol_t.insert(0, "140")
        self.sol_t.grid(row=row, column=1, **pad)

        ttk.Label(f, text="b_w (mm)").grid(row=row, column=2, sticky='e', **pad)
        self.sol_bw = ttk.Entry(f); self.sol_bw.insert(0, "140")
        self.sol_bw.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="h solera (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.sol_h = ttk.Entry(f); self.sol_h.insert(0, "200")
        self.sol_h.grid(row=row, column=1, **pad)

        ttk.Label(f, text="h losa (mm)").grid(row=row, column=2, sticky='e', **pad)
        self.sol_hlosa = ttk.Entry(f); self.sol_hlosa.insert(0, "180")
        self.sol_hlosa.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="f'c (MPa)").grid(row=row, column=0, sticky='e', **pad)
        self.sol_fc = ttk.Entry(f); self.sol_fc.insert(0, "21")
        self.sol_fc.grid(row=row, column=1, **pad)

        ttk.Label(f, text="f_y (MPa)").grid(row=row, column=2, sticky='e', **pad)
        self.sol_fy = ttk.Entry(f); self.sol_fy.insert(0, "420")
        self.sol_fy.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="φ (tracción)").grid(row=row, column=0, sticky='e', **pad)
        self.sol_phi = ttk.Entry(f); self.sol_phi.insert(0, "0.85")
        self.sol_phi.grid(row=row, column=1, **pad)

        ttk.Label(f, text="T_s (kN)").grid(row=row, column=2, sticky='e', **pad)
        self.sol_Ts = ttk.Entry(f); self.sol_Ts.insert(0, "80")
        self.sol_Ts.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="Ø barras long. (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.sol_db_long = ttk.Entry(f); self.sol_db_long.insert(0, "12")
        self.sol_db_long.grid(row=row, column=1, **pad)

        ttk.Label(f, text="Ø estribo (mm)").grid(row=row, column=2, sticky='e', **pad)
        self.sol_db_st = ttk.Entry(f); self.sol_db_st.insert(0, "8")
        self.sol_db_st.grid(row=row, column=3, **pad)

        ttk.Button(f, text="Calcular", command=self.run_solera).grid(row=row, column=3, sticky='e', **pad)

        row += 1
        self.sol_text = tk.Text(f, height=22, width=110)
        self.sol_text.grid(row=row, column=0, columnspan=4, **pad)

        row += 1
        ttk.Button(f, text="Exportar informe (TXT)", command=lambda: self.export_text(self.sol_text)).grid(row=row, column=3, sticky='e', **pad)

    def run_solera(self):
        try:
            params = dict(
                t_muro_mm = float(self.sol_t.get()),
                bw_mm = float(self.sol_bw.get()),
                h_mm = float(self.sol_h.get()),
                h_losa_mm = float(self.sol_hlosa.get()),
                fc_MPa = float(self.sol_fc.get()),
                fy_MPa = float(self.sol_fy.get()),
                phi_t = float(self.sol_phi.get()),
                Ts_kN = float(self.sol_Ts.get()),
                db_long_mm = float(self.sol_db_long.get()),
                db_st_mm = float(self.sol_db_st.get()),
            )
        except Exception as e:
            messagebox.showerror("Error de entrada", f"Verifique los datos:\n{e}")
            return

        reporte, out = diseno_viga_solera(params)
        self.sol_text.delete('1.0', tk.END)
        self.sol_text.insert(tk.END, reporte + "\n")
        self.sol_text.insert(tk.END, "\n— RECOMENDACIÓN DE ARMADO —\n")
        self.sol_text.insert(tk.END, f"  Barras longitudinales: {out['n_long']} Ø{out['d_long_mm']:.0f} mm  (A_s,prov = {out['As_prov_mm2']:.0f} mm²)\n")
        self.sol_text.insert(tk.END, f"  Estribos en extremos: Ø{float(self.sol_db_st.get()):.0f} mm @ ≤ {out['s_end_mm']:.0f} mm (primer estribo ≤ 100 mm de la cara)\n")

    def export_text(self, text_widget):
        content = text_widget.get('1.0', tk.END).strip()
        if not content:
            messagebox.showwarning("Aviso", "No hay contenido para exportar.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[["Texto", ".txt"]])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Exportado", f"Informe guardado en:\n{file_path}")


def predimension_muro(params):
    """
    Predimensionamiento de muros de albañilería confinada (NTE E.070):
    - t_min según 7.1.1(a) (Zona 1: h/25; Zonas 2 y 3: h/20).
    - Separación máx. entre columnas de confinamiento (7.2.1 b): s_max = min(2·h, 5 m).
    - Área mínima de sección de columna (8.6.3-a.1): Ac ≥ 15·t (en cm²).
    - Dimensiones mínimas de columna/solera (7.2.3 y 7.2.5).
    - Cuantía longitudinal mínima de columnas (E.060 21.4.5.1): ρ ≥ 1% del área de sección.
    """
    zona = params['zona']  # 1, 2 o 3
    h_m = params['h_m']
    L_m = params['L_m']
    t_prop_mm = params.get('t_prop_mm', 0.0)
    fy = params.get('fy_MPa', 420.0)
    fc = params.get('fc_MPa', 21.0)
    db_long = params.get('db_long_mm', 12.0)
    db_st = params.get('db_st_mm', 8.0)

    reporte = []
    out = {}

    reporte.append("PREDIMENSIONAMIENTO DE MUROS (NTE E.070 – Albañilería Confinada)")
    # t_min por zona (E.070 §7.1.1a)
    if zona == 1:
        t_min_mm = h_m * 1000.0 / 25.0
    else:
        t_min_mm = h_m * 1000.0 / 20.0
    # Redondeo práctico a 10 mm
    t_min_mm = math.ceil(t_min_mm / 10.0) * 10.0

    reporte.append(f"1) Altura libre h = {h_m:.2f} m, Longitud de muro L = {L_m:.2f} m, Zona sísmica = {zona}")
    if zona == 1:
        reporte.append("   - t_min = h/25 (E.070 §7.1.1a).")
    else:
        reporte.append("   - t_min = h/20 (E.070 §7.1.1a).")
    reporte.append(f"   → t_min recomendado = {t_min_mm:.0f} mm.")

    # Separación máx. entre columnas (E.070 §7.2.1 b)
    s_max_m = min(2.0 * h_m, 5.0)
    n_cols = max(2, math.floor(L_m / s_max_m) + 1)
    espac_m = L_m / (n_cols - 1)
    reporte.append("2) Separación máxima entre columnas de confinamiento (E.070 §7.2.1 b)")
    reporte.append("   - s_max = min(2·h, 5 m). Si L > s_max, se agregan columnas intermedias.")
    reporte.append(f"   → L = {L_m:.2f} m → n° columnas = {n_cols} (incluye extremos), espaciamiento ≈ {espac_m:.2f} m.")

    # Dimensiones mínimas de columnas/solera (E.070 §7.2.3 y §7.2.5)
    b_col_mm = max(t_min_mm, t_prop_mm) if t_prop_mm else t_min_mm
    h_col_mm = max(150.0, b_col_mm)  # peralte mínimo 150 mm (E.070 §7.2.5)
    # Área mínima por 8.6.3-a.1: Ac ≥ 15·t (t en cm)
    t_cm = b_col_mm / 10.0
    Ac_min_cm2 = 15.0 * t_cm
    Ac_min_mm2 = Ac_min_cm2 * 100.0
    if b_col_mm * h_col_mm < Ac_min_mm2:
        h_col_mm = math.ceil(Ac_min_mm2 / b_col_mm / 10.0) * 10.0

    reporte.append("3) Sección mínima de columna (E.070 §8.6.3-a.1 y §7.2.3, §7.2.5)")
    reporte.append(f"   - b_col = {b_col_mm:.0f} mm, h_col ≥ 150 mm → adoptado h_col = {h_col_mm:.0f} mm.")
    reporte.append(f"   - Ac ≥ 15·t (cm²) → Ac_min = {Ac_min_cm2:.1f} cm² → verificación: b·h = {b_col_mm*h_col_mm/100.0:.1f} cm².")

    # Cuantía mínima de acero longitudinal (E.060 §21.4.5.1) para predimensionar
    Ag_mm2 = b_col_mm * h_col_mm
    As_min_mm2 = 0.01 * Ag_mm2  # 1%
    n_long_min, d_long_use, As_prov_min = choose_bars(As_min_mm2, min_bars=4)

    # Estribos en zona confinada de columnas (E.060 §21.4.5.3)
    menor_dim = min(b_col_mm, h_col_mm)
    So_max_mm = min(8 * d_long_use, 0.5 * menor_dim, 100.0)

    reporte.append("4) Acero sugerido para predimensionamiento")
    reporte.append("   - Cuantía longitudinal mínima ρ ≥ 1% (E.060 §21.4.5.1).")
    reporte.append(f"   → As_min = 1%·Ag = {As_min_mm2:.0f} mm² → provisión: {n_long_min} Ø{d_long_use} → As = {As_prov_min:.0f} mm².")
    reporte.append("   - Estribos en zona confinada: So ≤ min(8·db_long, 0.5·dim_menor, 100 mm) (E.060 §21.4.5.3).")
    reporte.append(f"   → Adoptar estribos Ø{db_st:.0f} @ ≤ {So_max_mm:.0f} mm en extremos; fuera de Lo: ver §21.6.4.5.")

    out.update(dict(
        t_min_mm=t_min_mm,
        s_max_m=s_max_m,
        n_cols=n_cols,
        espac_m=espac_m,
        b_col_mm=b_col_mm,
        h_col_mm=h_col_mm,
        As_min_mm2=As_min_mm2,
        n_long=n_long_min,
        d_long_mm=d_long_use,
        So_max_mm=So_max_mm
    ))

    return "".join(reporte), out


class App(ttk.Frame):
    def _build_muros_tab(self):
        f = self.muros_tab
        pad = {'padx': 6, 'pady': 4}
        row = 0
        ttk.Label(f, text="Zona sísmica (E.030/E.070)").grid(row=row, column=0, sticky='e', **pad)
        self.m_zona = ttk.Combobox(f, values=[1,2,3], state='readonly'); self.m_zona.set(3)
        self.m_zona.grid(row=row, column=1, **pad)

        ttk.Label(f, text="Altura libre h (m)").grid(row=row, column=2, sticky='e', **pad)
        self.m_h = ttk.Entry(f); self.m_h.insert(0, "2.60")
        self.m_h.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="Longitud de muro L (m)").grid(row=row, column=0, sticky='e', **pad)
        self.m_L = ttk.Entry(f); self.m_L.insert(0, "4.80")
        self.m_L.grid(row=row, column=1, **pad)

        ttk.Label(f, text="t propuesto (mm, opcional)").grid(row=row, column=2, sticky='e', **pad)
        self.m_t = ttk.Entry(f); self.m_t.insert(0, "")
        self.m_t.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="f'c (MPa)").grid(row=row, column=0, sticky='e', **pad)
        self.m_fc = ttk.Entry(f); self.m_fc.insert(0, "21")
        self.m_fc.grid(row=row, column=1, **pad)

        ttk.Label(f, text="f_y (MPa)").grid(row=row, column=2, sticky='e', **pad)
        self.m_fy = ttk.Entry(f); self.m_fy.insert(0, "420")
        self.m_fy.grid(row=row, column=3, **pad)

        row += 1
        ttk.Label(f, text="Ø barras long. (mm)").grid(row=row, column=0, sticky='e', **pad)
        self.m_db_long = ttk.Entry(f); self.m_db_long.insert(0, "12")
        self.m_db_long.grid(row=row, column=1, **pad)

        ttk.Label(f, text="Ø estribo (mm)").grid(row=row, column=2, sticky='e', **pad)
        self.m_db_st = ttk.Entry(f); self.m_db_st.insert(0, "8")
        self.m_db_st.grid(row=row, column=3, **pad)

        ttk.Button(f, text="Predimensionar", command=self.run_muros).grid(row=row, column=3, sticky='e', **pad)

        row += 1
        self.m_text = tk.Text(f, height=22, width=110)
        self.m_text.grid(row=row, column=0, columnspan=4, **pad)

        row += 1
        ttk.Button(f, text="Exportar CSV", command=self.exportar_muro_csv).grid(row=row, column=3, sticky='e', **pad)

    def run_muros(self):
        try:
            zona = int(self.m_zona.get())
            h_m = float(self.m_h.get())
            L_m = float(self.m_L.get())
            t_prop = float(self.m_t.get()) if self.m_t.get().strip() != '' else 0.0
            params = dict(
                zona=zona, h_m=h_m, L_m=L_m, t_prop_mm=t_prop,
                fy_MPa=float(self.m_fy.get()), fc_MPa=float(self.m_fc.get()),
                db_long_mm=float(self.m_db_long.get()), db_st_mm=float(self.m_db_st.get())
            )
        except Exception as e:
            messagebox.showerror("Error de entrada", f"Verifique los datos:{e}")
            return

        rep, out = predimension_muro(params)
        self.m_text.delete('1.0', tk.END)
        self.m_text.insert(tk.END, rep + "")
        self._ultimo_predim = out

    def exportar_muro_csv(self):
        if not hasattr(self, '_ultimo_predim'):
            messagebox.showwarning("Aviso", "Ejecute primero el predimensionamiento.")
            return
        out = self._ultimo_predim
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[["CSV", ".csv"]])
        if not file_path:
            return
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["t_min_mm","s_max_m","n_cols","espac_m","b_col_mm","h_col_mm","As_min_mm2","n_long","d_long_mm","So_max_mm"])
            w.writerow([out['t_min_mm'], out['s_max_m'], out['n_cols'], out['espac_m'], out['b_col_mm'], out['h_col_mm'], out['As_min_mm2'], out['n_long'], out['d_long_mm'], out['So_max_mm']])
        messagebox.showinfo("Exportado", f"CSV guardado en:{file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Diseño – Vigas Soleras y Columnas de Confinamiento (RNE E.070/E.060)")
    root.geometry("1000x700")
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = App(root)
    root.mainloop()
