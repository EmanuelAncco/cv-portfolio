import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math

# --- CONSTANTES DE DISEÑO (Basadas en SENCICO y E.070) ---
# Área de 1 barra de 3/8" en cm^2
AREA_BARRA_3_8 = 0.71
# Área mínima práctica (SENCICO): 4 barras de 3/8" [3]
AS_MIN_PRACTICO = 4 * AREA_BARRA_3_8  # 2.84 cm^2
# Peralte mínimo práctico (SENCICO) en cm [3]
B_MIN_PRACTICO = 25.0
# Diámetro de estribo [4, 3]
ESTRIBO_DIAMETRO = "1/4\""
# Patrón de estribos (SENCICO) [4, 3]
ESTRIBO_PATRON = "1 @ 0.05m, 4 @ 0.10m, Resto @ 0.25m"


# --- FUNCIONES DE LÓGICA DE CÁLCULO (Idénticas a la versión anterior) ---

def verificar_esbeltez(h_libre_m, t_muro_cm):
    """
    Verifica la regla de esbeltez h/t <= 20 (NTE E.070).
    Referencia: [1, 2]
    """
    h_libre_cm = h_libre_m * 100
    t_min_req = h_libre_cm / 20.0

    if t_muro_cm >= t_min_req:
        status = "[CUMPLE]"
        mensaje = f"t_muro ({t_muro_cm} cm) >= t_min_req ({t_min_req:.2f} cm)"
    else:
        status = "[NO CUMPLE]"
        mensaje = f"t_muro ({t_muro_cm} cm) < t_min_req ({t_min_req:.2f} cm). Aumentar espesor de muro."

    return t_min_req, status, mensaje


def calcular_dimensiones(t_muro_cm, b_propuesto_cm):
    """
    Define dimensiones (t, b) y verifica Ac >= 15t (NTE E.070, Art. 27.3.a.1).
    Adopta b_min = 25 cm (SENCICO).
    Referencia: [4, 3]
    """
    t_columna = t_muro_cm

    if b_propuesto_cm < B_MIN_PRACTICO:
        b_columna = B_MIN_PRACTICO
        b_mensaje = f"Peralte propuesto ({b_propuesto_cm} cm) es menor al mínimo práctico. Se adopta b = {B_MIN_PRACTICO} cm. [3]"
    else:
        b_columna = b_propuesto_cm
        b_mensaje = f"Se adopta peralte propuesto: b = {b_columna} cm."

    # Verificación de Área Mínima (E.070)
    Ac = t_columna * b_columna
    Ac_min_req = 15 * t_columna

    if Ac >= Ac_min_req:
        status = "[CUMPLE]"
        ac_mensaje = f"Ac ({Ac:.2f} cm²) >= Ac_min ({Ac_min_req:.2f} cm²) [4]"
    else:
        status = "[NO CUMPLE]"
        ac_mensaje = f"Ac ({Ac:.2f} cm²) < Ac_min ({Ac_min_req:.2f} cm²)"

    return t_columna, b_columna, Ac, b_mensaje, Ac_min_req, status, ac_mensaje


def calcular_acero_longitudinal(Ac, f_c, f_y):
    """
    Calcula el acero longitudinal mínimo (NTE E.070, Art. 27.3.a.2).
    Compara el A_s calculado con el A_s práctico (SENCICO).
    Referencia: [4, 3, 1]
    """
    try:
        # Criterio 1: Mínimo Normativo (E.070) [4]
        As_min_calc = (0.1 * f_c * Ac) / f_y
    except ZeroDivisionError:
        return 0, 0, "", "fy no puede ser cero."

    # Criterio 2: Mínimo Práctico (SENCICO) [3]
    # AS_MIN_PRACTICO = 2.84 cm^2 (4 phi 3/8")

    # Se adopta el MÁXIMO de los dos
    As_requerido = max(As_min_calc, AS_MIN_PRACTICO)

    # Definir recomendación de barras
    if As_requerido <= AS_MIN_PRACTICO:
        recomendacion = f"4 barras de 3/8\" (As = {AS_MIN_PRACTICO:.2f} cm²)"
        msg_criterio = f"Gobierna el Mínimo Práctico SENCICO (2.84 cm² > {As_min_calc:.2f} cm²)"
    else:
        # Si se requiere más que el mínimo, calcular barras de 3/8"
        num_barras_3_8 = math.ceil(As_requerido / AREA_BARRA_3_8)
        # Asegurar número par >= 4
        if num_barras_3_8 < 4:
            num_barras_3_8 = 4
        elif num_barras_3_8 % 2 != 0:
            num_barras_3_8 += 1

        As_provisto = num_barras_3_8 * AREA_BARRA_3_8
        recomendacion = f"{num_barras_3_8} barras de 3/8\" (As = {As_provisto:.2f} cm²)"
        msg_criterio = f"Gobierna el Cálculo E.070 ({As_min_calc:.2f} cm² > 2.84 cm²)"

    return As_min_calc, As_requerido, recomendacion, msg_criterio


def definir_estribos():
    """
    Define el refuerzo transversal prescriptivo (SENCICO / E.070).
    Referencia: [4, 3]
    """
    return ESTRIBO_DIAMETRO, ESTRIBO_PATRON


# --- FUNCIÓN PRINCIPAL DE LA GUI ---

def generar_memoria_gui():
    """
    Toma los datos de la GUI, ejecuta los cálculos y muestra
    la memoria en el widget de texto.
    """

    # 1. Limpiar la salida anterior
    output_text.config(state='normal')
    output_text.delete('1.0', tk.END)

    try:
        # 2. Obtener datos de los campos de entrada
        h_libre_m = float(entry_h.get())
        t_muro_cm = float(entry_t.get())
        f_c = int(entry_fc.get())
        f_y = int(entry_fy.get())
        b_propuesto_cm = float(entry_b.get())
    except ValueError:
        messagebox.showerror("Error de Entrada", "Por favor, ingrese solo números válidos en todos los campos.")
        output_text.config(state='disabled')
        return

    # 3. Ejecutar los cálculos
    t_min_esbeltez, status_esbeltez, msg_esbeltez = verificar_esbeltez(h_libre_m, t_muro_cm)
    t_col, b_col, Ac_col, msg_b, Ac_min_norma, status_ac_min, msg_ac_min = calcular_dimensiones(t_muro_cm,
                                                                                                b_propuesto_cm)
    As_calc, As_req, As_recom, msg_as_criterio = calcular_acero_longitudinal(Ac_col, f_c, f_y)
    est_diam, est_patron = definir_estribos()

    # 4. Mostrar advertencia si la esbeltez no cumple
    if status_esbeltez == "[NO CUMPLE]":
        messagebox.showwarning("Advertencia de Esbeltez",
                               f"{msg_esbeltez}\nSe recomienda rediseñar el espesor del muro.")

    # 5. Construir la cadena de texto de la memoria de cálculo
    memoria_str = ""
    memoria_str += "==========================================================\n"
    memoria_str += "  MEMORIA DE CÁLCULO: PREDIMENSIONAMIENTO DE COLUMNA\n"
    memoria_str += "==========================================================\n\n"

    memoria_str += "1. DATOS DE ENTRADA:\n"
    memoria_str += "---------------------------------------------------------\n"
    memoria_str += f"- Altura Libre de Muro (h_libre):   {h_libre_m} m\n"
    memoria_str += f"- Espesor de Muro (t_muro):        {t_muro_cm} cm\n"
    memoria_str += f"- Resistencia Concreto (f'c):     {f_c} kg/cm²\n"
    memoria_str += f"- Fluencia Acero (fy):            {f_y} kg/cm²\n"
    memoria_str += f"- Peralte Propuesto (b_prop):     {b_propuesto_cm} cm\n\n"

    memoria_str += "2. VERIFICACIÓN GEOMÉTRICA (NTE E.070, Art. 20 y 27):\n"
    memoria_str += "---------------------------------------------------------\n"
    memoria_str += "2.1. Verificación de Esbeltez (h/t <= 20): [1, 2]\n"
    memoria_str += f"     - t_minimo_req = h / 20 = ({h_libre_m * 100} cm) / 20 = {t_min_esbeltez:.2f} cm\n"
    memoria_str += f"     - Verificación: {msg_esbeltez}\n"
    memoria_str += f"     - Resultado: {status_esbeltez}\n\n"

    memoria_str += "2.2. Dimensiones de Columna (t x b): [3]\n"
    memoria_str += f"     - Espesor (t) = t_muro = {t_col:.1f} cm\n"
    memoria_str += f"     - {msg_b}\n"
    memoria_str += f"     - Dimensiones Adoptadas: {t_col:.1f} cm x {b_col:.1f} cm\n\n"

    memoria_str += "2.3. Verificación de Área Mínima (Ac >= 15t): [4]\n"
    memoria_str += f"     - Area de Concreto (Ac) = {t_col:.1f} * {b_col:.1f} = {Ac_col:.2f} cm²\n"
    memoria_str += f"     - Area Mínima Req. (Ac_min) = 15 * t = 15 * {t_col:.1f} = {Ac_min_norma:.2f} cm²\n"
    memoria_str += f"     - Verificación: {msg_ac_min}\n"
    memoria_str += f"     - Resultado: {status_ac_min}\n\n"

    memoria_str += "3. CÁLCULO DE REFUERZO (NTE E.070, Art. 27.3):\n"
    memoria_str += "---------------------------------------------------------\n"
    memoria_str += "3.1. Acero Longitudinal (Vertical): [4, 3]\n"
    memoria_str += f"     - A_s_min (calculado, E.070) = (0.1 * f'c * Ac) / fy\n"
    memoria_str += f"     - A_s_min (calculado) = (0.1 * {f_c} * {Ac_col:.2f}) / {f_y} = {As_calc:.2f} cm² [4]\n"
    memoria_str += f"     - A_s_min (práctico, SENCICO) = {AS_MIN_PRACTICO:.2f} cm² (4 phi 3/8\") [3]\n"
    memoria_str += f"     - Criterio: {msg_as_criterio}\n"
    memoria_str += f"     - A_s_Requerido = max({As_calc:.2f} cm², {AS_MIN_PRACTICO:.2f} cm²) = {As_req:.2f} cm²\n"
    memoria_str += f"\n     - REFUERZO ADOPTADO: {As_recom}\n\n"

    memoria_str += "3.2. Acero Transversal (Estribos): [4, 3]\n"
    memoria_str += f"     - Diámetro Mínimo: {est_diam}\n"
    memoria_str += f"     - Espaciamiento (Patrón SENCICO, cumple E.070):\n"
    memoria_str += f"       [{est_patron}]\n"
    memoria_str += f"     - (Zona Confinada (Lo) = 45 cm)\n\n"

    memoria_str += "==========================================================\n"
    memoria_str += "         FIN DE LA MEMORIA DE CÁLCULO\n"
    memoria_str += "==========================================================\n"

    # 6. Insertar texto en la GUI y bloquear para edición
    output_text.insert(tk.END, memoria_str)
    output_text.config(state='disabled')


# --- CONFIGURACIÓN DE LA INTERFAZ GRÁFICA (GUI) ---

# Ventana principal
root = tk.Tk()
root.title("Calculadora de Predimensionamiento de Columnas de Confinamiento (E.070 / SENCICO)")
root.geometry("700x750")

# Estilo
style = ttk.Style()
style.configure('TButton', font=('Arial', 10, 'bold'), padding=10)
style.configure('TLabel', font=('Arial', 10), padding=5)
style.configure('TEntry', font=('Arial', 10), padding=5)
style.configure('TFrame', padding=10)
style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

# Frame principal
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Frame de Entradas
input_frame = ttk.Frame(main_frame, borderwidth=2, relief="groove")
input_frame.pack(fill=tk.X, padx=10, pady=10)
input_frame.columnconfigure(1, weight=1)  # Columna de entradas se expande

ttk.Label(input_frame, text="1. DATOS DE ENTRADA", style='Header.TLabel').grid(row=0, column=0, columnspan=3,
                                                                               sticky='w', pady=10)

# h_libre
ttk.Label(input_frame, text="Altura Libre de Muro (h_libre) en metros:").grid(row=1, column=0, sticky='e')
entry_h = ttk.Entry(input_frame, width=15)
entry_h.grid(row=1, column=1, sticky='w', padx=5)
entry_h.insert(0, "2.45")
ttk.Label(input_frame, text="ej. 2.45").grid(row=1, column=2, sticky='w')

# t_muro
ttk.Label(input_frame, text="Espesor de Muro (t_muro) en cm:").grid(row=2, column=0, sticky='e')
entry_t = ttk.Entry(input_frame, width=15)
entry_t.grid(row=2, column=1, sticky='w', padx=5)
entry_t.insert(0, "13")
ttk.Label(input_frame, text="ej. 13 (soga)").grid(row=2, column=2, sticky='w')

# f'c
ttk.Label(input_frame, text="Resistencia Concreto (f'c) en kg/cm²:").grid(row=3, column=0, sticky='e')
entry_fc = ttk.Entry(input_frame, width=15)
entry_fc.grid(row=3, column=1, sticky='w', padx=5)
entry_fc.insert(0, "175")
ttk.Label(input_frame, text="ej. 175").grid(row=3, column=2, sticky='w')

# fy
ttk.Label(input_frame, text="Fluencia Acero (fy) en kg/cm²:").grid(row=4, column=0, sticky='e')
entry_fy = ttk.Entry(input_frame, width=15)
entry_fy.grid(row=4, column=1, sticky='w', padx=5)
entry_fy.insert(0, "4200")
ttk.Label(input_frame, text="ej. 4200 (Grado 60)").grid(row=4, column=2, sticky='w')

# b_propuesto
ttk.Label(input_frame, text="Peralte Propuesto (b) en cm:").grid(row=5, column=0, sticky='e')
entry_b = ttk.Entry(input_frame, width=15)
entry_b.grid(row=5, column=1, sticky='w', padx=5)
entry_b.insert(0, "25")
ttk.Label(input_frame, text="ej. 25 (mínimo práctico SENCICO)").grid(row=5, column=2, sticky='w')

# Botón de Cálculo
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.X, padx=10, pady=5)
calc_button = ttk.Button(button_frame, text="GENERAR MEMORIA DE CÁLCULO", command=generar_memoria_gui)
calc_button.pack(expand=True, fill=tk.X)

# Frame de Salida
output_frame = ttk.Frame(main_frame)
output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
ttk.Label(output_frame, text="2. MEMORIA DE CÁLCULO GENERADA", style='Header.TLabel').pack(anchor='w')

# Texto de Salida
output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=80, height=30, font=("Courier New", 9))
output_text.pack(fill=tk.BOTH, expand=True, pady=5)
output_text.config(state='disabled')  # Inicia deshabilitado

# Iniciar la aplicación
root.mainloop()