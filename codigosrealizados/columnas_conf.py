import math

# Constantes de diseño
# Área de 1 barra de 3/8" en cm^2
AREA_BARRA_3_8 = 0.71
# Área mínima práctica (SENCICO): 4 barras de 3/8"
AS_MIN_PRACTICO = 4 * AREA_BARRA_3_8  # 2.84 cm^2
# Peralte mínimo práctico (SENCICO) en cm
B_MIN_PRACTICO = 25.0
# Diámetro de estribo
ESTRIBO_DIAMETRO = "1/4\""
# Patrón de estribos (SENCICO)
ESTRIBO_PATRON = "1 @ 0.05m, 4 @ 0.10m, Resto @ 0.25m"


def verificar_esbeltez(h_libre_m, t_muro_cm):
    """
    Verifica la regla de esbeltez h/t <= 20 (NTE E.070).
    Referencia:
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
    Referencia:
    """
    t_columna = t_muro_cm

    if b_propuesto_cm < B_MIN_PRACTICO:
        b_columna = B_MIN_PRACTICO
        b_mensaje = f"Peralte propuesto ({b_propuesto_cm} cm) es menor al mínimo práctico. Se adopta b = {B_MIN_PRACTICO} cm."
    else:
        b_columna = b_propuesto_cm
        b_mensaje = f"Se adopta peralte propuesto: b = {b_columna} cm."

    # Verificación de Área Mínima (E.070)
    Ac = t_columna * b_columna
    Ac_min_req = 15 * t_columna

    if Ac >= Ac_min_req:
        status = "[CUMPLE]"
        ac_mensaje = f"Ac ({Ac:.2f} cm²) >= Ac_min ({Ac_min_req:.2f} cm²)"
    else:
        status = "[NO CUMPLE]"  # Teóricamente imposible si b_min=25
        ac_mensaje = f"Ac ({Ac:.2f} cm²) < Ac_min ({Ac_min_req:.2f} cm²)"

    return t_columna, b_columna, Ac, b_mensaje, Ac_min_req, status, ac_mensaje


def calcular_acero_longitudinal(Ac, f_c, f_y):
    """
    Calcula el acero longitudinal mínimo (NTE E.070, Art. 27.3.a.2).
    Compara el A_s calculado con el A_s práctico (SENCICO).
    Referencia: [1, 3, 8]
    """
    # Criterio 1: Mínimo Normativo (E.070)
    # As_min_calc = (0.1 * f'c * Ac) / fy
    try:
        As_min_calc = (0.1 * f_c * Ac) / f_y
    except ZeroDivisionError:
        return 0, 0, "", "fy no puede ser cero."

    # Criterio 2: Mínimo Práctico (SENCICO)
    # AS_MIN_PRACTICO = 2.84 cm^2 (4 phi 3/8")

    # Se adopta el MÁXIMO de los dos
    As_requerido = max(As_min_calc, AS_MIN_PRACTICO)

    # Definir recomendación de barras
    if As_requerido == AS_MIN_PRACTICO:
        recomendacion = f"4 barras de 3/8\" (As = {AS_MIN_PRACTICO:.2f} cm²)"
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

    return As_min_calc, As_requerido, recomendacion


def definir_estribos():
    """
    Define el refuerzo transversal prescriptivo (SENCICO / E.070).
    Referencia:
    """
    return ESTRIBO_DIAMETRO, ESTRIBO_PATRON


def generar_memoria_calculo():
    """
    Función principal que solicita datos y genera la memoria.
    """

    print("*****************************************************************")
    print("** HERRAMIENTA DE PREDIMENSIONAMIENTO DE COLUMNAS DE CONFINAMIENTO **")
    print("**          Basada en NTE E.070 Albañilería y SENCICO          **")
    print("*****************************************************************")

    try:
        h_libre_m = float(input("Ingrese Altura Libre de Muro (h_libre) en metros (ej. 2.45): "))
        t_muro_cm = float(input("Ingrese Espesor de Muro (t_muro) en cm (ej. 13): "))
        f_c = int(input("Ingrese Resistencia del Concreto (f'c) en kg/cm² (ej. 175): "))
        f_y = int(input("Ingrese Fluencia del Acero (fy) en kg/cm² (ej. 4200): "))
        b_propuesto_cm = float(input("Ingrese Peralte Propuesto (b) en cm (o 0 para usar mínimo 25 cm): "))
    except ValueError:
        print("\n Entrada inválida. Por favor ingrese solo números.")
        return

    # --- INICIO DE CÁLCULOS ---

    # 1. Esbeltez
    t_min_esbeltez, status_esbeltez, msg_esbeltez = verificar_esbeltez(h_libre_m, t_muro_cm)

    # 2. Dimensiones
    t_col, b_col, Ac_col, msg_b, Ac_min_norma, status_ac_min, msg_ac_min = calcular_dimensiones(t_muro_cm,
                                                                                                b_propuesto_cm)

    # 3. Acero Longitudinal
    As_calc, As_req, As_recom = calcular_acero_longitudinal(Ac_col, f_c, f_y)

    # 4. Estribos
    est_diam, est_patron = definir_estribos()

    # --- IMPRESIÓN DE MEMORIA DE CÁLCULO ---

    print("\n\n" + "=" * 50)
    print("  MEMORIA DE CÁLCULO: PREDIMENSIONAMIENTO DE COLUMNA")
    print("=" * 50)
    print("\n1. DATOS DE ENTRADA:")
    print("---------------------------------------------------------")
    print(f"- Altura Libre de Muro (h_libre):   {h_libre_m} m")
    print(f"- Espesor de Muro (t_muro):        {t_muro_cm} cm")
    print(f"- Resistencia Concreto (f'c):     {f_c} kg/cm²")
    print(f"- Fluencia Acero (fy):            {f_y} kg/cm²")
    print(f"- Peralte Propuesto (b_prop):     {b_propuesto_cm} cm")

    print("\n2. VERIFICACIÓN GEOMÉTRICA (NTE E.070, Art. 20 y 27):")
    print("---------------------------------------------------------")
    print("2.1. Verificación de Esbeltez (h/t <= 20):")
    print(f"     - t_minimo_req = h / 20 = ({h_libre_m * 100} cm) / 20 = {t_min_esbeltez:.2f} cm")
    print(f"     - Verificación: {msg_esbeltez}")
    print(f"     - Resultado: {status_esbeltez} ")

    if status_esbeltez == "[NO CUMPLE]":
        print("\nADVERTENCIA: El espesor del muro no cumple la esbeltez. Rediseñar.")
        return

    print("\n2.2. Dimensiones de Columna (t x b):")
    print(f"     - Espesor (t) = t_muro = {t_col} cm")
    print(f"     - {msg_b}")
    print(f"     - Dimensiones Adoptadas: {t_col:.1f} cm x {b_col:.1f} cm")

    print("\n2.3. Verificación de Área Mínima (Ac >= 15t):")
    print(f"     - Area de Concreto (Ac) = {t_col:.1f} * {b_col:.1f} = {Ac_col:.2f} cm²")
    print(f"     - Area Mínima Req. (Ac_min) = 15 * t = 15 * {t_col:.1f} = {Ac_min_norma:.2f} cm²")
    print(f"     - Verificación: {msg_ac_min}")
    print(f"     - Resultado: {status_ac_min} ")

    print("\n3. CÁLCULO DE REFUERZO (NTE E.070, Art. 27.3):")
    print("---------------------------------------------------------")
    print("3.1. Acero Longitudinal (Vertical):")
    print(f"     - A_s_min (calculado, E.070) = (0.1 * f'c * Ac) / fy")
    print(f"     - A_s_min (calculado) = (0.1 * {f_c} * {Ac_col:.2f}) / {f_y} = {As_calc:.2f} cm² ")
    print(f"     - A_s_min (práctico, SENCICO) = {AS_MIN_PRACTICO:.2f} cm² (4 phi 3/8\") ")
    print(f"     - A_s_Requerido = max({As_calc:.2f} cm², {AS_MIN_PRACTICO:.2f} cm²) = {As_req:.2f} cm²")
    print(f"\n     -")

    print("\n3.2. Acero Transversal (Estribos):")
    print(f"     - Diámetro Mínimo: {est_diam} ")
    print(f"     - Espaciamiento (Patrón SENCICO, cumple E.070):")
    print(f"       [{est_patron}]")
    print(f"     - Zona Confinada (Lo) = 45 cm ")

    print("\n\n" + "=" * 50)
    print("         FIN DE LA MEMORIA DE CÁLCULO")
    print("=" * 50)


# --- Ejecutar la herramienta ---
if __name__ == "__main__":
    generar_memoria_calculo()