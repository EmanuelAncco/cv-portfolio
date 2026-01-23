import math

# ============================================================================
# HERRAMIENTA DE DISEÑO DE ALBAÑILERÍA CONFINADA
# Columnas y Vigas de Confinamiento
# Basada en NTE E.070 Albañilería y SENCICO
# Autor: Emanuel Ancco
# Fecha: 2025-01-11
# ============================================================================

# CONSTANTES DE DISEÑO - COLUMNAS
AREA_BARRA_3_8 = 0.71  # cm²
AS_MIN_PRACTICO_COL = 4 * AREA_BARRA_3_8  # 2.84 cm²
B_MIN_PRACTICO_COL = 25.0  # cm
ESTRIBO_DIAMETRO_COL = "1/4\""
ESTRIBO_PATRON_COL = "1 @ 0.05m, 4 @ 0.10m, Resto @ 0.25m"

# CONSTANTES DE DISEÑO - VIGAS
# Según E.070 y SENCICO
PERALTE_MIN_VIGA = 17.0  # cm (mínimo práctico)
ESPESOR_VIGA_FACTOR = 1.0  # t_viga = t_muro generalmente
AS_MIN_VIGA_SUP = 2 * AREA_BARRA_3_8  # 2 barras 3/8" (1.42 cm²)
AS_MIN_VIGA_INF = 2 * AREA_BARRA_3_8  # 2 barras 3/8" (1.42 cm²)
ESTRIBO_PATRON_VIGA = "1 @ 0.05m, 4 @ 0.10m, Resto @ 0.25m"

# LÍMITES DE ESBELTEZ (E.070)
H_T_MAX = 20.0  # h/t <= 20


def imprimir_encabezado():
    """Imprime el encabezado del programa"""
    print("\n" + "=" * 80)
    print(" " * 10 + "DISEÑO DE ALBAÑILERÍA CONFINADA")
    print(" " * 15 + "Columnas y Vigas de Confinamiento")
    print(" " * 20 + "NTE E.070 + SENCICO")
    print("=" * 80)


def obtener_datos_entrada():
    """Solicita datos al usuario"""
    print("\n" + "-" * 80)
    print("1. INGRESO DE DATOS")
    print("-" * 80)

    try:
        print("\n1.1. GEOMETRÍA DEL MURO:")
        h_libre_total = float(input("  Altura libre TOTAL del muro (h) en metros: "))
        t_muro = float(input("  Espesor del muro (t) en cm: "))
        longitud_muro = float(input("  Longitud del muro (L) en metros [opcional, 0 si no aplica]: ") or "0")

        print("\n1.2. MATERIALES:")
        fc = float(input("  Resistencia del concreto (f'c) en kg/cm² [175, 210, etc.]: "))
        fy = float(input("  Fluencia del acero (fy) en kg/cm² [4200]: "))
        fm = float(input("  Resistencia de la albañilería (f'm) en kg/cm² [65, 35, etc.]: ") or "0")

        print("\n1.3. COLUMNAS:")
        b_col_prop = float(input("  Peralte propuesto columna (b) en cm [0 = usar mínimo 25 cm]: ") or "0")

        datos = {
            'h_libre_total': h_libre_total,
            't_muro': t_muro,
            'longitud_muro': longitud_muro,
            'fc': fc,
            'fy': fy,
            'fm': fm,
            'b_col_propuesto': b_col_prop
        }

        return datos

    except ValueError:
        print("\n❌ ERROR: Entrada inválida. Use solo números.")
        return None


def verificar_necesidad_viga_intermedia(h_libre_m, t_muro_cm):
    """
    Verifica si se necesita viga de confinamiento intermedia
    Criterio: h/t <= 20 (E.070, Art. 20.a)
    """
    print("\n" + "-" * 80)
    print("2. VERIFICACIÓN DE ESBELTEZ DEL MURO")
    print("-" * 80)

    h_libre_cm = h_libre_m * 100
    esbeltez_actual = h_libre_cm / t_muro_cm

    print(f"\n2.1. Esbeltez actual del muro:")
    print(f"     h/t = {h_libre_cm:.0f} / {t_muro_cm:.1f} = {esbeltez_actual:.2f}")
    print(f"     Límite máximo (E.070): h/t ≤ {H_T_MAX:.0f}")

    if esbeltez_actual <= H_T_MAX:
        print(f"\n     ✓ [CUMPLE] - No se requiere viga intermedia")
        print(f"     El muro puede construirse sin arriostres horizontales intermedios.")
        return False, 1, h_libre_m, None

    else:
        print(f"\n     ✗ [NO CUMPLE] - Se requiere viga(s) de confinamiento intermedia(s)")

        # Calcular número de vigas necesarias
        num_vigas = math.ceil(esbeltez_actual / H_T_MAX) - 1
        num_paños = num_vigas + 1
        h_paño = h_libre_m / num_paños
        esbeltez_nueva = (h_paño * 100) / t_muro_cm

        print(f"\n2.2. Solución propuesta:")
        print(f"     Número de vigas intermedias requeridas: {num_vigas}")
        print(f"     Número de paños resultantes: {num_paños}")
        print(f"     Altura libre por paño: {h_paño:.2f} m = {h_paño * 100:.0f} cm")
        print(f"     Nueva esbeltez por paño: h/t = {esbeltez_nueva:.2f}")

        if esbeltez_nueva <= H_T_MAX:
            print(f"     ✓ Nueva esbeltez cumple (≤ {H_T_MAX:.0f})")
        else:
            print(f"     ⚠️  Verificar: puede requerir más vigas o aumentar espesor")

        # Calcular ubicaciones de vigas
        ubicaciones = []
        for i in range(1, num_vigas + 1):
            ubicacion = i * h_paño
            ubicaciones.append(ubicacion)

        print(f"\n2.3. Ubicación de vigas intermedias (desde nivel inferior):")
        for i, ubic in enumerate(ubicaciones, 1):
            print(f"     Viga #{i}: a {ubic:.2f} m de altura")

        return True, num_vigas, h_paño, ubicaciones


def diseñar_columna_confinamiento(h_libre_paño_m, t_muro, b_propuesto, fc, fy):
    """
    Diseña columna de confinamiento para un paño
    """
    print("\n" + "-" * 80)
    print("3. DISEÑO DE COLUMNAS DE CONFINAMIENTO")
    print("-" * 80)

    # Verificar esbeltez del paño
    h_paño_cm = h_libre_paño_m * 100
    t_min_req = h_paño_cm / H_T_MAX

    print(f"\n3.1. Verificación de esbeltez (por paño):")
    print(f"     h_paño = {h_paño_cm:.0f} cm")
    print(f"     t_mínimo = h / 20 = {t_min_req:.2f} cm")

    if t_muro >= t_min_req:
        print(f"     ✓ t_muro ({t_muro:.1f} cm) ≥ t_mín ({t_min_req:.2f} cm) - CUMPLE")
    else:
        print(f"     ✗ t_muro ({t_muro:.1f} cm) < t_mín ({t_min_req:.2f} cm) - NO CUMPLE")
        print(f"     ⚠️  Aumentar espesor del muro")

    # Dimensiones de columna
    t_col = t_muro

    if b_propuesto < B_MIN_PRACTICO_COL:
        b_col = B_MIN_PRACTICO_COL
        print(f"\n3.2. Dimensiones de columna:")
        print(f"     Peralte mínimo práctico: {B_MIN_PRACTICO_COL} cm")
        print(f"     Se adopta: b = {b_col:.0f} cm")
    else:
        b_col = b_propuesto
        print(f"\n3.2. Dimensiones de columna:")
        print(f"     Se adopta peralte propuesto: b = {b_col:.0f} cm")

    print(f"     Dimensiones finales: {t_col:.0f} cm × {b_col:.0f} cm")

    # Área de concreto
    Ac = t_col * b_col
    Ac_min = 15 * t_col

    print(f"\n3.3. Verificación de área mínima (E.070, Art. 27.3.a.1):")
    print(f"     Ac = {Ac:.2f} cm²")
    print(f"     Ac_mín = 15t = 15 × {t_col:.1f} = {Ac_min:.2f} cm²")

    if Ac >= Ac_min:
        print(f"     ✓ Ac ≥ Ac_mín - CUMPLE")
    else:
        print(f"     ✗ Ac < Ac_mín - NO CUMPLE (aumentar dimensiones)")

    # Acero longitudinal
    As_calc = (0.1 * fc * Ac) / fy
    As_req = max(As_calc, AS_MIN_PRACTICO_COL)

    print(f"\n3.4. Acero longitudinal (E.070, Art. 27.3.a.2):")
    print(f"     As_mín (calculado) = 0.1 f'c Ac / fy = {As_calc:.2f} cm²")
    print(f"     As_mín (práctico) = {AS_MIN_PRACTICO_COL:.2f} cm² (4 Ø 3/8\")")
    print(f"     As requerido = {As_req:.2f} cm²")

    # Selección de barras
    num_barras = max(4, math.ceil(As_req / AREA_BARRA_3_8))
    if num_barras % 2 != 0:
        num_barras += 1

    As_prov = num_barras * AREA_BARRA_3_8

    print(f"\n     Acero adoptado: {num_barras} Ø 3/8\"")
    print(f"     As provisto = {As_prov:.2f} cm²")

    # Estribos
    print(f"\n3.5. Estribos (E.070 + SENCICO):")
    print(f"     Diámetro: {ESTRIBO_DIAMETRO_COL}")
    print(f"     Patrón: {ESTRIBO_PATRON_COL}")

    return {
        't': t_col,
        'b': b_col,
        'Ac': Ac,
        'num_barras': num_barras,
        'As': As_prov,
        'estribo_diam': ESTRIBO_DIAMETRO_COL,
        'estribo_patron': ESTRIBO_PATRON_COL
    }


def diseñar_viga_confinamiento(t_muro, longitud_tramo, fc, fy):
    """
    Diseña viga de confinamiento horizontal
    E.070, Art. 27.3.b
    """
    print("\n" + "-" * 80)
    print("4. DISEÑO DE VIGA(S) DE CONFINAMIENTO")
    print("-" * 80)

    # Dimensiones
    t_viga = t_muro

    # Peralte de viga según criterio práctico
    # Generalmente h_viga = L/10 a L/12 (conservador)
    if longitud_tramo > 0:
        h_calc = (longitud_tramo * 100) / 12  # cm
        h_viga = max(PERALTE_MIN_VIGA, h_calc)
        h_viga = math.ceil(h_viga / 5) * 5  # Redondear a 5 cm

        print(f"\n4.1. Dimensiones de viga:")
        print(f"     Luz de cálculo: {longitud_tramo:.2f} m")
        print(f"     Peralte calculado (L/12): {h_calc:.1f} cm")
        print(f"     Peralte mínimo práctico: {PERALTE_MIN_VIGA:.0f} cm")
        print(f"     Peralte adoptado: {h_viga:.0f} cm")
    else:
        h_viga = 25.0  # Peralte por defecto
        print(f"\n4.1. Dimensiones de viga:")
        print(f"     Sin luz especificada - usando peralte típico")
        print(f"     Peralte adoptado: {h_viga:.0f} cm")

    print(f"     Dimensiones finales: {t_viga:.0f} cm × {h_viga:.0f} cm")

    # Área de concreto
    Ac_viga = t_viga * h_viga

    print(f"     Área de concreto: Ac = {Ac_viga:.2f} cm²")

    # Acero longitudinal (E.070, Art. 27.3.b)
    # Mínimo práctico: 2 barras arriba + 2 barras abajo

    print(f"\n4.2. Acero longitudinal:")
    print(f"     Acero superior: 2 Ø 3/8\" (As = {AS_MIN_VIGA_SUP:.2f} cm²)")
    print(f"     Acero inferior: 2 Ø 3/8\" (As = {AS_MIN_VIGA_INF:.2f} cm²)")
    print(f"     Nota: Mínimo normativo según E.070, Art. 27.3.b")

    # Para diseño más exacto con momentos
    if longitud_tramo > 0:
        # Momento estimado por carga de muro
        # Asumiendo carga uniforme conservadora
        print(f"\n     Nota: Para diseño exacto, considerar:")
        print(f"     - Peso del muro superior")
        print(f"     - Cargas de losa o techo")
        print(f"     - Momento: M = wL²/8 (viga simplemente apoyada)")

    # Estribos
    print(f"\n4.3. Estribos:")
    print(f"     Diámetro: {ESTRIBO_DIAMETRO_COL}")
    print(f"     Patrón: {ESTRIBO_PATRON_VIGA}")
    print(f"     Nota: Cerrar estribos en columnas de confinamiento")

    return {
        't': t_viga,
        'h': h_viga,
        'Ac': Ac_viga,
        'As_sup': AS_MIN_VIGA_SUP,
        'As_inf': AS_MIN_VIGA_INF,
        'barras_sup': '2 Ø 3/8"',
        'barras_inf': '2 Ø 3/8"',
        'estribo_diam': ESTRIBO_DIAMETRO_COL,
        'estribo_patron': ESTRIBO_PATRON_VIGA
    }


def imprimir_resumen_completo(datos, necesita_viga, num_vigas, h_paño,
                              ubicaciones, config_col, config_viga):
    """
    Imprime resumen completo del diseño
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "RESUMEN DEL DISEÑO")
    print("=" * 80)

    print("\n📋 DATOS GENERALES:")
    print(f"   Altura total del muro: {datos['h_libre_total']:.2f} m")
    print(f"   Espesor del muro: {datos['t_muro']:.0f} cm")
    if datos['longitud_muro'] > 0:
        print(f"   Longitud del muro: {datos['longitud_muro']:.2f} m")
    print(f"   f'c = {datos['fc']:.0f} kg/cm²")
    print(f"   fy = {datos['fy']:.0f} kg/cm²")

    print("\n" + "─" * 80)

    if necesita_viga:
        print("\n🔴 CONFIGURACIÓN: MURO CON VIGA(S) INTERMEDIA(S)")
        print(f"\n   Número de vigas intermedias: {num_vigas}")
        print(f"   Número de paños: {num_vigas + 1}")
        print(f"   Altura por paño: {h_paño:.2f} m")

        if ubicaciones:
            print(f"\n   Ubicación de vigas (desde nivel inferior):")
            for i, ubic in enumerate(ubicaciones, 1):
                print(f"     • Viga #{i}: {ubic:.2f} m")
    else:
        print("\n🟢 CONFIGURACIÓN: MURO SIN VIGAS INTERMEDIAS")
        print(f"   Paño único de {datos['h_libre_total']:.2f} m")

    print("\n" + "─" * 80)

    if config_col:
        print("\n📐 COLUMNAS DE CONFINAMIENTO:")
        print(f"   Dimensiones: {config_col['t']:.0f} × {config_col['b']:.0f} cm")
        print(f"   Acero longitudinal: {config_col['num_barras']} Ø 3/8\"")
        print(f"   As = {config_col['As']:.2f} cm²")
        print(f"   Estribos: Ø {config_col['estribo_diam']}")
        print(f"   Patrón: {config_col['estribo_patron']}")

    if necesita_viga and config_viga:
        print("\n" + "─" * 80)
        print("\n🔗 VIGA(S) DE CONFINAMIENTO INTERMEDIA(S):")
        print(f"   Dimensiones: {config_viga['t']:.0f} × {config_viga['h']:.0f} cm")
        print(f"   Acero superior: {config_viga['barras_sup']}")
        print(f"   Acero inferior: {config_viga['barras_inf']}")
        print(f"   Estribos: Ø {config_viga['estribo_diam']}")
        print(f"   Patrón: {config_viga['estribo_patron']}")

    print("\n" + "=" * 80)


def generar_esquema_elevacion(h_total, num_vigas, ubicaciones):
    """
    Genera un esquema ASCII de la elevación del muro
    """
    print("\n" + "─" * 80)
    print("ESQUEMA DE ELEVACIÓN DEL MURO:")
    print("─" * 80)

    print("\n")
    print("     VIGA SOLERA SUPERIOR")
    print("     ═════════════════════")

    if num_vigas > 0:
        for i in range(num_vigas, 0, -1):
            h_desde_base = ubicaciones[i - 1]
            h_desde_sup = h_total - h_desde_base
            print(f"     │                   │  ↑ {h_desde_sup:.2f} m")
            print(f"     │   PAÑO #{i + 1}      │")
            print(f"     │                   │")
            print(f"     ├───────────────────┤  ← VIGA INTERMEDIA #{i}")
            print(f"     │                   │")

        print(f"     │                   │  ↑ {ubicaciones[0]:.2f} m")
        print(f"     │   PAÑO #1         │")
        print(f"     │                   │")
    else:
        print(f"     │                   │  ↑ {h_total:.2f} m")
        print(f"     │   PAÑO ÚNICO      │")
        print(f"     │                   │")

    print("     ═════════════════════")
    print("     SOBRECIMIENTO / CIMIENTO")
    print("\n")


def main():
    """Función principal"""

    imprimir_encabezado()

    # 1. Obtener datos
    datos = obtener_datos_entrada()
    if not datos:
        return

    # 2. Verificar necesidad de viga intermedia
    necesita_viga, num_vigas, h_paño, ubicaciones = verificar_necesidad_viga_intermedia(
        datos['h_libre_total'],
        datos['t_muro']
    )

    # 3. Diseñar columnas
    config_col = diseñar_columna_confinamiento(
        h_paño,
        datos['t_muro'],
        datos['b_col_propuesto'],
        datos['fc'],
        datos['fy']
    )

    # 4. Diseñar vigas si es necesario
    config_viga = None
    if necesita_viga:
        config_viga = diseñar_viga_confinamiento(
            datos['t_muro'],
            datos['longitud_muro'],
            datos['fc'],
            datos['fy']
        )

    # 5. Resumen
    imprimir_resumen_completo(
        datos, necesita_viga, num_vigas, h_paño,
        ubicaciones, config_col, config_viga
    )

    # 6. Esquema
    if necesita_viga and ubicaciones:
        generar_esquema_elevacion(datos['h_libre_total'], num_vigas, ubicaciones)

    print("\n✅ Diseño completado.")
    print("\n📌 RECOMENDACIONES:")
    print("   • Verificar conexión columna-viga con chicotes cada 3-4 hiladas")
    print("   • Dentado del muro en columnas (mínimo 5 cm)")
    print("   • Viga intermedia debe anclarse adecuadamente en columnas")
    print("   • Considerar juntas de control según longitud del muro")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()