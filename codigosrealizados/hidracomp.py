# -*- coding: utf-8 -*-
"""
===================================================================
== CALCULADORA HIDRÁULICA DIRECTA PARA DISEÑO DE CANAL Y MODELO ==
===================================================================

Este script realiza todos los cálculos necesarios de forma secuencial
y muestra los resultados finales directamente en la consola.

Propósito: Proveer una herramienta rápida para obtener los valores
finales y facilitar la comprobación manual.

Instrucciones:
1. Modifique los valores en la sección 'PARÁMETROS DE ENTRADA'.
2. Ejecute el script en una terminal con: python calculadora_hidraulica.py
3. Revise el resumen de resultados impreso en la consola.
"""
import math

# --- 1. PARÁMETROS DE ENTRADA (MODIFICAR AQUÍ) ---
# -------------------------------------------------------------------
# Datos del Prototipo (Canal Real)
CAUDAL_DISENO = 0.87142  # Caudal de diseño (m^3/s)
PENDIENTE_CANAL = 0.00373  # Pendiente longitudinal del tramo (m/m)
COEFICIENTE_MANNING = 0.014  # Coeficiente de rugosidad para concreto (adimensional)
LONGITUD_PROTOTIPO = 15  # Longitud total del tramo del canal (m)

# Datos del Modelo Físico (Maqueta)
LONGITUD_MODELO = 1.0  # Longitud máxima deseada para la maqueta (m)

# Constantes Físicas
GRAVEDAD = 9.81  # Aceleración de la gravedad (m/s^2)


# -------------------------------------------------------------------


# --- 2. FUNCIONES DE CÁLCULO (LÓGICA INTERNA) ---

def calcular_seccion_optima(Q, n, S):
    """Calcula b y y para máxima eficiencia hidráulica (b=2y)."""
    try:
        y = ((Q * n) / (2 ** (1 / 3) * S ** 0.5)) ** (3 / 8)
        b = 2 * y
        return b, y
    except (ValueError, ZeroDivisionError):
        return None, None


def calcular_minima_infiltracion(Q, n, S):
    """Calcula b y y para mínima infiltración (b=4y)."""
    try:
        y = ((Q * n) / (4 * (2 / 3) ** (2 / 3) * S ** 0.5)) ** (3 / 8)
        b = 4 * y
        return b, y
    except (ValueError, ZeroDivisionError):
        return None, None


def calcular_parametros_hidraulicos(b, y, Q):
    """Calcula las propiedades hidráulicas resultantes."""
    if y is None or b is None or y <= 0 or b <= 0:
        return {}
    area = b * y
    perimetro = b + 2 * y
    radio_h = area / perimetro
    velocidad = Q / area
    froude = velocidad / math.sqrt(GRAVEDAD * y)
    regimen = "Subcrítico" if froude < 1 else "Supercrítico" if froude > 1 else "Crítico"
    return {
        "Área (A)": area, "Perímetro (P)": perimetro, "Radio Hidráulico (R)": radio_h,
        "Velocidad (V)": velocidad, "Ancho (b)": b, "Tirante (y)": y,
        "Número de Froude (Fr)": froude, "Régimen": regimen
    }


def calcular_escalas(Lp, Lm):
    """Calcula las escalas dimensionales por semejanza de Froude."""
    lambda_L = Lp / Lm
    return {"λ_L": lambda_L, "λ_V": lambda_L ** 0.5, "λ_Q": lambda_L ** 2.5}


def escalar_modelo(prototipo_params, escalas, Qp):
    """Calcula los parámetros del modelo a escala."""
    b_p, y_p, V_p = prototipo_params["Ancho (b)"], prototipo_params["Tirante (y)"], prototipo_params["Velocidad (V)"]
    lambda_L, lambda_V, lambda_Q = escalas["λ_L"], escalas["λ_V"], escalas["λ_Q"]
    return {
        "Ancho (b_m)": b_p / lambda_L, "Tirante (y_m)": y_p / lambda_L,
        "Velocidad (V_m)": V_p / lambda_V, "Caudal (Q_m)": Qp / lambda_Q
    }


# --- 3. EJECUCIÓN PRINCIPAL Y PRESENTACIÓN DE RESULTADOS ---

if __name__ == "__main__":
    # --- PASO A: CÁLCULO DE DIMENSIONES DEL PROTOTIPO ---
    b_opt, y_opt = calcular_seccion_optima(CAUDAL_DISENO, COEFICIENTE_MANNING, PENDIENTE_CANAL)
    params_optimos = calcular_parametros_hidraulicos(b_opt, y_opt, CAUDAL_DISENO)

    b_inf, y_inf = calcular_minima_infiltracion(CAUDAL_DISENO, COEFICIENTE_MANNING, PENDIENTE_CANAL)
    params_infiltracion = calcular_parametros_hidraulicos(b_inf, y_inf, CAUDAL_DISENO)

    # --- PASO B: ANÁLISIS DIMENSIONAL (Basado en el diseño óptimo) ---
    escalas_calculadas = calcular_escalas(LONGITUD_PROTOTIPO, LONGITUD_MODELO)

    # --- PASO C: CÁLCULO DE PARÁMETROS DEL MODELO A ESCALA ---
    params_modelo = escalar_modelo(params_optimos, escalas_calculadas, CAUDAL_DISENO)

    # --- PASO D: IMPRESIÓN DEL RESUMEN DE SALIDAS ---
    print("\n" + "=" * 70)
    print("== RESUMEN DE CÁLCULOS HIDRÁULICOS PARA VERIFICACIÓN MANUAL ==")
    print("=" * 70)

    print("\n[ 1 ] PARÁMETROS DE ENTRADA UTILIZADOS:")
    print(f"    - Caudal (Q):                {CAUDAL_DISENO:.4f} m^3/s")
    print(f"    - Pendiente (S):             {PENDIENTE_CANAL:.5f} m/m")
    print(f"    - Rugosidad (n):             {COEFICIENTE_MANNING:.3f}")
    print(f"    - Longitud Prototipo (Lp):   {LONGITUD_PROTOTIPO:.2f} m")
    print(f"    - Longitud Modelo (Lm):      {LONGITUD_MODELO:.2f} m")

    print("\n[ 2 ] RESULTADOS DEL PROTOTIPO (CANAL REAL):")

    print("\n    --- a) Diseño por Máxima Eficiencia Hidráulica (b=2y) ---")
    if params_optimos:
        for key, value in params_optimos.items():
            if isinstance(value, float):
                print(f"        - {key:<25}: {value:.4f}")
            else:
                print(f"        - {key:<25}: {value}")
    else:
        print("        Error en el cálculo de sección óptima.")

    print("\n    --- b) Diseño por Mínima Infiltración (b=4y) ---")
    if params_infiltracion:
        for key, value in params_infiltracion.items():
            if isinstance(value, float):
                print(f"        - {key:<25}: {value:.4f}")
            else:
                print(f"        - {key:<25}: {value}")
    else:
        print("        Error en el cálculo de mínima infiltración.")

    print("\n[ 3 ] RESULTADOS DEL ANÁLISIS DIMENSIONAL:")
    print(f"    - Escala de Longitud (λ_L):  {escalas_calculadas['λ_L']:.2f}")
    print(f"    - Escala de Velocidad (λ_V): {escalas_calculadas['λ_V']:.4f}")
    print(f"    - Escala de Caudal (λ_Q):    {escalas_calculadas['λ_Q']:.2f}")

    print("\n[ 4 ] RESULTADOS DEL MODELO FÍSICO A ESCALA (MAQUETA):")
    print("    (Basado en el diseño de Máxima Eficiencia)")
    if params_modelo:
        # Conversión a unidades prácticas de laboratorio
        ancho_cm = params_modelo["Ancho (b_m)"] * 100
        tirante_cm = params_modelo["Tirante (y_m)"] * 100
        caudal_lps = params_modelo["Caudal (Q_m)"] * 1000

        print(f"    - Ancho de solera (b_m):     {params_modelo['Ancho (b_m)']:.4f} m  ({ancho_cm:.2f} cm)")
        print(f"    - Tirante de agua (y_m):     {params_modelo['Tirante (y_m)']:.4f} m  ({tirante_cm:.2f} cm)")
        print(f"    - Velocidad del flujo (V_m): {params_modelo['Velocidad (V_m)']:.4f} m/s")
        print(f"    - Caudal a simular (Q_m):    {params_modelo['Caudal (Q_m)']:.6f} m³/s ({caudal_lps:.3f} L/s)")
    else:
        print("    Error en el cálculo del modelo a escala.")

    print("\n" + "=" * 70)
    print("== CÁLCULOS FINALIZADOS ==")
    print("=" * 70 + "\n")
