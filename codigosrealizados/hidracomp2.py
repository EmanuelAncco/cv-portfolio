# -*- coding: utf-8 -*-
"""
================================================================================
== CALCULADORA HIDRÁULICA INVERSA (VERSIÓN AJUSTABLE) ==
================================================================================

Este script resuelve el problema inverso, pero con un enfoque en la
exploración de escenarios. Permite ajustar fácilmente la escala de longitud
para encontrar un tamaño de prototipo que sea realista y relevante para el
contexto de estudio deseado.

Basado en la crítica de que un prototipo de 8m es demasiado grande, este
script utiliza una escala más moderada para obtener un canal real más pequeño
y común.

Metodología:
1. Definir los parámetros del Modelo de laboratorio.
2. Usar la Ecuación de Manning para encontrar el tirante y la velocidad en el modelo.
3. Calcular el Fr_m.
4. Definir una ESCALA DE LONGITUD AJUSTABLE (λ_L).
5. Escalar inversamente para encontrar las características del prototipo.
6. Validar que la semejanza se cumple (Fr_p = Fr_m) y que los parámetros del
   prototipo (especialmente su ancho y rugosidad) son realistas.
"""
import math
from scipy.optimize import fsolve

# --- 1. PARÁMETROS DE ENTRADA (MODIFICAR AQUÍ) ---
# -------------------------------------------------------------------
# --- Parámetros de la Maqueta de Laboratorio ---
ANCHO_MODELO = 0.20  # Ancho fijo: 20 cm
CAUDAL_MODELO_ESTIMADO = 0.001  # Caudal bajo: 1 L/s
COEFICIENTE_MANNING_MODELO = 0.009  # Material: PVC o acrílico

# --- Parámetros del Entorno y de Diseño ---
PENDIENTE_CANAL = 0.00373
GRAVEDAD = 9.81

# --- PARÁMETRO DE EXPLORACIÓN ---
# ¡Este es el valor clave a modificar para cambiar el tamaño del prototipo!
# Un valor más bajo (ej. 10-15) dará un prototipo más pequeño.
ESCALA_LONGITUD_DESEADA = 15.0  # λ_L: Prototipo será 15 veces más grande


# -------------------------------------------------------------------

# --- 2. FUNCIONES DE CÁLCULO (sin cambios) ---

def manning_equation_for_y(y, Q, n, S, b):
    """Función para resolver con fsolve. Busca el 'y' que hace la ecuación cero."""
    if y <= 0:
        return float('inf')
    A = b * y
    P = b + 2 * y
    Rh = A / P
    return Q - (1 / n) * A * (Rh ** (2 / 3)) * (S ** 0.5)


def calcular_parametros_hidraulicos(b, y, Q, n, S, g, etiqueta):
    """Calcula y empaqueta todas las propiedades hidráulicas."""
    if y is None or b is None or y <= 0 or b <= 0: return {}
    area = b * y
    velocidad = Q / area if area > 0 else 0
    froude = velocidad / math.sqrt(g * y) if y > 0 else 0
    regimen = "Subcrítico" if froude < 1 else "Supercrítico" if froude > 1 else "Crítico"

    return {
        f"Ancho ({etiqueta})": b, f"Tirante ({etiqueta})": y, f"Área ({etiqueta})": area,
        f"Velocidad ({etiqueta})": velocidad, f"Caudal ({etiqueta})": Q,
        f"Número de Froude ({etiqueta})": froude, f"Régimen ({etiqueta})": regimen,
        f"Coef. Manning ({etiqueta})": n
    }


# --- 3. EJECUCIÓN DEL CÁLCULO INVERSO ---

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("== CÁLCULO HIDRÁULICO INVERSO: DE LA MAQUETA AL MUNDO REAL (VERSIÓN AJUSTABLE) ==")
    print("=" * 80)

    # --- PASO A: ANÁLISIS DEL MODELO DE LABORATORIO ---
    print("\n[ 1 ] Analizando el MODELO FÍSICO (Maqueta) con parámetros definidos...")

    initial_guess_y = 0.02  # Ajustamos el valor inicial a algo más cercano a lo esperado
    y_m_solved = fsolve(manning_equation_for_y, initial_guess_y,
                        args=(CAUDAL_MODELO_ESTIMADO, COEFICIENTE_MANNING_MODELO, PENDIENTE_CANAL, ANCHO_MODELO))[0]

    modelo = calcular_parametros_hidraulicos(ANCHO_MODELO, y_m_solved, CAUDAL_MODELO_ESTIMADO,
                                             COEFICIENTE_MANNING_MODELO, PENDIENTE_CANAL, GRAVEDAD, "m")

    print("\n    Parámetros resultantes en la maqueta:")
    for key, value in modelo.items():
        unit = ""
        if 'Ancho' in key or 'Tirante' in key: unit = f"  (o {value * 100:.2f} cm)"
        if 'Caudal' in key: unit = f"  (o {value * 1000:.4f} L/s)"
        print(f"    - {key:<25}: {value:.5f}{unit}" if isinstance(value, float) else f"    - {key:<25}: {value}")

    # --- PASO B: ESCALAMIENTO INVERSO HACIA EL PROTOTIPO ---
    print(f"\n[ 2 ] Escalando INVERSAMENTE hacia el PROTOTIPO con una escala λ_L = {ESCALA_LONGITUD_DESEADA:.1f}...")

    lambda_L = ESCALA_LONGITUD_DESEADA

    b_p = modelo["Ancho (m)"] * lambda_L
    y_p = modelo["Tirante (m)"] * lambda_L
    area_p = b_p * y_p

    lambda_V = math.sqrt(lambda_L)
    V_p = modelo["Velocidad (m)"] * lambda_V

    Q_p = V_p * area_p

    n_p = COEFICIENTE_MANNING_MODELO * (lambda_L ** (1 / 6))

    # --- PASO C: RESULTADOS Y VERIFICACIÓN DEL PROTOTIPO ---
    print("\n[ 3 ] Características del PROTOTIPO (Canal Real) que se estaría simulando:")
    prototipo = calcular_parametros_hidraulicos(b_p, y_p, Q_p, n_p, PENDIENTE_CANAL, GRAVEDAD, "p")
    for key, value in prototipo.items():
        print(f"    - {key:<25}: {value:.5f}" if isinstance(value, float) else f"    - {key:<25}: {value}")

    print(f"    - Rugosidad del prototipo (n_p): {n_p:.4f} <-- ¿Es un valor realista?")
    if 0.012 <= n_p <= 0.017:
        print("        (Sí, este valor es consistente con concreto u hormigón)")
    else:
        print("        (Este valor podría no corresponder a un material de canal común)")

    # --- PASO D: VALIDACIÓN FINAL ---
    print("\n[ 4 ] VALIDACIÓN DE SEMEJANZA DE FROUDE")
    fr_p = prototipo.get("Número de Froude (p)", -1)
    fr_m = modelo.get("Número de Froude (m)", -2)

    print(f"    - Número de Froude del Modelo (Fr_m):   {fr_m:.4f}")
    print(f"    - Número de Froude del Prototipo (Fr_p): {fr_p:.4f}")

    if math.isclose(fr_p, fr_m, rel_tol=1e-5):
        print("\n    ---> ¡VALIDACIÓN EXITOSA! El proceso inverso es consistente.")
    else:
        print("\n    ---> ¡ERROR DE VALIDACIÓN! Hay un error en la lógica inversa.")

    print("\n" + "=" * 80 + "\n")
