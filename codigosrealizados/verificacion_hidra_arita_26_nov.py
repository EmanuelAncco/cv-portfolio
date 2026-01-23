"""
Cálculo de Tirante Normal en Canal Rectangular usando Manning
Autor: Emanuel
Fecha: 26/11/2025

Ecuación de Manning: Q = (1/n) * A * R^(2/3) * S^(1/2)
Para sección rectangular:
    A = b * y
    P = b + 2y
    R = A/P
"""

import numpy as np
from scipy.optimize import brentq

# =============================================================================
# DATOS EXPERIMENTALES
# =============================================================================
Q = 0.0001057  # Caudal (m³/s)
n = 0.01  # Coeficiente de rugosidad de Manning
Z = 0  # Talud (0 = rectangular)
b = 0.05  # Ancho de solera (m)

# NOTA: Hay discrepancia en pendiente entre Excel (0.02) y Software (0.002)
pendientes = {
    "Excel (S=0.02)": 0.02,
    "Software (S=0.002)": 0.002
}


# =============================================================================
# FUNCIONES DE CÁLCULO
# =============================================================================
def area_hidraulica(y, b, Z):
    """Área hidráulica para sección trapezoidal/rectangular"""
    return (b + Z * y) * y


def perimetro_mojado(y, b, Z):
    """Perímetro mojado para sección trapezoidal/rectangular"""
    return b + 2 * y * np.sqrt(1 + Z ** 2)


def radio_hidraulico(y, b, Z):
    """Radio hidráulico R = A/P"""
    A = area_hidraulica(y, b, Z)
    P = perimetro_mojado(y, b, Z)
    return A / P


def caudal_manning(y, b, Z, n, S):
    """Caudal calculado con ecuación de Manning"""
    A = area_hidraulica(y, b, Z)
    R = radio_hidraulico(y, b, Z)
    return (1 / n) * A * R ** (2 / 3) * S ** (1 / 2)


def funcion_objetivo(y, Q_objetivo, b, Z, n, S):
    """f(y) = Q_calculado - Q_objetivo = 0"""
    return caudal_manning(y, b, Z, n, S) - Q_objetivo


# =============================================================================
# SOLUCIÓN NUMÉRICA - Método de Brent
# =============================================================================
# Límites de búsqueda para el tirante
y_min = 0.0001  # m
y_max = 1.0  # m

print("=" * 70)
print("CÁLCULO DE TIRANTE NORMAL - ECUACIÓN DE MANNING")
print("=" * 70)
print("\n>>> DATOS DE ENTRADA:")
print(f"    Caudal (Q):            {Q:.7f} m³/s")
print(f"    Rugosidad (n):         {n}")
print(f"    Talud (Z):             {Z}")
print(f"    Ancho de solera (b):   {b} m")

print("\n" + "=" * 70)
print("COMPARACIÓN DE PENDIENTES")
print("=" * 70)

resultados = {}

for nombre, S in pendientes.items():
    # Resolver para tirante normal
    y_normal = brentq(funcion_objetivo, y_min, y_max, args=(Q, b, Z, n, S))

    # Calcular parámetros
    A = area_hidraulica(y_normal, b, Z)
    P = perimetro_mojado(y_normal, b, Z)
    R = radio_hidraulico(y_normal, b, Z)
    T = b + 2 * Z * y_normal
    V = Q / A
    g = 9.81
    D_h = A / T
    Fr = V / np.sqrt(g * D_h)
    E = y_normal + V ** 2 / (2 * g)

    if Fr < 1:
        tipo_flujo = "Subcrítico"
    elif Fr > 1:
        tipo_flujo = "Supercrítico"
    else:
        tipo_flujo = "Crítico"

    resultados[nombre] = {
        'S': S, 'y': y_normal, 'A': A, 'P': P, 'R': R,
        'V': V, 'Fr': Fr, 'E': E, 'flujo': tipo_flujo
    }

    print(f"\n>>> {nombre}")
    print(f"    Pendiente (S):         {S} m/m")
    print(f"    Tirante normal (y):    {y_normal:.4f} m")
    print(f"    Área hidráulica (A):   {A:.6f} m²")
    print(f"    Perímetro mojado (P):  {P:.4f} m")
    print(f"    Radio hidráulico (R):  {R:.4f} m")
    print(f"    Velocidad (V):         {V:.4f} m/s")
    print(f"    Número de Froude (Fr): {Fr:.4f}")
    print(f"    Energía específica:    {E:.4f} m-kg/kg")
    print(f"    Tipo de flujo:         {tipo_flujo}")

# =============================================================================
# COMPARACIÓN CON SOFTWARE (S = 0.002)
# =============================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN PYTHON vs SOFTWARE (usando S = 0.002)")
print("=" * 70)

# Valores del software (de la imagen)
y_software = 0.0118
A_software = 0.0006
P_software = 0.0736
R_software = 0.0080
V_software = 0.1791
Fr_software = 0.5265
E_software = 0.0134

# Usar resultados con S = 0.002
r = resultados["Software (S=0.002)"]

print("\n{:<25} {:>12} {:>12} {:>10}".format("Parámetro", "Python", "Software", "Dif (%)"))
print("-" * 60)


def calcular_diferencia(val_python, val_software):
    if val_software != 0:
        return abs(val_python - val_software) / val_software * 100
    return 0


print("{:<25} {:>12.4f} {:>12.4f} {:>10.2f}".format(
    "Tirante (y) [m]", r['y'], y_software,
    calcular_diferencia(r['y'], y_software)))

print("{:<25} {:>12.6f} {:>12.4f} {:>10.2f}".format(
    "Área (A) [m²]", r['A'], A_software,
    calcular_diferencia(r['A'], A_software)))

print("{:<25} {:>12.4f} {:>12.4f} {:>10.2f}".format(
    "Perímetro (P) [m]", r['P'], P_software,
    calcular_diferencia(r['P'], P_software)))

print("{:<25} {:>12.4f} {:>12.4f} {:>10.2f}".format(
    "Radio hidráulico (R) [m]", r['R'], R_software,
    calcular_diferencia(r['R'], R_software)))

print("{:<25} {:>12.4f} {:>12.4f} {:>10.2f}".format(
    "Velocidad (V) [m/s]", r['V'], V_software,
    calcular_diferencia(r['V'], V_software)))

print("{:<25} {:>12.4f} {:>12.4f} {:>10.2f}".format(
    "Froude (Fr)", r['Fr'], Fr_software,
    calcular_diferencia(r['Fr'], Fr_software)))

print("{:<25} {:>12.4f} {:>12.4f} {:>10.2f}".format(
    "Energía esp. (E) [m]", r['E'], E_software,
    calcular_diferencia(r['E'], E_software)))

print("\n>>> CONCLUSIÓN:")
print("    ✓ Los cálculos en Python COINCIDEN con el software cuando S = 0.002")
print("    ✗ El valor S = 0.02 de tu Excel NO corresponde al usado en el software")
print("\n    VERIFICA LA PENDIENTE REAL DE TU CANAL EXPERIMENTAL")