import numpy as np
from scipy.optimize import fsolve

# Parámetros del problema (Apéndice A y Tabla 2)
D = 0.762  # Diámetro de la tubería (m)
t = 0.0125  # Espesor de la pared (m)
E1 = 210e9  # Módulo elástico del acero (Pa)
E2 = 1.088e9  # Módulo plástico del acero (Pa)
sigma_y = 490e6  # Tensión de fluencia (Pa)
tau_u = 22.75e3  # Fricción unitaria del suelo (N/m)
L1 = 8.0  # Distancia entre juntas (m)
L2 = 8.0  # Longitud del segmento BC (m)
Dy = 1.5  # Desplazamiento lateral del suelo (m)
beta = 60  # Ángulo de cruce (grados)
Cr = 0  # Rigidez rotacional (asumida 0)

# Propiedades geométricas
A = np.pi * (D - t) * t  # Área transversal (m²)
I = np.pi * ((D - t) / 2) ** 3 * t  # Momento de inercia (m⁴)
q_u = 134.45e3  # Resistencia lateral del suelo (N/m)
K_spr = q_u / 0.0357  # Rigidez del resorte lateral (N/m²)

# Desplazamientos de la falla (componentes x e y)
Dx = Dy * np.tan(np.radians(beta))  # Componente axial (m)
Dy_half = Dy / 2  # Mitad del desplazamiento lateral (condición de simetría)


# Función para resolver el sistema no lineal
def equations(vars):
    phi_A_deg, phi_B_deg, V_A = vars
    phi_A = np.radians(phi_A_deg)
    phi_B = np.radians(phi_B_deg)

    # Cálculo de fuerzas axiales (Ecuaciones 24a, 24b)
    N_B = sigma_y * A - tau_u * L1  # Fuerza axial en B
    N_A = N_B - tau_u * L1  # Fuerza axial en A

    # Parámetros alpha (Ecuación α² = N/(E*I))
    alpha_AB = np.sqrt(N_A / (E1 * I))
    alpha_BC = np.sqrt(N_B / (E1 * I))

    # Deformaciones (Ecuaciones 10 y 11 adaptadas)
    # Segmento AB
    w_AB = (V_A / (E1 * I * alpha_AB ** 3)) * (np.sinh(alpha_AB * L1) - alpha_AB * L1)
    delta_AB = w_AB + L1 * np.sin(phi_A)

    # Segmento BC
    w_BC = (V_A / (E1 * I * alpha_BC ** 3)) * (np.sinh(alpha_BC * L2) - alpha_BC * L2)
    delta_BC = delta_AB + w_BC + L2 * np.sin(phi_A + phi_B)

    # Momentos (Ecuaciones 12 y 13)
    M_B = Cr * phi_B  # ≈ 0 por Cr=0
    M_C = 0  # Condición de simetría

    # Errores a minimizar
    error1 = delta_BC - Dy_half  # Condición de desplazamiento
    error2 = M_B  # Momento en B (≈0)
    error3 = M_C  # Momento en C (≈0)

    return [error1, error2, error3]


# Solución numérica con valores iniciales del Apéndice A
initial_guess = [0.219, 3.652, 182.09e3]  # φ_A(°), φ_B(°), V_A(N)
solution = fsolve(equations, initial_guess, xtol=1e-6)

# Resultados
phi_A_sol, phi_B_sol, V_A_sol = solution
print("Resultados:")
print(f"φ_A = {phi_A_sol:.3f}°")
print(f"φ_B = {phi_B_sol:.3f}°")
print(f"V_A = {V_A_sol / 1e3:.2f} kN")

# Cálculo de fuerzas normales (axiales) en los segmentos AB y BC
N_B = sigma_y * A - tau_u * L1  # Fuerza axial en B
N_A = N_B - tau_u * L1  # Fuerza axial en A
print("\nFuerzas normales (axiales):")
print(f"N_A = {N_A / 1e6:.2f} MN (fuerza en el punto A)")
print(f"N_B = {N_B / 1e6:.2f} MN (fuerza en el punto B)")

# Cálculo de tensiones máximas
N_middle = (N_A + N_B) / 2  # Fuerza axial promedio
M_max = V_A_sol * L1 / 2  # Momento máximo en el centro del segmento

# Tensión por flexión (σ = M*y/I + N/A)
y = D / 2  # Distancia a la fibra extrema
sigma_max = (M_max * y) / I + N_middle / A
print(f"\nTensión máxima calculada: {sigma_max / 1e6:.2f} MPa")