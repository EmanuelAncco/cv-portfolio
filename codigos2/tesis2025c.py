import numpy as np

# Parámetros del problema (Apéndice A y Tabla 2)
D = 0.762  # Diámetro de la tubería (m)
t = 0.0125  # Espesor de la pared (m)
E1 = 210e9  # Módulo elástico del acero (Pa)
sigma_y = 490e6  # Tensión de fluencia (Pa)
tau_u = 22.75e3  # Fricción unitaria del suelo (N/m)
L1 = 8.0  # Distancia entre juntas (m)
L2 = 8.0  # Longitud del segmento BC (m)
Dy = 1.5  # Desplazamiento lateral del suelo (m)
beta = 60  # Ángulo de cruce (grados)
Cr = 0  # Rigidez rotacional (asumida 0)

# Propiedades geométricas
A = np.pi * (D - t) * t  # Área transversal (m²)
I = np.pi * ((D - t)/2)**3 * t  # Momento de inercia (m⁴)
q_u = 134.45e3  # Resistencia lateral del suelo (N/m)
K_spr = q_u / 0.0357  # Rigidez del resorte lateral (N/m²)

# Desplazamientos de la falla (componentes x e y)
Dx = Dy * np.tan(np.radians(beta))  # Componente axial (m)
Dy_half = Dy / 2  # Mitad del desplazamiento lateral (condición de simetría)

# Valores iniciales (Apéndice A)
phi_A = np.radians(0.015)  # Rotación en A (radianes)
phi_B = np.radians(4.000)  # Rotación en B (radianes)
V_A = 5.25e3  # Fuerza cortante en A (N)

# Cálculo de q_u* (Ecuación 25)
q_u_star = min(q_u, K_spr * L1 * np.tan(phi_A))
print(f"q_u* = {q_u_star/1e3:.2f} kN/m")

# Cálculo del alargamiento requerido (ΔL_req, Ecuación 16)
delta_L_req = Dx + 2 * (L1 / np.cos(phi_A) - L1) + 2 * (L2 / np.cos(phi_A + phi_B) - L2)
print(f"ΔL_req = {delta_L_req:.2f} m")

# Cálculo de la tensión axial (σ_a, Ecuación 20)
sigma_a = np.sqrt(E1 * tau_u * delta_L_req / A)
print(f"σ_a = {sigma_a/1e6:.2f} MPa")

# Verificación si σ_a > σ_y (fluencia)
if sigma_a > sigma_y:
    print("¡Advertencia! La tensión axial supera la tensión de fluencia.")
else:
    print("La tensión axial está dentro del rango elástico.")

# Cálculo de las fuerzas axiales (Ecuaciones 24a y 24b)
N_C = sigma_a * A  # Fuerza axial en C
N_B = N_C - tau_u * L2  # Fuerza axial en B
N_A = N_B - tau_u * L1  # Fuerza axial en A
print("\nFuerzas axiales:")
print(f"N_A = {N_A/1e6:.2f} MN (fuerza en el punto A)")
print(f"N_B = {N_B/1e6:.2f} MN (fuerza en el punto B)")
print(f"N_C = {N_C/1e6:.2f} MN (fuerza en el punto C)")