import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Arrow, FancyArrowPatch

# --- Parámetros de Entrada ---
D_e = 0.61  # Diámetro exterior de la tubería (m)
t = 0.0095  # Espesor de pared de la tubería (m)
P_int = 7e6  # Presión interna (Pa)
H_cubierta = 1.5  # Cubierta de suelo (m)
gamma_suelo = 19e3  # Densidad del suelo (N/m^3)
E_acero = 207e9  # Módulo de Young del acero (Pa)
nu_acero = 0.3  # Coeficiente de Poisson del acero
alpha_T = 1.2e-5  # Coeficiente de expansión térmica (1/°C)
delta_T = -15  # Cambio de temperatura (°C)
S_y = 448e6 # Límite elástico del acero API 5L X65 (Pa)


# Longitud del tramo para diagramas longitudinales
L_tramo = 100  # m
num_secciones = 10 # Divisiones para el gráfico (aunque los valores son constantes)

# --- Parámetros Derivados ---
D_m = D_e - t  # Diámetro medio (m)
R_m = D_m / 2  # Radio medio (m)
R_e = D_e / 2 # Radio exterior (m)
R_i = R_e - t # Radio interior (m)


# --- Funciones de Cálculo ---

def calc_esfuerzo_circunferencial_presion(P, D_ext, espesor):
    """Calcula el esfuerzo circunferencial por presión interna (Fórmula de Barlow)."""
    sigma_h = (P * D_ext) / (2 * espesor)
    return sigma_h

def calc_carga_vertical_suelo(gamma, H_c, D_ext):
    """Calcula la carga vertical del suelo por unidad de longitud de tubería."""
    W_s = gamma * H_c * D_ext
    return W_s

def calc_momento_flector_corona(W_total_vertical, R_medio, coeff_M=0.125):
    """Calcula el momento flector en la corona debido a la carga vertical."""
    M_corona = coeff_M * W_total_vertical * R_medio
    return M_corona

def calc_modulo_seccion_pared(espesor):
    """Calcula el módulo de sección de la pared por unidad de longitud."""
    Z = (1 * espesor**2) / 6
    return Z

def calc_esfuerzo_flexion(Momento, Z_seccion):
    """Calcula el esfuerzo de flexión máximo (en las fibras extremas)."""
    sigma_b = Momento / Z_seccion
    return sigma_b

def calc_esfuerzo_axial_presion_poisson(sigma_h_val, nu_val):
    """Calcula el esfuerzo axial debido al efecto Poisson de la presión interna."""
    sigma_L_p = nu_val * sigma_h_val
    return sigma_L_p

def calc_esfuerzo_axial_termico(E_val, alpha_val, delta_T_val):
    """Calcula el esfuerzo axial debido a cambios de temperatura (restringido)."""
    sigma_L_T = E_val * alpha_val * delta_T_val
    return sigma_L_T

def calc_esfuerzo_axial_total(sigma_L_p_val, sigma_L_T_val):
    """Calcula el esfuerzo axial total."""
    return sigma_L_p_val + sigma_L_T_val

def calc_esfuerzo_von_mises(sigma_circ, sigma_axial):
    """Calcula el esfuerzo de Von Mises para un estado biaxial (sigma_radial=0)."""
    sigma_vm = np.sqrt(sigma_circ**2 - sigma_circ * sigma_axial + sigma_axial**2)
    return sigma_vm

def calc_deformacion_axial_total(sigma_L_val, sigma_h_presion_val, nu_val, E_val_mod, alpha_val, delta_T_temp_val):
    """Calcula la deformación axial total."""
    epsilon_L = (sigma_L_val - nu_val * sigma_h_presion_val) / E_val_mod + alpha_val * delta_T_temp_val
    return epsilon_L

def calc_deformacion_circunferencial_total(sigma_theta_val, sigma_L_axial_val, nu_val, E_val_mod, alpha_val, delta_T_temp_val):
    """Calcula la deformación circunferencial total."""
    epsilon_theta = (sigma_theta_val - nu_val * sigma_L_axial_val) / E_val_mod + alpha_val * delta_T_temp_val
    return epsilon_theta


def calc_esfuerzo_circunferencial_distribuido(sigma_h_presion_val, sigma_b_flexion_max_corona_val, theta_rad_array):
    """Calcula la distribución del esfuerzo circunferencial total en la pared interna."""
    sigma_theta_dist = sigma_h_presion_val + sigma_b_flexion_max_corona_val * np.cos(2 * theta_rad_array)
    return sigma_theta_dist

def calc_momento_flector_circunferencial(M_corona_max_val, theta_rad_array):
    """Calcula la distribución del momento flector alrededor de la circunferencia."""
    M_theta = M_corona_max_val * np.cos(2 * theta_rad_array)
    return M_theta

def calc_fuerza_cortante_circunferencial(M_corona_max_val, R_medio_val, theta_rad_array):
    """Calcula la distribución de la fuerza cortante alrededor de la circunferencia."""
    V_theta = (-2 * M_corona_max_val / R_medio_val) * np.sin(2 * theta_rad_array)
    return V_theta


# --- Ejecución de Cálculos ---
print(f"--- Parámetros Geométricos ---")
print(f"Radio exterior (R_e): {R_e*1000:.1f} mm")
print(f"Radio interior (R_i): {R_i*1000:.1f} mm") # Corregido R_i = R_e - t
print(f"Radio medio (R_m): {R_m*1000:.1f} mm")
print(f"Espesor (t): {t*1000:.1f} mm")


# 1. Esfuerzo circunferencial por presión interna
sigma_h = calc_esfuerzo_circunferencial_presion(P_int, D_e, t)
print(f"\n--- Resultados Numéricos de Esfuerzos y Deformaciones ---")
print(f"Esfuerzo circunferencial por presión interna (σ_h): {sigma_h/1e6:.2f} MPa")

# 2. Esfuerzo de flexión en la corona por carga vertical
W_suelo = calc_carga_vertical_suelo(gamma_suelo, H_cubierta, D_e)
print(f"Carga vertical del suelo (W_suelo): {W_suelo/1e3:.2f} kN/m")

M_corona = calc_momento_flector_corona(W_suelo, R_m)
print(f"Momento flector MÁXIMO en la corona (M_corona): {M_corona:.2f} Nm/m")

Z_pared = calc_modulo_seccion_pared(t)
print(f"Módulo de sección de la pared (Z): {Z_pared*1e6:.2f} cm³/m")

sigma_b_corona_max = calc_esfuerzo_flexion(M_corona, Z_pared) # Esfuerzo en fibra extrema
print(f"Esfuerzo de flexión MÁXIMO en corona (σ_b,coro,max): {sigma_b_corona_max/1e6:.2f} MPa")

# 3. Esfuerzos totales circunferenciales en corona (fibra interna y externa)
sigma_theta_total_corona_interna = sigma_h + sigma_b_corona_max
print(f"Esfuerzo circunferencial total en corona (σ_θ,coro, interna): {sigma_theta_total_corona_interna/1e6:.2f} MPa")
sigma_theta_total_corona_externa = sigma_h - sigma_b_corona_max
print(f"Esfuerzo circunferencial total en corona (σ_θ,coro, externa): {sigma_theta_total_corona_externa/1e6:.2f} MPa")


# 4. Esfuerzo axial y Von Mises
sigma_L_p = calc_esfuerzo_axial_presion_poisson(sigma_h, nu_acero)
print(f"Esfuerzo axial por presión (σ_L,p): {sigma_L_p/1e6:.2f} MPa")

sigma_L_T = calc_esfuerzo_axial_termico(E_acero, alpha_T, delta_T)
print(f"Esfuerzo axial por temperatura (σ_L,T): {sigma_L_T/1e6:.2f} MPa")

sigma_L_total = calc_esfuerzo_axial_total(sigma_L_p, sigma_L_T)
print(f"Esfuerzo axial total (σ_L): {sigma_L_total/1e6:.2f} MPa")

sigma_VM_corona_interna = calc_esfuerzo_von_mises(sigma_theta_total_corona_interna, sigma_L_total)
print(f"Esfuerzo de Von Mises en corona (σ_VM,coro, interna): {sigma_VM_corona_interna/1e6:.2f} MPa")
sigma_VM_corona_externa = calc_esfuerzo_von_mises(sigma_theta_total_corona_externa, sigma_L_total)
print(f"Esfuerzo de Von Mises en corona (σ_VM,coro, externa): {sigma_VM_corona_externa/1e6:.2f} MPa")


# 5. Deformaciones
epsilon_L_total = calc_deformacion_axial_total(sigma_L_total, sigma_h, nu_acero, E_acero, alpha_T, delta_T)
print(f"Deformación axial total (ε_L): {epsilon_L_total*1e6:.2f} µε (microstrain)")

epsilon_theta_total_corona_interna = calc_deformacion_circunferencial_total(
    sigma_theta_total_corona_interna, sigma_L_total, nu_acero, E_acero, alpha_T, delta_T
)
print(f"Deformación circunferencial total en corona (ε_θ,coro, interna): {epsilon_theta_total_corona_interna*1e6:.2f} µε")

epsilon_theta_total_corona_externa = calc_deformacion_circunferencial_total(
    sigma_theta_total_corona_externa, sigma_L_total, nu_acero, E_acero, alpha_T, delta_T
)
print(f"Deformación circunferencial total en corona (ε_θ,coro, externa): {epsilon_theta_total_corona_externa*1e6:.2f} µε")


# --- Generación de Gráficos ---
posicion_longitudinal = np.linspace(0, L_tramo, num_secciones)
theta_deg_dist = np.linspace(0, 360, 360) # Renombrado para evitar conflicto con variable theta_deg en Diagrama G
theta_rad_dist = np.deg2rad(theta_deg_dist)

# A. Diagrama longitudinal de σ_VM vs posición (0–100 m)
sigma_VM_array = np.full_like(posicion_longitudinal, sigma_VM_corona_interna / 1e6)
plt.figure("Gráfico A", figsize=(10, 6))
# ... (código del Gráfico A sin cambios) ...
plt.plot(posicion_longitudinal, sigma_VM_array, marker='o')
plt.title('Diagrama Longitudinal de Esfuerzo de Von Mises (σ_VM) en Corona (Fibra Interna)')
plt.xlabel('Posición a lo largo de la tubería (m)')
plt.ylabel('Esfuerzo de Von Mises (σ_VM) (MPa)')
plt.grid(True)
plt.ylim(bottom=0, top=max(S_y/1e6, sigma_VM_corona_interna/1e6 * 1.2))
plt.axhline(y=S_y/1e6, color='r', linestyle='--', label=f'Límite Elástico Sy = {S_y/1e6:.0f} MPa')
plt.legend()
plt.show()


# B. Diagrama longitudinal de deformación axial ε
epsilon_L_array = np.full_like(posicion_longitudinal, epsilon_L_total * 1e6)
plt.figure("Gráfico B", figsize=(10, 6))
# ... (código del Gráfico B sin cambios) ...
plt.plot(posicion_longitudinal, epsilon_L_array, marker='o')
plt.title('Diagrama Longitudinal de Deformación Axial Total (ε_L)')
plt.xlabel('Posición a lo largo de la tubería (m)')
plt.ylabel('Deformación Axial Total (ε_L) (µε)')
plt.grid(True)
min_strain_L = min(0, epsilon_L_total * 1e6 * 1.2 if epsilon_L_total < 0 else epsilon_L_total * 1e6 * 0.8)
max_strain_L = max(0, epsilon_L_total * 1e6 * 1.2 if epsilon_L_total > 0 else epsilon_L_total * 1e6 * 0.8)
if abs(min_strain_L - max_strain_L) < 1e-9 :
    min_strain_L = (epsilon_L_total * 1e6) - 10 if (epsilon_L_total * 1e6) != 0 else -10
    max_strain_L = (epsilon_L_total * 1e6) + 10 if (epsilon_L_total * 1e6) != 0 else 10
plt.ylim(min_strain_L, max_strain_L)
plt.show()

# C. Gráfico polar σ_θ(θ) (0–360°)
sigma_theta_distribuido_interno = calc_esfuerzo_circunferencial_distribuido(sigma_h, sigma_b_corona_max, theta_rad_dist) / 1e6
plt.figure("Gráfico C", figsize=(8, 8))
# ... (código del Gráfico C sin cambios) ...
ax_polar = plt.subplot(111, projection='polar')
ax_polar.plot(theta_rad_dist, sigma_theta_distribuido_interno)
ax_polar.set_title('Distribución Circunferencial del Esfuerzo σ_θ (MPa) en Fibra Interna', va='bottom', pad=20)
ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.plot(0, sigma_theta_distribuido_interno[0], 'ro', markersize=8, label=f'Corona (θ=0°): {sigma_theta_distribuido_interno[0]:.2f} MPa')
ax_polar.legend(loc="lower left", bbox_to_anchor=(1.05, 0))
plt.show()

# D. Diagrama esquemático de esfuerzos en sección transversal
fig_d, ax_d = plt.subplots(figsize=(8, 8))
# ... (código del Diagrama D sin cambios) ...
circ_externo = plt.Circle((0, 0), R_e, color='lightgray', fill=True, ec='black', label='Tubería')
circ_interno = plt.Circle((0, 0), R_i, color='white', fill=True, ec='black')
ax_d.add_artist(circ_externo)
ax_d.add_artist(circ_interno)
arrow_scale = R_e * 0.3
num_arrows_sigma_h = 8
for i in range(num_arrows_sigma_h):
    angle = 2 * np.pi * i / num_arrows_sigma_h
    start_x, start_y = R_m * np.cos(angle), R_m * np.sin(angle)
    end_x, end_y = (R_m + arrow_scale*0.8) * np.cos(angle), (R_m + arrow_scale*0.8) * np.sin(angle)
    ax_d.add_patch(FancyArrowPatch((start_x, start_y), (end_x, end_y), arrowstyle='->', color='blue', mutation_scale=15, lw=1))
ax_d.text(R_e * 1.1, R_e * 0.8, 'σ_h (presión)', color='blue', ha='left', va='center')
ax_d.add_patch(FancyArrowPatch((0, R_i - arrow_scale*0.6), (0, R_i), arrowstyle='->', color='red', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((0, -R_i + arrow_scale*0.6), (0, -R_i), arrowstyle='->', color='red', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((0, R_e), (0, R_e - arrow_scale*0.6), arrowstyle='->', color='green', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((0, -R_e), (0, -R_e + arrow_scale*0.6), arrowstyle='->', color='green', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((R_i, 0), (R_i - arrow_scale*0.6, 0), arrowstyle='->', color='green', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((-R_i, 0), (-R_i + arrow_scale*0.6, 0), arrowstyle='->', color='green', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((R_e - arrow_scale*0.6, 0), (R_e, 0), arrowstyle='->', color='red', mutation_scale=15, lw=1))
ax_d.add_patch(FancyArrowPatch((-R_e + arrow_scale*0.6, 0), (-R_e, 0), arrowstyle='->', color='red', mutation_scale=15, lw=1))
ax_d.text(R_e * 1.1, R_e * 0.6, 'σ_b (flexión, tracción)', color='red', ha='left', va='center')
ax_d.text(R_e * 1.1, R_e * 0.4, 'σ_b (flexión, compresión)', color='green', ha='left', va='center')
ax_d.add_patch(FancyArrowPatch((0, R_e + arrow_scale*1.2), (0, R_e + arrow_scale*0.2), arrowstyle='-|>', color='black', mutation_scale=20, lw=1.5, shrinkB=5))
ax_d.text(0, R_e + arrow_scale*1.4, 'Carga Suelo (W_suelo)', color='black', ha='center', va='bottom')
ax_d.text(0, 0, 'σ_L\n(axial)', color='purple', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, ec='purple', boxstyle='circle'))
ax_d.set_xlim(-R_e*1.8, R_e*1.8)
ax_d.set_ylim(-R_e*1.8, R_e*1.8)
ax_d.axis('off')
ax_d.set_title('Diagrama Esquemático de Esfuerzos en Sección Transversal')
plt.tight_layout()
plt.show()


# E. Diagrama de Momento Flector Circunferencial M(θ)
M_theta_dist = calc_momento_flector_circunferencial(M_corona, theta_rad_dist)
plt.figure("Gráfico E", figsize=(10, 6))
# ... (código del Gráfico E sin cambios) ...
plt.plot(theta_deg_dist, M_theta_dist)
plt.title('Distribución de Momento Flector Circunferencial $M(\\theta)$')
plt.xlabel('Ángulo desde la corona (θ) (grados)')
plt.ylabel('Momento Flector (Nm/m)')
plt.xticks(np.arange(0, 361, 45))
plt.grid(True)
plt.axhline(0, color='black', lw=0.5)
plt.show()

# F. Diagrama de Fuerza Cortante Circunferencial V(θ)
V_theta_dist = calc_fuerza_cortante_circunferencial(M_corona, R_m, theta_rad_dist)
plt.figure("Gráfico F", figsize=(10, 6))
# ... (código del Gráfico F sin cambios) ...
plt.plot(theta_deg_dist, V_theta_dist)
plt.title('Distribución de Fuerza Cortante Circunferencial $V(\\theta)$')
plt.xlabel('Ángulo desde la corona (θ) (grados)')
plt.ylabel('Fuerza Cortante (N/m)')
plt.xticks(np.arange(0, 361, 45))
plt.grid(True)
plt.axhline(0, color='black', lw=0.5)
plt.show()

# G. Diagrama Esfuerzo-Deformación del Material y Punto de Operación en Corona
epsilon_yield = S_y / E_acero
plt.figure("Gráfico G", figsize=(10, 7))
# ... (código del Gráfico G sin cambios) ...
plt.plot([0, epsilon_yield * 1.5 * 1e6], [0, S_y * 1.5 / 1e6 ], 'k--', label=f'Comportamiento elástico (E={E_acero/1e9:.0f} GPa)')
plt.plot([0, epsilon_yield*1e6], [0, S_y/1e6], color='gray', linestyle='-', linewidth=3, label=f'Región elástica hasta Sy')
plt.axhline(y=S_y/1e6, color='r', linestyle=':', label=f'Límite Elástico $S_y = {S_y/1e6:.0f}$ MPa')
plt.axvline(x=epsilon_yield*1e6, color='r', linestyle=':', label=f'Deformación de Fluencia $\\epsilon_y = {epsilon_yield*1e6:.0f}$ µε')
plt.plot(epsilon_L_total*1e6, sigma_L_total/1e6, 'bo', markersize=8, label=f'Axial (σ_L, ε_L): ({sigma_L_total/1e6:.1f} MPa, {epsilon_L_total*1e6:.1f} µε)')
plt.plot(epsilon_theta_total_corona_interna*1e6, sigma_theta_total_corona_interna/1e6, 'go', markersize=8, label=f'Circunf. Corona (σ_θ, ε_θ): ({sigma_theta_total_corona_interna/1e6:.1f} MPa, {epsilon_theta_total_corona_interna*1e6:.1f} µε)')
plt.axhline(y=sigma_VM_corona_interna/1e6, color='purple', linestyle='-.', label=f'σ_VM corona = {sigma_VM_corona_interna/1e6:.1f} MPa')
plt.title('Estado de Esfuerzo-Deformación en Corona (Fibra Interna) vs. Material')
plt.xlabel('Deformación Total (µε)')
plt.ylabel('Esfuerzo (MPa)')
max_strain_plot = max(abs(epsilon_L_total*1e6), abs(epsilon_theta_total_corona_interna*1e6), epsilon_yield*1e6) * 1.2
max_stress_plot = max(sigma_L_total/1e6, sigma_theta_total_corona_interna/1e6, S_y/1e6, sigma_VM_corona_interna/1e6) * 1.2
min_strain_plot = min(epsilon_L_total*1e6, -max_strain_plot*0.1)
plt.xlim(min(min_strain_plot, -100) , max_strain_plot)
plt.ylim(0, max_stress_plot)
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()


# H. Distribución de Esfuerzo y Deformación Circunferencial a Través del Espesor en la Corona
fig_h, ax_h = plt.subplots(1, 2, figsize=(12, 7))
# Coordenada y a través del espesor (0 = fibra interna, t = fibra externa)
y_coords = np.array([0, t]) # m
y_coords_mm = y_coords * 1000 # mm

# Perfil de Esfuerzo Circunferencial (MPa)
sigma_theta_perfil = np.array([sigma_theta_total_corona_interna, sigma_theta_total_corona_externa]) / 1e6
ax_h[0].plot(sigma_theta_perfil, y_coords_mm)
ax_h[0].set_xlabel('Esfuerzo Circunferencial $\\sigma_\\theta$ (MPa)')
ax_h[0].set_ylabel('Posición a través del espesor (mm)\n(0=Interna, {:.1f}=Externa)'.format(t*1000))
ax_h[0].set_title('Distribución de $\\sigma_\\theta$ en Corona')
ax_h[0].axhline(0, color='gray', linestyle='--') # Fibra interna
ax_h[0].axhline(t*1000, color='gray', linestyle='--') # Fibra externa
ax_h[0].axvline(0, color='black', lw=0.5) # Eje de esfuerzo cero
ax_h[0].grid(True, linestyle=':', alpha=0.7)
# Anotar valores
ax_h[0].text(sigma_theta_total_corona_interna/1e6, 0, f' {sigma_theta_total_corona_interna/1e6:.1f} MPa', va='bottom', ha='left' if sigma_theta_total_corona_interna > 0 else 'right')
ax_h[0].text(sigma_theta_total_corona_externa/1e6, t*1000, f' {sigma_theta_total_corona_externa/1e6:.1f} MPa', va='top', ha='left' if sigma_theta_total_corona_externa > 0 else 'right')


# Perfil de Deformación Circunferencial (µε)
epsilon_theta_perfil = np.array([epsilon_theta_total_corona_interna, epsilon_theta_total_corona_externa]) * 1e6
ax_h[1].plot(epsilon_theta_perfil, y_coords_mm)
ax_h[1].set_xlabel('Deformación Circunferencial $\\epsilon_\\theta$ (µε)')
ax_h[1].set_ylabel('Posición a través del espesor (mm)')
ax_h[1].set_title('Distribución de $\\epsilon_\\theta$ en Corona')
ax_h[1].axhline(0, color='gray', linestyle='--')
ax_h[1].axhline(t*1000, color='gray', linestyle='--')
ax_h[1].axvline(0, color='black', lw=0.5)
ax_h[1].grid(True, linestyle=':', alpha=0.7)
# Anotar valores
ax_h[1].text(epsilon_theta_total_corona_interna*1e6, 0, f' {epsilon_theta_total_corona_interna*1e6:.1f} µε', va='bottom', ha='left' if epsilon_theta_total_corona_interna > 0 else 'right')
ax_h[1].text(epsilon_theta_total_corona_externa*1e6, t*1000, f' {epsilon_theta_total_corona_externa*1e6:.1f} µε', va='top', ha='left' if epsilon_theta_total_corona_externa > 0 else 'right')


fig_h.suptitle('Distribución de Esfuerzo y Deformación Circunferencial a Través del Espesor de la Pared en la Corona', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para el supertítulo
plt.show()

