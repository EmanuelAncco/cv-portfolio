import matplotlib.pyplot as plt
import numpy as np

# --- Datos extraídos de las tablas del Capítulo 6 ---

# Ángulos en grados
angulos_grados = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])
angulos_rad = np.deg2rad(angulos_grados)

# Límite elástico del material (SMYS)
smys = 448  # MPa

# --- Datos SURCO ---
vm_ext_surco = np.array([282.02, 281.91, 281.75, 280.93, 280.56, 280.96, 279.38, 275.92, 275.92, 275.90, 272.93, 272.93, 272.93])
vm_int_surco = np.array([263.74, 263.84, 264.06, 264.88, 265.23, 264.85, 266.31, 268.32, 268.32, 268.33, 270.36, 270.36, 270.35])
h_flex_ext_surco = np.array([132.63, 132.32, 131.79, 129.73, 128.98, 129.81, 126.24, 120.73, 120.73, 120.71, 115.62, 115.62, 115.64])
h_flex_int_surco = np.array([114.35, 114.59, 115.01, 116.19, 116.84, 115.97, 119.24, 124.75, 124.75, 124.77, 129.86, 129.86, 129.84])
h_total_ext_surco = np.array([260.86, 260.55, 260.02, 257.96, 257.21, 258.04, 254.47, 248.96, 248.96, 248.94, 243.85, 243.85, 243.87])
l_total_ext_surco = np.array([299.75, 299.66, 299.50, 298.88, 298.65, 298.90, 297.83, 296.18, 296.18, 296.17, 294.65, 294.65, 294.65])

# --- Datos VILLA EL SALVADOR ---
vm_ext_ves = np.array([449.03, 448.94, 448.83, 448.52, 448.24, 447.81, 450.80, 447.81, 448.24, 450.01, 448.83, 448.94, 450.56])
vm_int_ves = np.array([449.11, 449.19, 449.29, 449.50, 449.68, 449.95, 449.82, 449.95, 449.68, 450.11, 449.29, 449.19, 450.23])
h_flex_ext_ves = np.array([41.66, 41.33, 40.91, 39.76, 38.74, 37.14, 37.49, 37.14, 38.74, 35.28, 40.91, 41.33, 33.38])
h_flex_int_ves = np.array([32.76, 33.19, 33.61, 34.76, 35.78, 37.38, 34.19, 37.38, 35.78, 39.24, 33.61, 33.19, 31.54])
h_total_ext_ves = np.array([61.66, 61.33, 60.91, 59.76, 58.74, 57.14, 57.49, 57.14, 58.74, 55.28, 60.91, 61.33, 53.38])
l_total_ext_ves = np.array([478.04, 477.94, 477.81, 477.47, 477.16, 476.68, 476.79, 476.68, 477.16, 476.12, 477.81, 477.94, 475.55])

# --- Datos SAN JUAN DE LURIGANCHO ---
vm_ext_sjl = np.array([162.23, 162.15, 162.04, 161.76, 161.50, 161.07, 161.27, 161.07, 161.50, 160.81, 162.04, 162.15, 160.45])
vm_int_sjl = np.array([162.40, 162.48, 162.58, 162.79, 162.97, 163.24, 163.11, 163.24, 162.97, 163.40, 162.58, 162.48, 163.52])
h_flex_ext_sjl = h_flex_ext_ves
h_flex_int_sjl = h_flex_int_ves
h_total_ext_sjl = h_total_ext_ves
l_total_ext_sjl = np.array([188.24, 188.14, 188.01, 187.67, 187.36, 186.88, 186.99, 186.88, 187.36, 186.32, 188.01, 188.14, 185.75])

# Datos para los gráficos de barras
distritos = ['Surco', 'Villa El Salvador', 'S.J. Lurigancho']
esfuerzos_maximos = [282.02, 450.80, 163.24]
factores_seguridad = [1.588, 0.994, 2.744]
colores_barras = ['#4C72B0', '#C44E52', '#55A868']

# --- Creación de los Gráficos ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- Gráficos Polares de Esfuerzo de Von Mises ---
def plot_polar(ax, angulos, datos, color, titulo, label):
    ax.plot(angulos, datos, marker='o', color=color, label=label)
    theta_smys = np.linspace(0, 2 * np.pi, 100)
    r_smys = np.full_like(theta_smys, smys)
    ax.plot(theta_smys, r_smys, 'r--', label=f'Límite Elástico ({smys} MPa)')
    ax.set_title(titulo, fontsize=14, pad=20)
    ax.set_rlabel_position(22.5)
    ax.set_rmax(smys * 1.1)
    ax.legend(loc="lower left")

fig1, ax1 = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
fig1.canvas.manager.set_window_title('Gráfico 1: Esfuerzo Polar en Surco')
plot_polar(ax1, angulos_rad, vm_ext_surco, 'b', 'Distribución de Esfuerzo VM en Surco', 'Esfuerzo VM')
plt.show()

fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
fig2.canvas.manager.set_window_title('Gráfico 2: Esfuerzo Polar en V.E.S.')
plot_polar(ax2, angulos_rad, vm_ext_ves, 'g', 'Distribución de Esfuerzo VM en V.E.S.', 'Esfuerzo VM')
plt.show()

fig3, ax3 = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
fig3.canvas.manager.set_window_title('Gráfico 3: Esfuerzo Polar en S.J.L.')
plot_polar(ax3, angulos_rad, vm_ext_sjl, 'purple', 'Distribución de Esfuerzo VM en S.J.L.', 'Esfuerzo VM')
plt.show()

# --- Gráficos de Líneas de Esfuerzos en Fibras ---
def plot_fibras(titulo, angulos, ext_data, int_data):
    plt.figure(figsize=(10, 6))
    plt.get_current_fig_manager().set_window_title(titulo)
    plt.plot(angulos, ext_data, 'o-', color='darkorange', label='Esfuerzo en Fibra Externa ($\sigma_{ext}$)')
    plt.plot(angulos, int_data, 's-', color='darkcyan', label='Esfuerzo en Fibra Interna ($\sigma_{int}$)')
    plt.axhline(y=0, color='gray', linestyle=':')
    plt.title(titulo, fontsize=14)
    plt.xlabel('Ángulo (θ) desde la clave [Grados]', fontsize=12)
    plt.ylabel('Esfuerzo (MPa)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_fibras('Gráfico 4: Esfuerzos en Fibras (Surco)', angulos_grados, h_flex_ext_surco, h_flex_int_surco)
plot_fibras('Gráfico 5: Esfuerzos en Fibras (V.E.S.)', angulos_grados, h_flex_ext_ves, h_flex_int_ves)
plot_fibras('Gráfico 6: Esfuerzos en Fibras (S.J.L.)', angulos_grados, h_flex_ext_sjl, h_flex_int_sjl)

# --- Gráficos de Líneas de Descomposición de Esfuerzos ---
def plot_descomposicion(titulo, angulos, h_total, l_total, vm_total):
    plt.figure(figsize=(10, 6))
    plt.get_current_fig_manager().set_window_title(titulo)
    plt.plot(angulos, h_total, 'o-', color='green', label='Esfuerzo Circunferencial Total ($\sigma_{h,Total}$)')
    plt.plot(angulos, l_total, 's-', color='blue', label='Esfuerzo Longitudinal Total ($\sigma_{L,Total}$)')
    plt.plot(angulos, vm_total, '^-', color='red', label='Esfuerzo de Von Mises ($\sigma_{VM}$)')
    plt.axhline(y=smys, color='black', linestyle='--', label=f'Límite Elástico ({smys} MPa)')
    plt.title(titulo, fontsize=14)
    plt.xlabel('Ángulo (θ) desde la clave [Grados]', fontsize=12)
    plt.ylabel('Esfuerzo (MPa)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_descomposicion('Gráfico 7: Descomposición de Esfuerzos (Surco)', angulos_grados, h_total_ext_surco, l_total_ext_surco, vm_ext_surco)
plot_descomposicion('Gráfico 8: Descomposición de Esfuerzos (V.E.S.)', angulos_grados, h_total_ext_ves, l_total_ext_ves, vm_ext_ves)
plot_descomposicion('Gráfico 9: Descomposición de Esfuerzos (S.J.L.)', angulos_grados, h_total_ext_sjl, l_total_ext_sjl, vm_ext_sjl)

# --- Gráficos de Barras Comparativos ---
plt.figure(figsize=(10, 7))
plt.get_current_fig_manager().set_window_title('Gráfico 10: Comparación de Esfuerzos Máximos')
bars1 = plt.bar(distritos, esfuerzos_maximos, color=colores_barras)
plt.axhline(y=smys, color='r', linestyle='--', label=f'Límite Elástico (SMYS = {smys} MPa)')
plt.ylabel('Esfuerzo Máximo de Von Mises (MPa)', fontsize=12)
plt.title('Comparación de Esfuerzos Máximos por Distrito', fontsize=14)
plt.legend()
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
plt.get_current_fig_manager().set_window_title('Gráfico 11: Comparación de Factores de Seguridad')
bars2 = plt.bar(distritos, factores_seguridad, color=colores_barras)
plt.axhline(y=1.0, color='r', linestyle='--', label='Límite de Falla (FS=1.0)')
plt.ylabel('Factor de Seguridad', fontsize=12)
plt.title('Comparación de Factores de Seguridad por Distrito', fontsize=14)
plt.legend()
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')
plt.tight_layout()
plt.show()
