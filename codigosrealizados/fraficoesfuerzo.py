import matplotlib.pyplot as plt
import numpy as np

# --- Datos extraídos de las tablas del Capítulo 6 ---

# Datos para los gráficos polares (Esfuerzo de Von Mises vs. Ángulo)
# Ángulos en grados
angulos_grados = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])
# Convertir ángulos a radianes para el gráfico polar
angulos_rad = np.deg2rad(angulos_grados)

# Esfuerzos de Von Mises (externos) para cada distrito
esfuerzos_surco = np.array([282.02, 281.91, 281.75, 280.93, 280.56, 280.96, 279.38, 275.92, 275.92, 275.90, 272.93, 272.93, 272.93])
esfuerzos_ves = np.array([449.03, 448.94, 448.83, 448.52, 448.24, 447.81, 450.80, 447.81, 448.24, 450.01, 448.83, 448.94, 450.56])
esfuerzos_sjl = np.array([162.23, 162.15, 162.04, 161.76, 161.50, 161.07, 161.27, 161.07, 161.50, 160.81, 162.04, 162.15, 160.45])

# Límite elástico del material (SMYS)
smys = 448  # MPa

# Datos para los gráficos de barras (Resumen comparativo)
distritos = ['Surco', 'Villa El Salvador', 'S.J. Lurigancho']
esfuerzos_maximos = [282.02, 450.80, 163.24]
factores_seguridad = [1.588, 0.994, 2.744]

# --- Creación de los Gráficos ---

# Configuración general de la figura
fig = plt.figure(figsize=(18, 10))
plt.style.use('seaborn-v0_8-whitegrid')

# --- Gráficos Polares ---

# Gráfico para Surco
ax1 = fig.add_subplot(2, 3, 1, polar=True)
ax1.plot(angulos_rad, esfuerzos_surco, marker='o', color='b', label='Esfuerzo en Surco')
# Añadir el círculo del límite elástico
theta_smys = np.linspace(0, 2 * np.pi, 100)
r_smys = np.full_like(theta_smys, smys)
ax1.plot(theta_smys, r_smys, 'r--', label=f'Límite Elástico (SMYS = {smys} MPa)')
ax1.set_title('Distribución de Esfuerzo en Surco', fontsize=14, pad=20)
ax1.set_rlabel_position(22.5)
ax1.set_rmax(max(esfuerzos_ves) * 1.1) # Usar el máximo global para consistencia
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Gráfico para Villa El Salvador
ax2 = fig.add_subplot(2, 3, 2, polar=True)
ax2.plot(angulos_rad, esfuerzos_ves, marker='o', color='g', label='Esfuerzo en V.E.S.')
ax2.plot(theta_smys, r_smys, 'r--', label=f'Límite Elástico (SMYS = {smys} MPa)')
ax2.set_title('Distribución de Esfuerzo en Villa El Salvador', fontsize=14, pad=20)
ax2.set_rlabel_position(22.5)
ax2.set_rmax(max(esfuerzos_ves) * 1.1)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Gráfico para San Juan de Lurigancho
ax3 = fig.add_subplot(2, 3, 3, polar=True)
ax3.plot(angulos_rad, esfuerzos_sjl, marker='o', color='purple', label='Esfuerzo en S.J.L.')
ax3.plot(theta_smys, r_smys, 'r--', label=f'Límite Elástico (SMYS = {smys} MPa)')
ax3.set_title('Distribución de Esfuerzo en S.J. Lurigancho', fontsize=14, pad=20)
ax3.set_rlabel_position(22.5)
ax3.set_rmax(max(esfuerzos_ves) * 1.1)
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))


# --- Gráficos de Barras ---

# Gráfico de Esfuerzos Máximos
ax4 = fig.add_subplot(2, 2, 3)
bars1 = ax4.bar(distritos, esfuerzos_maximos, color=['blue', 'red', 'green'])
ax4.axhline(y=smys, color='r', linestyle='--', label=f'Límite Elástico (SMYS = {smys} MPa)')
ax4.set_ylabel('Esfuerzo Máximo de Von Mises (MPa)', fontsize=12)
ax4.set_title('Comparación de Esfuerzos Máximos por Distrito', fontsize=14)
ax4.legend()
# Añadir etiquetas de valor sobre las barras
for bar in bars1:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')


# Gráfico de Factores de Seguridad
ax5 = fig.add_subplot(2, 2, 4)
bars2 = ax5.bar(distritos, factores_seguridad, color=['blue', 'red', 'green'])
ax5.axhline(y=1.0, color='r', linestyle='--', label='Límite de Falla (FS=1.0)')
ax5.set_ylabel('Factor de Seguridad', fontsize=12)
ax5.set_title('Comparación de Factores de Seguridad por Distrito', fontsize=14)
ax5.legend()
# Añadir etiquetas de valor sobre las barras
for bar in bars2:
    yval = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')


# Ajustar diseño y mostrar
plt.tight_layout(pad=3.0)
plt.show()
