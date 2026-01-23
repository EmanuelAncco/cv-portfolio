import matplotlib.pyplot as plt
import numpy as np

# --- 1. Simulación de Datos (Modelo Analítico) ---
# Se genera un rango de ángulos de 0 a 180 grados para la sección de la tubería.
# En un caso real, aquí usarías los datos de salida de tu modelo de Ansys o Matlab.
angulos_grados = np.linspace(0, 180, 200)  # Ángulo en grados
angulos_rad = np.radians(angulos_grados)      # Ángulo en radianes para cálculos

# --- Modelo para Esfuerzo Axial ---
# Esta es una función de ejemplo que simula la forma de tu Figura 3.15.
# Se basa en una función coseno para representar la compresión variable.
# DEBES REEMPLAZAR esta fórmula con la de tu propio estudio.
esfuerzo_constante = -3.18e4
amplitud_axial = 0.07e4
esfuerzo_axial = esfuerzo_constante - amplitud_axial * np.cos(2 * angulos_rad)

# --- Modelo para Momento Flector ---
# Esta es una función de ejemplo que simula la forma de tu Figura 3.16.
# Se basa en una función seno para representar la flexión.
# DEBES REEMPLAZAR esta fórmula con la de tu propio estudio.
amplitud_momento = 420
momento_flector = amplitud_momento * np.sin(angulos_rad) - 400 * np.cos(angulos_rad)**2 # Ejemplo más complejo

# --- 2. Creación de los Gráficos ---
# Creamos una figura con dos subplots, uno encima del otro.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('Análisis de Esfuerzos en la Sección Transversal de la Tubería', fontsize=16)

# --- Gráfico 1: Esfuerzo Axial ---
ax1.plot(angulos_grados, esfuerzo_axial, color='#E74C3C', linewidth=2.5, label='Modelo Analítico (Python)')
ax1.set_title('Esfuerzo Axial vs. Ángulo', fontsize=14)
ax1.set_xlabel('Ángulo (grados)', fontsize=12)
ax1.set_ylabel('Fuerza Axial (N/m)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()
# Formato del eje Y para que coincida con tu gráfico (notación científica)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# --- Gráfico 2: Momento Flector ---
ax2.plot(angulos_grados, momento_flector, color='#2E86C1', linewidth=2.5, label='Modelo Analítico (Python)')
ax2.set_title('Momento Flector vs. Ángulo', fontsize=14)
ax2.set_xlabel('Ángulo (grados)', fontsize=12)
ax2.set_ylabel('Momento (N.m)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# --- 3. Mostrar Gráficos ---
# Ajusta el espaciado para evitar que los títulos se superpongan.
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Muestra la ventana con los gráficos.
plt.show()
