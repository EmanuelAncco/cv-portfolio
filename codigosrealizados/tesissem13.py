import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# --- ESTILO Y DATOS ---
# Usar un estilo profesional para los gráficos
style.use('seaborn-v0_8-whitegrid')

# Datos calculados de la Tabla 4.1 del documento de la tesis
# Estos son los resultados de las ecuaciones para el gasoducto de 20 pulgadas.
angulos_grados = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])
angulos_rad = np.deg2rad(angulos_grados) # Convertir a radianes para el gráfico polar
fuerza_axial = np.array([190, 158, 77, -45, -186, -318, -420, -481, -506, -511, -524, -547, -561])
momento_flector = np.array([156, 136, 91, 30, -34, -83, -111, -123, -126, -126, -127, -128, -129])
esfuerzo_ext = np.array([10.42, 9.18, 6.14, 1.95, -2.33, -5.58, -7.42, -8.24, -8.45, -8.48, -8.57, -8.62, -8.64])
esfuerzo_int = np.array([-10.38, -9.00, -5.98, -2.04, 2.23, 5.45, 7.28, 8.09, 8.32, 8.35, 8.43, 8.48, 8.52])

# Parámetros de la tubería para el gráfico de sección
t = 0.0095 # espesor en metros

# --- GRÁFICO 1: ESFUERZOS INTERNOS VS. ÁNGULO ---
plt.figure(figsize=(12, 7))
plt.plot(angulos_grados, fuerza_axial, 'o-', label='Fuerza Axial (N)', color='royalblue')
plt.plot(angulos_grados, momento_flector, 's-', label='Momento Flector (N·m)', color='firebrick')
plt.title('Distribución de Esfuerzos Internos por Peso Propio', fontsize=16, fontweight='bold')
plt.xlabel('Ángulo (θ) desde la base [Grados]', fontsize=12)
plt.ylabel('Magnitud', fontsize=12)
plt.xticks(np.arange(0, 181, 15))
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('grafico_esfuerzos_internos.png') # Guardar el gráfico
#plt.show() # Descomentar para mostrar el gráfico al ejecutar

# --- GRÁFICO 2: ESFUERZOS EN FIBRAS VS. ÁNGULO (EL MÁS IMPORTANTE) ---
plt.figure(figsize=(12, 7))
plt.plot(angulos_grados, esfuerzo_ext, 'o-', label='Esfuerzo en Fibra Externa ($\sigma_{ext}$)', color='darkorange')
plt.plot(angulos_grados, esfuerzo_int, 's-', label='Esfuerzo en Fibra Interna ($\sigma_{int}$)', color='teal')
# Marcar los puntos de esfuerzo máximo
plt.axhline(0, color='black', linewidth=0.5, linestyle='--') # Línea de cero esfuerzo
plt.title('Distribución de Esfuerzos en las Fibras por Peso Propio', fontsize=16, fontweight='bold')
plt.xlabel('Ángulo (θ) desde la base [Grados]', fontsize=12)
plt.ylabel('Esfuerzo (MPa)', fontsize=12)
plt.xticks(np.arange(0, 181, 15))
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('grafico_esfuerzos_fibras.png') # Guardar el gráfico
#plt.show() # Descomentar para mostrar el gráfico al ejecutar


# --- GRÁFICO 3: DIAGRAMA POLAR DEL MOMENTO FLECTOR ---
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))
# Dibujar el círculo que representa la tubería (sin deformar)
ax.plot(angulos_rad, np.full_like(angulos_rad, 1), color='gray', linestyle='--', label='Sección de la Tubería')
# Escalar el momento para una mejor visualización
# El signo negativo es para que los momentos positivos se dibujen hacia afuera
factor_escala = 0.005
ax.plot(angulos_rad, 1 - momento_flector * factor_escala, color='firebrick', linewidth=2, label='Momento Flector')
# Rellenar las áreas para una mejor visualización
ax.fill(angulos_rad, 1 - momento_flector * factor_escala, alpha=0.2, color='firebrick')
ax.set_theta_zero_location('S') # Poner 0 grados en la parte inferior (base)
ax.set_theta_direction(-1) # Dirección de los ángulos en sentido horario
ax.set_yticklabels([]) # Ocultar las etiquetas del radio
plt.title('Diagrama Polar del Momento Flector ($M_\\theta$)', fontsize=16, fontweight='bold')
plt.legend()
plt.savefig('diagrama_polar_momento.png') # Guardar el gráfico
#plt.show() # Descomentar para mostrar el gráfico al ejecutar


# --- GRÁFICO 4: DIAGRAMA DE ESFUERZO EN LA SECCIÓN CRÍTICA (BASE, θ=0°) ---
# Datos para el punto más crítico
N_critico = fuerza_axial[0]
M_critico = momento_flector[0]
y = np.linspace(-t/2, t/2, 11) # Posición a través del espesor

# Calcular los componentes del esfuerzo
esfuerzo_axial_componente = N_critico / (t * 1) / 1e6 # en MPa
esfuerzo_flexion_componente = (M_critico * y) / (t**3 / 12) / 1e6 # en MPa
esfuerzo_total = esfuerzo_axial_componente + esfuerzo_flexion_componente

# Crear el gráfico
plt.figure(figsize=(10, 8))
# 1. Componente Axial
plt.plot([esfuerzo_axial_componente, esfuerzo_axial_componente], [-t/2, t/2], '--', color='royalblue', label='Componente Axial ($\sigma_N$)')
plt.fill_betweenx([-t/2, t/2], 0, esfuerzo_axial_componente, color='royalblue', alpha=0.2)
for pos in np.linspace(-t/2, t/2, 11):
    plt.arrow(0, pos, esfuerzo_axial_componente, 0, head_width=0.00015, head_length=0.2, fc='royalblue', ec='royalblue')

# 2. Componente de Flexión
plt.plot(esfuerzo_flexion_componente, y, '--', color='firebrick', label='Componente de Flexión ($\sigma_M$)')
plt.fill_betweenx(y, 0, esfuerzo_flexion_componente, where=esfuerzo_flexion_componente>=0, color='firebrick', alpha=0.2, interpolate=True)
plt.fill_betweenx(y, 0, esfuerzo_flexion_componente, where=esfuerzo_flexion_componente<=0, color='firebrick', alpha=0.2, interpolate=True)
for i, pos in enumerate(y):
    if esfuerzo_flexion_componente[i] != 0:
        plt.arrow(0, pos, esfuerzo_flexion_componente[i], 0, head_width=0.00015, head_length=0.2, fc='firebrick', ec='firebrick')

# 3. Esfuerzo Total (en un gráfico separado para mayor claridad)
plt.figure(figsize=(8, 8))
plt.plot(esfuerzo_total, y, '-', color='black', linewidth=2.5, label='Esfuerzo Total ($\sigma_{total}$)')
plt.fill_betweenx(y, 0, esfuerzo_total, where=esfuerzo_total>=0, color='black', alpha=0.2, interpolate=True)
plt.fill_betweenx(y, 0, esfuerzo_total, where=esfuerzo_total<=0, color='black', alpha=0.2, interpolate=True)
for i, pos in enumerate(y):
    if esfuerzo_total[i] != 0:
        plt.arrow(0, pos, esfuerzo_total[i], 0, head_width=0.00015, head_length=0.2, fc='black', ec='black')


# Formato del gráfico
plt.axvline(0, color='gray', linewidth=0.7)
plt.title('Distribución de Esfuerzo en la Sección Crítica (Base, $\\theta=0^\\circ$)', fontsize=16, fontweight='bold')
plt.xlabel('Esfuerzo (MPa)', fontsize=12)
plt.ylabel('Posición en el Espesor (m)', fontsize=12)
plt.legend(fontsize=12)
plt.ylim(-t/2, t/2)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.gca().invert_yaxis() # Para que la fibra externa (+t/2) esté arriba
plt.savefig('diagrama_esfuerzo_seccion.png') # Guardar el gráfico
plt.show() # Mostrar todos los gráficos al final
