import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# DATOS DE ENTRADA
# ---------------------------------------------------------
Q = 1.944544  # m3/s
b = 2.5  # m (Ancho)
g = 9.81  # m/s2
yn = 0.4442  # m (Tirante Normal)

# Definimos una altura de muro asumiendo un diseño estándar
# (Normalmente H = yn + borde libre).
# Si yn=0.44, un muro de 1.0m o 1.2m es razonable.
Altura_Muro_Supuesta = 1.0  # m (Ajustable)
Borde_Libre_Minimo = 0.30  # m (Seguridad requerida)
Y1_Maximo_Seguro = Altura_Muro_Supuesta - Borde_Libre_Minimo

# ---------------------------------------------------------
# CÁLCULO DE LA CURVA DE OPERACIÓN
# ---------------------------------------------------------
# Vamos a barrer aperturas desde 15cm hasta 60cm
aperturas = np.linspace(0.15, 0.60, 100)
y1_resultados = []
estados = []

for a in aperturas:
    # 1. Coeficiente de contracción aproximado
    Cc = 0.61
    y2 = Cc * a

    # 2. Iteración para encontrar y1 (Usando ecuación de descarga)
    # Q = Cd * b * a * sqrt(2 * g * y1)
    # Asumimos Cd constante inicial para despeje directo,
    # aunque en realidad varía ligeramente. Para diseño preliminar es válido.
    # Cd promedio aprox 0.58 - 0.60
    Cd_est = 0.59

    # Despejamos y1 de la ecuación de descarga:
    # y1 = (Q / (Cd * b * a))^2 / (2g)
    val = Q / (Cd_est * b * a)
    y1_calc = (val ** 2) / (2 * g)

    y1_resultados.append(y1_calc)

# ---------------------------------------------------------
# GRAFICACIÓN
# ---------------------------------------------------------
plt.figure(figsize=(10, 7))

# Curva Principal
plt.plot(aperturas * 100, y1_resultados, label='Nivel Aguas Arriba (y1)', color='#0056b3', linewidth=2.5)

# Líneas de Límite
plt.axhline(y=Altura_Muro_Supuesta, color='red', linestyle='--',
            label=f'Altura Muro (Supuesta {Altura_Muro_Supuesta}m)')
plt.axhline(y=Y1_Maximo_Seguro, color='orange', linestyle='--', label=f'Límite Seguro ({Y1_Maximo_Seguro}m)')
plt.axhline(y=yn, color='green', linestyle='-.', label=f'Tirante Normal ({yn}m)')

# Zona de Peligro
plt.fill_between(aperturas * 100, Altura_Muro_Supuesta, max(y1_resultados), color='red', alpha=0.1,
                 label='ZONA DESBORDE')
plt.fill_between(aperturas * 100, Y1_Maximo_Seguro, Altura_Muro_Supuesta, color='orange', alpha=0.1,
                 label='ZONA RIESGO')

# Encontrar el punto exacto donde entramos a zona segura
idx_seguro = next((i for i, y in enumerate(y1_resultados) if y <= Y1_Maximo_Seguro), None)
if idx_seguro:
    a_optimo = aperturas[idx_seguro]
    y1_optimo = y1_resultados[idx_seguro]
    plt.scatter(a_optimo * 100, y1_optimo, color='green', s=150, zorder=5, edgecolors='black')
    plt.annotate(f'¡PUNTO ÓPTIMO!\na = {a_optimo * 100:.1f} cm\ny1 = {y1_optimo:.2f} m',
                 xy=(a_optimo * 100, y1_optimo),
                 xytext=(a_optimo * 100 + 5, y1_optimo + 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

# Decoración
plt.title('Curva de Operación: Apertura vs. Nivel del Agua', fontsize=14)
plt.xlabel('Apertura de Compuerta "a" (cm)', fontsize=12)
plt.ylabel('Tirante Aguas Arriba y1 (m)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('curva_operacion_compuerta.png', dpi=300)
print("Gráfico generado: curva_operacion_compuerta.png")