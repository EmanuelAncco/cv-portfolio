import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch, Rectangle

# --- Constantes y Parámetros Globales ---
# Estas son las constantes físicas que se usarán en los cálculos.
E_ACERO = 2.07e11  # Módulo de Young en Pa
NU_ACERO = 0.3  # Coeficiente de Poisson
ALPHA_T_ACERO = 1.2e-5  # Coeficiente de Expansión Térmica en °C⁻¹

# --- Datos de los Casos de Estudio ---
# Aquí se definen los parámetros específicos para cada uno de los tres distritos
# que estás analizando en tu tesis.
CASOS_DE_ESTUDIO = {
    "Surco": {
        "D": 0.508, "t": 0.00953, "p_i": 50e5, "delta_T": -15,
        "PGV": 0.50, "C": 400, "alpha_seismic": 1.0, "steel_grade": "API 5L X65"
    },
    "Villa El Salvador": {
        "D": 0.254, "t": 0.00635, "p_i": 10e5, "delta_T": -15,
        "PGV": 0.60, "C": 250, "alpha_seismic": 1.0, "steel_grade": "API 5L X65"
    },
    "S.J. Lurigancho": {
        "D": 0.254, "t": 0.00635, "p_i": 10e5, "delta_T": -15,
        "PGV": 0.40, "C": 400, "alpha_seismic": 1.0, "steel_grade": "API 5L X65"
    }
}


def calcular_esfuerzos_principales(params):
    """
    Calcula los esfuerzos fundamentales (circunferencial y longitudinal)
    basado en los parámetros de entrada. Esta es una versión simplificada
    del motor de cálculo principal, enfocada en los valores que se mostrarán
    en el esquema.
    """
    D_e = params['D']
    t = params['t']
    p_i = params['p_i']

    # Esfuerzo Circunferencial (Hoop Stress) por presión interna.
    D_i = D_e - 2 * t
    sigma_h_p = (p_i * D_i) / (2 * t) if t > 0 else 0

    # Esfuerzo Longitudinal Total (Uniforme)
    # 1. Componente por presión (Efecto Poisson)
    sigma_a_p = NU_ACERO * sigma_h_p
    # 2. Componente por temperatura
    sigma_a_T = E_ACERO * ALPHA_T_ACERO * params['delta_T']
    # 3. Componente por sismo (onda de propagación)
    sigma_a_w = E_ACERO * params['alpha_seismic'] * (params['PGV'] / params['C']) if params['C'] > 0 else 0

    sigma_L_total = sigma_a_p + sigma_a_T + sigma_a_w

    return {
        "sigma_h": sigma_h_p,
        "sigma_L": sigma_L_total
    }


def crear_esquema_grafico(nombre_caso, datos_entrada, resultados):
    """
    Genera y guarda un gráfico esquemático de los esfuerzos en la tubería
    para un caso de estudio específico con un layout mejorado.
    """
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Esquema de Esfuerzos Principales: Caso {nombre_caso}", fontsize=16, weight='bold')

    # Definir la geometría de los subplots para un control preciso
    ax_transversal = fig.add_axes([0.05, 0.1, 0.35, 0.7])
    ax_longitudinal = fig.add_axes([0.45, 0.1, 0.5, 0.7])

    # --- Configuración de la Vista Transversal ---
    ax_transversal.set_aspect('equal', adjustable='box')
    ax_transversal.axis('off')
    ax_transversal.set_title("Vista A: Sección Transversal", fontsize=12, pad=20)
    ax_transversal.set_xlim(-1, 1)
    ax_transversal.set_ylim(-1, 1)

    R = 0.5  # Radio normalizado para el dibujo
    t_norm = (datos_entrada['t'] / (datos_entrada['D'] / 2)) * R  # Espesor normalizado
    r_int = R - t_norm

    circ_ext = Circle((0, 0), R, facecolor='#B0B0B0', edgecolor='black', linewidth=1.5)
    circ_int = Circle((0, 0), r_int, facecolor='white', edgecolor='black', linewidth=1)
    ax_transversal.add_patch(circ_ext)
    ax_transversal.add_patch(circ_int)
    ax_transversal.text(0, 0, 'p', color='blue', ha='center', va='center', fontsize=14, style='italic')
    ax_transversal.plot([-0.8, 0.8], [0, 0], 'k--', linewidth=1)
    ax_transversal.text(-0.9, 0, 'A', fontsize=12)
    ax_transversal.text(0.9, 0, 'A', fontsize=12)

    # --- Configuración de la Vista Longitudinal y de Información ---
    ax_longitudinal.axis('off')
    ax_longitudinal.set_title("Corte A-A y Resultados", fontsize=12, pad=20)
    ax_longitudinal.set_xlim(0, 2)
    ax_longitudinal.set_ylim(-1, 1)

    # Dibujo del corte longitudinal
    long_len = 1.0
    ax_longitudinal.add_patch(
        Rectangle((0.1, r_int), long_len, t_norm, facecolor='#B0B0B0', edgecolor='black', linewidth=1.5))
    ax_longitudinal.add_patch(
        Rectangle((0.1, -R), long_len, t_norm, facecolor='#B0B0B0', edgecolor='black', linewidth=1.5))
    ax_longitudinal.text(0.1 + long_len / 2, 0, 'Pared Tubería', ha='center', va='center', style='italic', alpha=0.7)

    # Elemento de Esfuerzo
    elem_x, elem_y, elem_size = 0.6, 0, 0.2
    ax_longitudinal.add_patch(
        Rectangle((elem_x - elem_size / 2, elem_y - elem_size / 2), elem_size, elem_size, fill=False, edgecolor='k',
                  linestyle='--'))

    # Flechas de Esfuerzo
    arrow_props = dict(arrowstyle='<->', color='red', lw=1.5)
    ax_longitudinal.annotate('', xy=(elem_x + elem_size / 2, elem_y), xytext=(elem_x - elem_size / 2, elem_y),
                             arrowprops=arrow_props)
    ax_longitudinal.text(elem_x, elem_y + elem_size * 0.7, r'$\sigma_L$', color='red', ha='center', va='bottom',
                         fontsize=14)
    ax_longitudinal.annotate('', xy=(elem_x, elem_y + elem_size / 2), xytext=(elem_x, elem_y - elem_size / 2),
                             arrowprops=arrow_props)
    ax_longitudinal.text(elem_x + elem_size * 0.7, elem_y, r'$\sigma_h$', color='red', ha='left', va='center',
                         fontsize=14)

    # Cuadro de Información
    info_texto = (
        f"$\\bf{{Datos\ del\ Caso: {nombre_caso}}}$\n"
        f"--------------------------------------\n"
        f"Tubería: Acero {datos_entrada['steel_grade']}\n"
        f"Presión Interna: {datos_entrada['p_i'] / 1e5:.1f} bar\n\n"
        f"$\\bf{{Resultados\ Calculados:}}$\n"
        f"--------------------------------------\n"
        f"Esfuerzo Circunferencial ($\\sigma_h$): {resultados['sigma_h'] / 1e6:.2f} MPa\n"
        f"Esfuerzo Longitudinal ($\\sigma_L$): {resultados['sigma_L'] / 1e6:.2f} MPa"
    )

    ax_longitudinal.text(1.25, 0, info_texto,
                         ha='left', va='center', fontsize=11,
                         bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="black", lw=1))

    # Guardar el gráfico
    nombre_archivo = f"Esquema_Esfuerzos_{nombre_caso.replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"Gráfico guardado como: {nombre_archivo}")


# --- Bucle Principal de Ejecución ---
if __name__ == "__main__":
    print("Iniciando la generación de esquemas para los casos de estudio de la tesis...")
    for nombre, params in CASOS_DE_ESTUDIO.items():
        print(f"Calculando y graficando para el distrito: {nombre}")
        resultados_esfuerzos = calcular_esfuerzos_principales(params)
        crear_esquema_grafico(nombre, params, resultados_esfuerzos)
    print("\nGeneración de gráficos completada.")

