import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle


def configurar_estilo_graficos():

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
    })


def obtener_datos():
    
    datos = {
        'Surco': {
            'D_m': 0.508, 't_m': 0.00953,
            'theta_grados': np.array(
                [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315,
                 330, 345]),
            'ur_mm': np.array(
                [0.206, 0.206, 0.205, 0.202, 0.201, 0.202, 0.199, 0.192, 0.192, 0.192, 0.187, 0.187, 0.187, 0.187,
                 0.187, 0.192, 0.192, 0.192, 0.199, 0.202, 0.201, 0.202, 0.205, 0.206]),
            'sigma_h_mpa': np.array(
                [260.86, 260.55, 260.02, 257.96, 257.21, 258.04, 254.47, 248.96, 248.96, 248.94, 243.85, 243.85, 243.87,
                 243.85, 243.85, 248.94, 248.96, 248.96, 254.47, 258.04, 257.21, 257.96, 260.02, 260.55]),
            'sigma_l_mpa': np.array(
                [299.75, 299.66, 299.50, 298.88, 298.65, 298.90, 297.83, 296.18, 296.18, 296.17, 294.65, 294.65, 294.65,
                 294.65, 294.65, 296.17, 296.18, 296.18, 297.83, 298.90, 298.65, 298.88, 299.50, 299.66]),
            'von_mises_mpa': np.array(
                [282.02, 281.91, 281.75, 280.93, 280.56, 280.96, 279.38, 275.92, 275.92, 275.90, 272.93, 272.93, 272.93,
                 272.93, 272.93, 275.90, 275.92, 275.92, 279.38, 280.96, 280.56, 280.93, 281.75, 281.91]),
            'stress_profiles': {  # [sigma_h_int, sigma_h_ext] en MPa
                90: [242.58, 260.86],  # Clave (Top)
                0: [267.85, 254.47],  # Costado (Right)
                270: [267.19, 243.87],  # Base (Bottom)
                180: [267.85, 254.47]  # Costado (Left)
            }
        },
        'Villa El Salvador': {
            'D_m': 0.254, 't_m': 0.00635,
            'theta_grados': np.array(
                [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315,
                 330, 345]),
            'ur_mm': np.array(
                [-0.049, -0.049, -0.050, -0.050, -0.051, -0.052, -0.051, -0.052, -0.051, -0.052, -0.050, -0.049, -0.053,
                 -0.049, -0.050, -0.052, -0.051, -0.052, -0.051, -0.052, -0.051, -0.050, -0.050, -0.049]),
            'sigma_h_mpa': np.array(
                [61.66, 61.33, 60.91, 59.76, 58.74, 57.14, 57.49, 57.14, 58.74, 55.28, 60.91, 61.33, 53.38, 61.33,
                 60.91, 55.28, 58.74, 57.14, 57.49, 57.14, 58.74, 59.76, 60.91, 61.33]),
            'sigma_l_mpa': np.array(
                [478.04, 477.94, 477.81, 477.47, 477.16, 476.68, 476.79, 476.68, 477.16, 476.12, 477.81, 477.94, 475.55,
                 477.94, 477.81, 476.12, 477.16, 476.68, 476.79, 476.68, 477.16, 477.47, 477.81, 477.94]),
            'von_mises_mpa': np.array(
                [449.03, 448.94, 448.83, 448.52, 448.24, 447.81, 450.80, 447.81, 448.24, 450.01, 448.83, 448.94, 450.56,
                 448.94, 448.83, 450.01, 448.24, 447.81, 450.80, 447.81, 448.24, 448.52, 448.83, 448.94]),
            'stress_profiles': {
                90: [52.76, 61.66],  # Clave (Top)
                0: [64.09, 57.49],  # Costado (Right)
                270: [61.04, 53.38],  # Base (Bottom)
                180: [64.09, 57.49]  # Costado (Left)
            }
        },
        'S.J. Lurigancho': {
            'D_m': 0.254, 't_m': 0.00635,
            'theta_grados': np.array(
                [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315,
                 330, 345]),
            'ur_mm': np.array(
                [0.003, 0.003, 0.003, 0.002, 0.004, 0.004, 0.001, 0.004, 0.004, -0.000, 0.003, 0.003, -0.000, 0.003,
                 0.003, -0.000, 0.004, 0.004, 0.001, 0.004, 0.004, 0.002, 0.003, 0.003]),
            'sigma_h_mpa': np.array(
                [61.66, 61.33, 60.91, 59.76, 58.74, 57.14, 57.49, 57.14, 58.74, 55.28, 60.91, 61.33, 53.38, 61.33,
                 60.91, 55.28, 58.74, 57.14, 57.49, 57.14, 58.74, 59.76, 60.91, 61.33]),
            'sigma_l_mpa': np.array(
                [188.24, 188.14, 188.01, 187.67, 187.36, 186.88, 186.99, 186.88, 187.36, 186.32, 188.01, 188.14, 185.75,
                 188.14, 188.01, 186.32, 187.36, 186.88, 186.99, 186.88, 187.36, 187.67, 188.01, 188.14]),
            'von_mises_mpa': np.array(
                [162.23, 162.15, 162.04, 161.76, 161.50, 161.07, 161.27, 161.07, 161.50, 160.81, 162.04, 162.15, 160.45,
                 162.15, 162.04, 160.81, 161.50, 161.07, 161.27, 161.07, 161.50, 161.76, 162.04, 162.15]),
            'stress_profiles': {  # Mismas cargas estáticas que VES
                90: [52.76, 61.66],  # Clave (Top)
                0: [64.09, 57.49],  # Costado (Right)
                270: [61.04, 53.38],  # Base (Bottom)
                180: [64.09, 57.49]  # Costado (Left)
            }
        }
    }
    return datos


def graficar_seccion_con_esfuerzos(distrito, data):
    """
    Gráfico 6 (Nuevo): Genera la visualización solicitada por el asesor,
    mostrando la sección de la tubería y los perfiles de esfuerzo en puntos cardinales.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # --- 1. Dibujar la sección de la tubería ---
    D = data['D_m']
    t = data['t_m']
    R_ext = D / 2
    R_int = R_ext - t

    pipe_outer = Circle((0, 0), R_ext, facecolor='lightgrey', edgecolor='black', linewidth=1.5)
    pipe_inner = Circle((0, 0), R_int, facecolor='white', edgecolor='black', linewidth=1.5)
    ax.add_patch(pipe_outer)
    ax.add_patch(pipe_inner)

    # --- 2. Preparar el gráfico ---
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-R_ext * 2.5, R_ext * 2.5)
    ax.set_ylim(-R_ext * 2.5, R_ext * 2.5)
    ax.axis('off')
    ax.set_title(f"Distribución de Esfuerzo Circunferencial ($\sigma_h$) - {distrito}", fontsize=16, weight='bold',
                 y=0.95)

    # --- 3. Dibujar los perfiles de esfuerzo ---
    profiles = data['stress_profiles']
    max_stress_abs = max(abs(s) for p in profiles.values() for s in p)
    if max_stress_abs == 0: max_stress_abs = 1  # Evitar división por cero
    scale = R_ext / max_stress_abs  # Escala para que los diagramas se vean bien

    # Posiciones: { ángulo: ((x_offset, y_offset), rotación_grados) }
    positions = {
        90: ((0, R_ext + 0.1 * D), 0),  # Top
        0: ((R_ext + 0.1 * D, 0), 90),  # Right
        270: ((0, -R_ext - 0.1 * D), 0),  # Bottom
        180: ((-R_ext - 0.1 * D, 0), 90)  # Left
    }

    for angle, (offset, rotation) in positions.items():
        offset_x, offset_y = offset
        s_int, s_ext = profiles[angle]

        # Coordenadas del perfil de esfuerzo
        y_local = np.array([-t / 2, t / 2])
        stress_local = np.array([s_int, s_ext]) * scale

        # Rotar y trasladar
        theta_rad = np.deg2rad(rotation)
        rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                               [np.sin(theta_rad), np.cos(theta_rad)]])

        coords_local = np.vstack([stress_local, y_local])
        coords_rotated = rot_matrix @ coords_local

        # El vector de traslación debe tener la misma forma para la suma
        translation_vector = np.array([[offset_x], [offset_y]])
        coords_global = coords_rotated + translation_vector

        # Dibujar perfil
        ax.plot(coords_global[0, :], coords_global[1, :], 'k-', lw=1)

        # Rellenar tracción (rojo) y compresión (azul)
        y_fill = np.linspace(-t / 2, t / 2, 100)
        stress_fill = np.linspace(s_int, s_ext, 100) * scale

        coords_fill_local = np.vstack([stress_fill, y_fill])
        coords_fill_rotated = rot_matrix @ coords_fill_local
        coords_fill_global = coords_fill_rotated + translation_vector

        zero_line_local = np.vstack([np.zeros_like(y_fill), y_fill])
        zero_line_rotated = rot_matrix @ zero_line_local
        zero_line_global = zero_line_rotated + translation_vector

        ax.fill_between(coords_fill_global[0, :], zero_line_global[0, :], where=stress_fill > 0, interpolate=True,
                        color='salmon', alpha=0.7, zorder=10)
        ax.fill_between(coords_fill_global[0, :], zero_line_global[0, :], where=stress_fill < 0, interpolate=True,
                        color='cornflowerblue', alpha=0.7, zorder=10)

        # Anotaciones
        ha_map = {90: 'left', 0: 'center', 270: 'left', 180: 'center'}
        ax.text(coords_global[0, 0], coords_global[1, 0], f' {s_int:.1f}', ha=ha_map[angle], va='center', fontsize=9)
        ax.text(coords_global[0, 1], coords_global[1, 1], f' {s_ext:.1f}', ha=ha_map[angle], va='center', fontsize=9)

    fig.tight_layout()
    plt.savefig(f"grafico_seccion_esfuerzos_{distrito.replace(' ', '_')}.png", dpi=300)
    plt.show()


# --- Main execution block remains the same, you can add the new function call ---
if __name__ == '__main__':
    # --- Flujo Principal ---
    configurar_estilo_graficos()
    datos = obtener_datos()

    # Opcional: Descomentar para generar los otros gráficos
    # print("--- Generando Gráficos de Análisis General ---")
    # graficar_perfil_deformacion_polar(datos)
    # graficar_comparativa_deformaciones(datos)
    # graficar_esfuerzos_vs_angulo(datos)
    # graficar_contorno_von_mises_suavizado(datos)

    print("\n--- Generando Gráfico de Sección de Esfuerzos (Asesor) ---")
    for distrito, data_distrito in datos.items():
        print(f"Generando para {distrito}...")
        graficar_seccion_con_esfuerzos(distrito, data_distrito)

