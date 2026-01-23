import numpy as np
import scipy.linalg as la
import pandas as pd
import logging
import sys

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL ENTORNO Y LOGGING (Estándar de Ingeniería)
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analisis_modal.log", mode='w')
    ]
)


def main():
    try:
        logging.info("=== INICIO DEL PROCESO DE ANÁLISIS MODAL ===")
        logging.info("Cargando matrices del sistema (Datos del Prompt)...")

        # ---------------------------------------------------------
        # 1. Definición de Matrices (Hardcoded para reproducibilidad exacta)
        # ---------------------------------------------------------
        # Matriz de Masa (M) - Valores exactos de tu tabla
        M = np.array([
            [43.036, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 38.483, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 24.924, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 43.036, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 38.483, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 24.924, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4319.295, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3358.883, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 2186.995]
        ])

        # Matriz de Rigidez (K) - Valores exactos de tu tabla (NO usar los del txt)
        K = np.array([
            [56737.5516, -35175.8669, 0.0000, 4960.7180, -2614.3529, 0.0000, -107974.0712, 59580.5673, 0.0000],
            [-35175.8669, 65547.7979, -30371.9310, -2614.3529, 5228.7058, -2614.3529, 80428.0440, -85985.1536,
             34818.2909],
            [0.0000, -30371.9310, 30371.9310, 0.0000, -2614.3529, 2614.3529, 0.0000, 26404.5863, -34818.2909],
            [4960.7180, -2614.3529, 0.0000, 60254.3918, -38840.8529, 0.0000, 149981.5204, -65590.5473, 0.0000],
            [-2614.3529, 5228.7058, -2614.3529, -38840.8529, 68073.8341, -29232.9812, -129201.5305, 28261.5728,
             14147.1642],
            [0.0000, -2614.3529, 2614.3529, 0.0000, -29232.9812, 29232.9812, 0.0000, 37328.9745, -14147.1642],
            [-107974.0712, 80428.0440, 0.0000, 149981.5204, -129201.5305, 0.0000, 10022738.6982, -5881389.8959, 0.0000],
            [59580.5673, -85985.1536, 26404.5863, -65590.5473, 28261.5728, 37328.9745, -5881389.8959, 10124411.0973,
             -4374881.1107],
            [0.0000, 34818.2909, -34818.2909, 0.0000, 14147.1642, -14147.1642, 0.0000, -4374881.1107, 4375382.5453]
        ])

        # ---------------------------------------------------------
        # 2. Solución Numérica (Eigenproblem)
        # ---------------------------------------------------------
        logging.info("Resolviendo problema de valores propios generalizado (K·phi = w^2·M·phi)...")
        evals, evecs = la.eigh(K, M, type=1)

        # ---------------------------------------------------------
        # 3. Post-Procesamiento y Normalización
        # ---------------------------------------------------------
        logging.info("Normalizando vectores respecto a la masa (Phi^T * M * Phi = 1)...")

        # Normalización de Masa (Esencial para que coincida con tu tabla)
        for i in range(evecs.shape[1]):
            m_gen_i = evecs[:, i].T @ M @ evecs[:, i]
            scale_factor = 1.0 / np.sqrt(m_gen_i)
            evecs[:, i] *= scale_factor

        # --- ALINEACIÓN DE SIGNOS ---
        # Forzar signos para coincidir con tu Excel
        target_signs_x1 = np.array([-1, 1, 1, -1, -1, 1, -1, -1, 1])

        for i in range(9):
            current_sign = np.sign(evecs[0, i])
            desired_sign = target_signs_x1[i]
            if current_sign != desired_sign and current_sign != 0:
                evecs[:, i] *= -1

        # Cálculos de Periodos
        omega = np.sqrt(evals)
        periods = 2 * np.pi / omega
        frequencies = omega / (2 * np.pi)

        # ---------------------------------------------------------
        # 4. Cálculo de Parámetros Modales
        # ---------------------------------------------------------
        # Vectores de influencia R
        R_x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
        R_y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
        R_theta = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

        # L = phi^T * M * R
        L_x = evecs.T @ M @ R_x
        L_y = evecs.T @ M @ R_y
        L_theta = evecs.T @ M @ R_theta

        # Masa Efectiva = L^2 / M_gen (M_gen = 1.0)
        M_eff_x = L_x ** 2
        M_eff_y = L_y ** 2
        M_eff_theta = L_theta ** 2

        # Masas Totales (calculadas directo de M para exactitud)
        Total_Mass_X = np.sum(np.diag(M)[0:3])
        Total_Mass_Y = np.sum(np.diag(M)[3:6])
        Total_Mass_Theta = np.sum(np.diag(M)[6:9])

        # Porcentajes
        Per_X = (M_eff_x / Total_Mass_X) * 100
        Per_Y = (M_eff_y / Total_Mass_Y) * 100
        Per_Theta = (M_eff_theta / Total_Mass_Theta) * 100

        # ---------------------------------------------------------
        # 5. Generación de Reporte Formateado (Excel-like)
        # ---------------------------------------------------------
        row_labels = ['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'giro1', 'giro2', 'giro3']
        col_labels = [f'modo {i + 1}' for i in range(9)]

        # DataFrame de Modos
        df_modes = pd.DataFrame(evecs, index=row_labels, columns=col_labels)
        df_modes['MASAS'] = np.diag(M)

        # Configuración para mostrar todo
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.5f}'.format)

        # --- TABLA XX ---
        df_params_X = pd.DataFrame([L_x, np.ones(9), L_x, M_eff_x, Per_X],
                                   index=['Lx', 'M', 'Lx/M', 'L2/M', 'porcentaje'],
                                   columns=col_labels)

        # --- TABLA YY ---
        df_params_Y = pd.DataFrame([L_y, np.ones(9), L_y, M_eff_y, Per_Y],
                                   index=['Ly', 'M', 'Ly/M', 'L2/M', 'porcentaje'],
                                   columns=col_labels)

        # --- TABLA GIRO ---
        df_params_Giro = pd.DataFrame([L_theta, np.ones(9), L_theta, M_eff_theta, Per_Theta],
                                      index=['Lgiro', 'M', 'L/M', 'L2/M', 'porcentaje'],
                                      columns=col_labels)

        # --- RESUMEN PERIODOS ---
        df_resumen = pd.DataFrame({
            'Periodo (seg)': periods,
            'XX': Per_X,
            'YY': Per_Y,
            'GIRO': Per_Theta
        }, index=col_labels)

        # Guardar archivo con formato personalizado para que sea IDÉNTICO
        output_file = 'resultados_analisis.txt'
        logging.info(f"Guardando resultados en {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== RESULTADOS DEL ANÁLISIS MODAL ===\n\n")

            f.write("--- Parametros generalizados DIRECCION XX ---\n")
            f.write(df_modes.to_string())
            f.write("\n\n")
            f.write(df_params_X.to_string(float_format="{:.4f}".format))
            f.write(f"\nMASA TOTAL XX: {Total_Mass_X:.4f}\n\n")

            f.write("--- Parametros generalizados DIRECCION YY ---\n")
            f.write(df_modes.to_string())
            f.write("\n\n")
            f.write(df_params_Y.to_string(float_format="{:.4f}".format))
            f.write(f"\nMASA TOTAL YY: {Total_Mass_Y:.4f}\n\n")

            f.write("--- Parametros generalizados DIRECCION GIRO ZZ ---\n")
            f.write(df_modes.to_string())
            f.write("\n\n")
            f.write(df_params_Giro.to_string(float_format="{:.4f}".format))
            f.write(f"\nMASA TOTAL GIRO: {Total_Mass_Theta:.4f}\n\n")

            f.write("--- Analisis de resultados: Periodos y Participación ---\n")
            f.write(df_resumen.to_string(float_format="{:.4f}".format))
            f.write("\n")
            f.write(f"Suma XX: {Per_X.sum():.2f}%\n")
            f.write(f"Suma YY: {Per_Y.sum():.2f}%\n")
            f.write(f"Suma GIRO: {Per_Theta.sum():.2f}%\n")

        print("\n--- Periodos (s) ---")
        print(periods)
        print("\n--- Primeros componentes del Modo 1 (Verificación) ---")
        print(df_modes.iloc[:, 0])

    except Exception as e:
        logging.error("FALLO CRÍTICO", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()