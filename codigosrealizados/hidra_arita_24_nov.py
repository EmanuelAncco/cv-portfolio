import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
from scipy.optimize import newton
from datetime import datetime
import yaml

# --- CONFIGURACIÓN DE LOGGING ---
# Principio: Trazabilidad completa. Escribimos a consola y archivo.
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ChannelExperiment:
    """
    Clase para manejar la validación hidráulica de un canal experimental con geometría fija.
    No asume Máxima Eficiencia Hidráulica, sino que respeta las restricciones físicas (b fijo).
    """

    def __init__(self, b_solera_m, z_talud, slope_m_m):
        """
        Inicializa la geometría FIJA del canal experimental.

        Args:
            b_solera_m (float): Ancho de la base del canal real (metros).
            z_talud (float): Pendiente lateral (0 para rectangular).
            slope_m_m (float): Pendiente longitudinal del canal (m/m).
        """
        self.b = b_solera_m
        self.z = z_talud
        self.s = slope_m_m
        logger.info(f"Canal inicializado: b={self.b}m, z={self.z}, s={self.s}")

    def _area(self, y):
        return (self.b + self.z * y) * y

    def _perimeter(self, y):
        return self.b + 2 * y * np.sqrt(1 + self.z ** 2)

    def calculate_theoretical_y(self, Q_m3s, n_roughness):
        """
        Calcula el tirante normal (y) resolviendo la ec. de Manning para un Q y n dados.
        Usa método numérico (Newton-Raphson) ya que y está implícito.

        Manning: Q = (1/n) * A * R^(2/3) * S^(1/2)
        """
        sqrt_s = np.sqrt(self.s)

        # Función objetivo: f(y) = Q_calc - Q_target = 0
        def manning_error(y):
            if y <= 0: return -Q_m3s  # Evitar valores negativos físicos
            A = self._area(y)
            P = self._perimeter(y)
            R = A / P
            Q_calc = (1 / n_roughness) * A * (R ** (2 / 3)) * sqrt_s
            return Q_calc - Q_m3s

        try:
            # Estimación inicial: Asumir canal ancho como semilla (y ~ (Q*n / (b*s^0.5))^(3/5))
            y_guess = (Q_m3s * n_roughness / (self.b * sqrt_s)) ** (3 / 5)
            y_solved = newton(manning_error, x0=y_guess, maxiter=50)
            return y_solved
        except RuntimeError as e:
            logger.error(f"Fallo en convergencia numérica para Q={Q_m3s}: {e}")
            return None

    def calibrate_roughness(self, observed_data):
        """
        Calcula el 'n' de Manning real basado en los datos observados (Q y y medidos).
        Esto es ingeniería inversa: tenemos la realidad, buscamos el parámetro del modelo.
        """
        logger.info("Iniciando calibración de rugosidad (n) basada en datos experimentales...")
        results = []

        n_calculated_values = []

        for i, obs in enumerate(observed_data):
            Q_obs = obs['Q']
            y_obs = obs['y_measured']

            # Despejar n de Manning: n = (A * R^(2/3) * S^(1/2)) / Q
            try:
                A = self._area(y_obs)
                P = self._perimeter(y_obs)
                R = A / P
                n_calc = (A * (R ** (2 / 3)) * np.sqrt(self.s)) / Q_obs

                results.append({
                    'id': i + 1,
                    'Q_obs': Q_obs,
                    'y_obs': y_obs,
                    'n_derived': n_calc
                })
                n_calculated_values.append(n_calc)
                logger.info(f"Exp #{i + 1}: Q={Q_obs:.5f}, y_real={y_obs:.4f} -> n_calculado={n_calc:.5f}")

            except Exception as e:
                logger.error(f"Error procesando dato #{i + 1}: {e}")

        # Estadística básica
        if n_calculated_values:
            n_mean = np.mean(n_calculated_values)
            n_std = np.std(n_calculated_values)
            logger.info(f"--- Calibración Finalizada ---")
            logger.info(f"n Promedio del canal: {n_mean:.5f} (+/- {n_std:.5f})")

            # Guardar artefacto YAML
            artifact = {
                'channel_geometry': {'b': self.b, 'z': self.z, 's': self.s},
                'calibration_results': {
                    'n_mean': float(n_mean),
                    'n_std': float(n_std),
                    'samples': len(results)
                }
            }
            with open(os.path.join(log_dir, 'calibration_results.yaml'), 'w') as f:
                yaml.dump(artifact, f)

            return n_mean, results
        else:
            logger.error("No se pudo calibrar. Revisa los datos de entrada.")
            return None, []


# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    try:
        # 1. Configuración del Experimento (Datos REALES de tu canal físico)
        # SUPOSICIÓN: Tu canal mide 5cm de ancho (0.05m) según la imagen izquierda.
        ANCHO_CANAL_REAL_M = 0.05
        TALUD = 0.0  # Rectangular
        PENDIENTE = 0.002  # m/m

        # 2. Datos Observados (Tus 3 experimentos)
        # Reemplaza 'y_measured' con lo que mediste con tu limnímetro/regla
        experimentos = [
            {'Q': 0.0001057, 'y_measured': 0.0120},  # Ejemplo: dato cercano al calculado
            {'Q': 0.0002000, 'y_measured': 0.0185},  # Ejemplo hipotético
            {'Q': 0.0000500, 'y_measured': 0.0080}  # Ejemplo hipotético
        ]

        logger.info("Iniciando script de validación hidráulica...")

        # Instanciar el modelo físico (NO el de máxima eficiencia)
        channel = ChannelExperiment(ANCHO_CANAL_REAL_M, TALUD, PENDIENTE)

        # Paso A: Calibrar el modelo (Encontrar el 'n' real)
        n_real, calibration_data = channel.calibrate_roughness(experimentos)

        # Paso B: Generar Curva de Validación
        if n_real:
            # Generar rango de caudales para graficar la curva teórica calibrada
            q_range = np.linspace(0.00001, 0.00025, 50)
            y_theoretical = [channel.calculate_theoretical_y(q, n_real) for q in q_range]

            # Graficar
            plt.figure(figsize=(10, 6))

            # Curva teórica (Modelo Calibrado)
            plt.plot(q_range, y_theoretical, label=f'Modelo Calibrado (n={n_real:.4f}, b={ANCHO_CANAL_REAL_M}m)',
                     color='blue')

            # Puntos experimentales
            q_exp = [d['Q'] for d in calibration_data]
            y_exp = [d['y_obs'] for d in calibration_data]
            plt.scatter(q_exp, y_exp, color='red', s=100, label='Datos Experimentales', zorder=5)

            # Comparación con Máxima Eficiencia (Para ilustrar el error)
            # MEH para canal rectangular: b = 2y. Sustituyendo en Manning obtenemos una curva distinta.
            # Solo ilustrativo para mostrar por qué NO usarlo.
            y_meh = []
            for q in q_range:
                # Para MEH rectangular: A=2y^2, P=4y, R=y/2. Q = (1/n)*2y^2*(y/2)^(2/3)*S^0.5
                # Resolviendo para y... simplificado:
                # Q*n/S^0.5 = 2 * y^2 * (0.5*y)^(2/3) = 2 * 0.63 * y^(2.66)
                val = (q * n_real) / np.sqrt(PENDIENTE)
                y_opt = (val / (2 * (0.5 ** (2 / 3)))) ** (3 / 8)  # Aprox potencia
                y_meh.append(y_opt)

            plt.plot(q_range, y_meh, label='Curva Máxima Eficiencia (DISEÑO - NO USAR)', color='green', linestyle='--',
                     alpha=0.5)

            plt.title("Validación: Modelo Físico Real vs Diseño Teórico")
            plt.xlabel("Caudal (m3/s)")
            plt.ylabel("Tirante (m)")
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            output_plot = os.path.join(log_dir, "validacion_hidraulica.png")
            plt.savefig(output_plot)
            logger.info(f"Gráfico de validación guardado en: {output_plot}")

    except Exception as e:
        logger.critical(f"Error crítico en el proceso: {e}", exc_info=True)