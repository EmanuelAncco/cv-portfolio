import math
import logging

# ==============================================================================
# CONFIGURACIÓN DEL LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("diseno_canal.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# ==============================================================================
# CLASE AUXILIAR PARA CÁLCULOS HIDRÁULICOS
# ==============================================================================
class CanalRectangular:
    """Representa un canal rectangular y calcula sus propiedades hidráulicas."""

    def __init__(self, b, y, n, S, g=9.81, nombre="Canal"):
        self.b, self.y, self.n, self.S, self.g = b, y, n, S, g
        self.nombre = nombre
        self.area, self.perimetro_mojado, self.radio_hidraulico = None, None, None
        self.velocidad, self.caudal, self.numero_froude, self.regimen_flujo = None, None, None, None
        self.calcular_propiedades()

    def calcular_propiedades(self):
        """Orquesta el cálculo de todas las propiedades hidráulicas."""
        self.area = self.b * self.y
        self.perimetro_mojado = self.b + 2 * self.y
        if self.perimetro_mojado == 0: raise ValueError("El perímetro no puede ser cero.")
        self.radio_hidraulico = self.area / self.perimetro_mojado
        self.velocidad = (1 / self.n) * (self.radio_hidraulico ** (2 / 3)) * (self.S ** 0.5)
        self.caudal = self.area * self.velocidad
        if self.y <= 0: raise ValueError("El tirante debe ser un valor positivo.")
        self.numero_froude = self.velocidad / math.sqrt(self.g * self.y)
        if self.numero_froude < 1:
            self.regimen_flujo = "Subcrítico"
        elif self.numero_froude == 1:
            self.regimen_flujo = "Crítico"
        else:
            self.regimen_flujo = "Supercrítico"
        logging.info(f"Propiedades calculadas para {self.nombre}.")


# ==============================================================================
# CLASE PRINCIPAL PARA DISEÑO INVERSO
# ==============================================================================
class DisenadorInversoSME:
    """
    Diseña un prototipo SME a partir de las restricciones de un modelo de laboratorio.
    """

    def __init__(self, Q_m_obj, S, escala_L_obj):
        self.Q_m_obj = Q_m_obj
        self.S = S
        self.escala_longitud = escala_L_obj
        self.prototipo = None
        self.modelo = None
        logging.info(f"Diseñador Inverso creado con Q_m={Q_m_obj} m³/s, S={S}, Lambda_L={escala_L_obj}")

    def disenar_desde_modelo(self, b_m_fijo):
        """
        Calcula las especificaciones del modelo y del prototipo.
        """
        logging.info(f"Iniciando diseño inverso desde el modelo con b_m fijo = {b_m_fijo} m.")

        # 1. Definir la geometría del modelo basada en SME.
        # Si el prototipo es SME, el modelo también debe serlo.
        b_m = b_m_fijo
        y_m = b_m / 2.0
        logging.info(f"Geometría del modelo SME: b_m={b_m:.4f} m, y_m={y_m:.4f} m.")

        area_m = b_m * y_m
        radio_hid_m = y_m / 2.0

        # 2. Calcular la rugosidad 'n_m' requerida para el modelo.
        # Usamos la ecuación de Manning para despejar 'n'
        if self.Q_m_obj <= 0: raise ValueError("El caudal objetivo del modelo debe ser positivo.")
        n_m = (area_m * (radio_hid_m ** (2 / 3)) * (self.S ** 0.5)) / self.Q_m_obj
        logging.info(f"Rugosidad requerida para el modelo (n_m) = {n_m:.5f}")

        # Crear la instancia del modelo para validación.
        self.modelo = CanalRectangular(b_m, y_m, n_m, self.S, nombre="Modelo")

        # 3. Calcular las especificaciones del prototipo usando la escala.
        logging.info("Calculando especificaciones del prototipo real (SME)...")
        b_p = b_m * self.escala_longitud
        y_p = y_m * self.escala_longitud

        # La rugosidad del prototipo es un RESULTADO del diseño, no un dato de entrada.
        n_p = n_m * (self.escala_longitud ** (1 / 6))
        logging.info(f"Rugosidad resultante del prototipo (n_p) = {n_p:.5f}")

        # Crear la instancia del prototipo.
        self.prototipo = CanalRectangular(b_p, y_p, n_p, self.S, nombre="Prototipo (SME)")

    def mostrar_resumen_comparativo(self):
        """Imprime una tabla comparativa entre el prototipo y el modelo."""
        if not self.prototipo or not self.modelo:
            print("El diseño no ha sido ejecutado.")
            return

        print("\n" + "=" * 70)
        print("    RESUMEN DE DISEÑO INVERSO: PROTOTIPO (SME) Y MODELO FIJO")
        print("=" * 70)
        print(f"{'Parámetro':<25} | {'Prototipo (Real)':^20} | {'Modelo (Laboratorio)':^20}")
        print("-" * 70)
        print(f"{'Ancho de Solera (b)':<25} | {self.prototipo.b:^20.5f} | {self.modelo.b:^20.5f} m")
        print(f"{'Tirante (y)':<25} | {self.prototipo.y:^20.5f} | {self.modelo.y:^20.5f} m")
        print(f"{'Rugosidad (n)':<25} | {self.prototipo.n:^20.5f} | {self.modelo.n:^20.5f} (calculada)")
        print(f"{'Pendiente (S)':<25} | {self.prototipo.S:^20.5f} | {self.modelo.S:^20.5f}")
        print(f"{'Caudal (Q)':<25} | {self.prototipo.caudal:^20.5f} | {self.modelo.caudal:^20.5f} m^3/s")
        print(
            f"{'Número de Froude (Fr)':<25} | {self.prototipo.numero_froude:^20.4f} | {self.modelo.numero_froude:^20.4f}")
        print(f"{'Régimen de Flujo':<25} | {self.prototipo.regimen_flujo:^20} | {self.modelo.regimen_flujo:^20}")
        print("-" * 70)
        print(f"Escala de Longitud (Lambda_L): {self.escala_longitud:.2f} (definida)")
        self.verificar_similitud()
        print("=" * 70)

    def verificar_similitud(self):
        """Verifica si los números de Froude son consistentes."""
        fr_p = self.prototipo.numero_froude
        fr_m = self.modelo.numero_froude
        if abs(fr_p - fr_m) < 1e-4:
            print(f"\n[VALIDACIÓN EXITOSA]: Los Números de Froude coinciden ({fr_p:.4f} aprox. {fr_m:.4f}).")
            logging.info("La similitud dinámica de Froude se ha verificado exitosamente.")
        else:
            print(f"\n[ADVERTENCIA]: Discrepancia en Números de Froude ({fr_p:.4f} != {fr_m:.4f}).")


# ==============================================================================
# EJECUCIÓN DEL SCRIPT
# ==============================================================================
if __name__ == "__main__":
    logging.info("Inicio del script de diseño inverso de canal SME.")

    # --- PARÁMETROS FIJOS DEL MODELO Y DEL PROYECTO ---
    CAUDAL_MODELO_OBJETIVO = 0.001  # Caudal fijo del modelo (1 L/s)
    ANCHO_MODELO_FIJO = 0.200  # Ancho fijo del canal del laboratorio (m)
    PENDIENTE = 0.00373  # Pendiente del sistema (m/m)
    ESCALA_OBJETIVO = 15.0  # Escala de longitud deseada (ej. 15:1)

    try:
        # 1. Crear una instancia del diseñador inverso.
        disenador = DisenadorInversoSME(
            Q_m_obj=CAUDAL_MODELO_OBJETIVO,
            S=PENDIENTE,
            escala_L_obj=ESCALA_OBJETIVO
        )

        # 2. Diseñar el sistema partiendo de las restricciones del modelo.
        disenador.disenar_desde_modelo(ANCHO_MODELO_FIJO)

        # 3. Mostrar el resumen completo y la validación.
        disenador.mostrar_resumen_comparativo()

    except (ValueError, Exception) as e:
        logging.critical(f"Ha ocurrido un error crítico durante el diseño: {e}")
        print(f"\nERROR: {e}")

    logging.info("Fin del script.")

