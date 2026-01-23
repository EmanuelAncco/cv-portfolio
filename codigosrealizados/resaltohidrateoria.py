# =========================================================================
# CALCULADORA DIDÁCTICA DE RESALTO HIDRÁULICO
# Implementación con Python, Streamlit y SciPy
#
# OBJETIVO:
# Este script no solo calcula los parámetros de un resalto hidráulico,
# sino que también sirve como una guía de enseñanza paso a paso,
# integrando los conceptos teóricos fundamentales (basados en el
# documento "Resalto Hidráulico: Enseñanza y Aplicaciones").
# =========================================================================

# --- PASO 1: Importar las Bibliotecas Necesarias ---
# Importamos las herramientas que nos permitirán realizar los cálculos
# numéricos, crear la interfaz web y graficar los resultados.
# (Ref: Documento PDF, Sección 11.2 - Implementación con Python)

import streamlit as st  # Framework para crear la aplicación web interactiva.
import numpy as np  # Biblioteca para cálculos numéricos eficientes.
import matplotlib.pyplot as plt  # Biblioteca para crear gráficos estáticos.
from scipy.optimize import fsolve  # Función para resolver ecuaciones numéricamente.

# --- PASO 2: Definir Constantes Físicas ---
g = 9.81  # Aceleración de la gravedad (m/s^2)


# =========================================================================
# --- PASO 3: Definir las Funciones Geométricas ---
#
# Para analizar el resalto, necesitamos la "Fuerza Específica" (M).
# La fórmula de M depende de la geometría del canal (Área A y Centroide y_c).
# Estas funciones calculan dichas propiedades para un canal trapezoidal.
# =========================================================================

def area_trap(y, b, z):
    """
    Calcula el área (A) de una sección trapezoidal.
    A = (b + z*y) * y
    """
    return (b + z * y) * y


def centroide_trap(y, b, z):
    """
    Calcula la distancia desde la BASE del canal al centroide (y_c)
    de la sección trapezoidal.

    NOTA TEÓRICA:
    La Fuerza Específica (M) se define como: M = (Q^2 / gA) + A * y_c
    (Ref: Documento PDF, Sec 2.1 y Sec 6).

    Es crucial que y_c se mida desde la BASE. La fórmula estándar es:
    y_c = (y / 3) * ((2*b + z*y) / (b + z*y))

    (El código anterior tenía comentarios que discutían una posible
    ambigüedad en la definición de M. Esta implementación utiliza la
    definición estándar de la ingeniería hidráulica, consistente con
    los ejemplos del PDF).
    """
    if y == 0:
        return 0
    denominador = (b + z * y)
    if denominador == 0:
        return 0

    # Fórmula estándar del centroide (y_c) medido desde la base.
    y_c_base = (y / 3) * ((2 * b + z * y) / (b + z * y))
    return y_c_base


# =========================================================================
# --- PASO 4: Definir la Ecuación Gobernante (Fuerza Específica) ---
#
# PRINCIPIO FÍSICO CLAVE (Ref: Documento PDF, Sec 2.1):
# La energía (E) NO se conserva en un resalto debido a la turbulencia.
# El principio que SÍ se conserva es el MOMENTO.
#
# La "Fuerza Específica" (M) es la suma de la fuerza de presión
# y el flujo de momento por unidad de peso.
#
# M = (Q^2 / gA) + A * y_c
#
# La ecuación gobernante del resalto hidráulico es:
# M_1 = M_2
#
# (Fuerza Específica en Sección 1 = Fuerza Específica en Sección 2)
# =========================================================================

def fuerza_especifica_trap(y, Q, b, z):
    """
    Calcula la Fuerza Específica (M) para un tirante (y) dado.
    Implementa la Ecuación Gobernante.
    """
    if y <= 0:
        return np.inf  # Evitar división por cero o valores no físicos

    A = area_trap(y, b, z)
    if A == 0:
        return np.inf

    y_c = centroide_trap(y, b, z)  # Distancia de la base al centroide

    # M = (Flujo de Momento) + (Fuerza de Presión)
    return (Q ** 2 / (g * A)) + (A * y_c)


# =========================================================================
# --- PASO 5: Función Principal de Cálculo del Resalto ---
#
# Aquí se orquesta todo el análisis, siguiendo la teoría.
# =========================================================================

def calcular_resalto_avanzado(Q, b, y1, tipo_canal, z=0):
    """Calcula todos los parámetros de un resalto hidráulico."""

    # --- 5.1: Análisis de Condiciones Aguas Arriba (Sección 1) ---
    st.write(f"PASO 5.1: Analizando Sección 1 (y1 = {y1:.3f} m)...")

    if tipo_canal == "Rectangular":
        A1 = b * y1
        T1 = b
    else:  # Trapezoidal
        A1 = area_trap(y1, b, z)
        T1 = b + 2 * z * y1

    if A1 == 0:
        raise ValueError("El área de flujo inicial (A1) no puede ser cero.")

    V1 = Q / A1

    # CONCEPTO TEÓRICO: NÚMERO DE FROUDE (Fr)
    # (Ref: Documento PDF, Sec 3.1)
    # Fr = V / sqrt(g * D) -> Relaciona fuerzas de inercia y gravedad.
    # D = Profundidad Hidráulica = A / T
    # Fr > 1: Flujo Supercrítico (Rápido, Torrencial)
    # Fr < 1: Flujo Subcrítico (Lento, Tranquilo)
    #
    # Un resalto es SIEMPRE una transición de Fr > 1 a Fr < 1.
    D1 = A1 / T1
    Fr1 = V1 / np.sqrt(g * D1)

    # CONCEPTO TEÓRICO: ENERGÍA ESPECÍFICA (E)
    # (Ref: Documento PDF, Sec 2.2)
    # E = y + V^2 / (2g)
    E1 = y1 + V1 ** 2 / (2 * g)

    st.write(f"-> Fr1 = {Fr1:.3f}. Verificando régimen...")

    if Fr1 <= 1.0:
        raise ValueError(f"El flujo de entrada debe ser supercrítico (Fr > 1). Fr1 calculado: {Fr1:.2f}")

    st.write("-> Flujo Supercrítico (Fr > 1) confirmado. El resalto es posible.")

    # --- 5.2: Cálculo del Tirante Conjugado (y2) ---
    #
    # (Ref: Documento PDF, Sec 3.2 y 6)
    # Buscamos y2 tal que M(y1) = M(y2).
    # y1 e y2 se llaman "tirantes conjugados".
    st.write("PASO 5.2: Calculando tirante conjugado (y2)...")

    if tipo_canal == "Rectangular":
        # MÉTODO ANALÍTICO: Ecuación de Bélanger
        # (Ref: Documento PDF, Sec 3.2, Ecuación 7)
        # Es la solución directa de M1=M2 para un canal rectangular.
        st.write("-> Usando Ecuación de Bélanger (analítica) para canal rectangular.")
        y2 = (y1 / 2) * (np.sqrt(1 + 8 * Fr1 ** 2) - 1)

    else:  # Trapezoidal
        # MÉTODO NUMÉRICO: Solución de Ecuaciones
        # (Ref: Documento PDF, Sec 6)
        # No existe una "Ecuación de Bélanger" para trapecios.
        # Debemos resolver numéricamente M(y2) - M1 = 0.
        st.write("-> Usando Método Numérico (fsolve) para canal trapezoidal.")

        # Calculamos la Fuerza Específica en la Sección 1.
        M1 = fuerza_especifica_trap(y1, Q, b, z)

        # Creamos la función que fsolve debe resolver (buscar su raíz).
        # func(y2) = M(y2) - M1 = 0
        func_a_resolver = lambda y2_guess: fuerza_especifica_trap(y2_guess, Q, b, z) - M1

        # Proporcionamos una estimación inicial (la solución rectangular
        # suele ser una buena aproximación).
        y2_rect_guess = (y1 / 2) * (np.sqrt(1 + 8 * Fr1 ** 2) - 1)

        # fsolve (Find-Solve) encuentra el valor de 'y2' que hace
        # que 'func_a_resolver' sea igual a cero.
        solucion = fsolve(func_a_resolver, x0=y2_rect_guess * 1.1, full_output=True)

        if not solucion[2] == 1:  # 'ier' no es 1, fsolve falló
            raise RuntimeError(f"El solucionador numérico (fsolve) no pudo converger. Mensaje: {solucion[3]}")

        y2 = solucion[0][0]  # El resultado es un array

    st.write(f"-> Tirante Conjugado y2 = {y2:.3f} m.")

    # --- 5.3: Análisis de Condiciones Aguas Abajo (Sección 2) ---
    st.write("PASO 5.3: Analizando Sección 2 (aguas abajo)...")
    if tipo_canal == "Rectangular":
        A2 = b * y2
        T2 = b
    else:  # Trapezoidal
        A2 = area_trap(y2, b, z)
        T2 = b + 2 * z * y2

    V2 = Q / A2
    D2 = A2 / T2
    Fr2 = V2 / np.sqrt(g * D2)
    E2 = y2 + V2 ** 2 / (2 * g)
    st.write(f"-> Fr2 = {Fr2:.3f}. (Debe ser < 1, confirmando flujo Subcrítico).")

    # --- 5.4: Cálculo de Parámetros del Resalto ---
    st.write("PASO 5.4: Cuantificando el resalto...")

    # CONCEPTO TEÓRICO: PÉRDIDA DE ENERGÍA (Disipación)
    # (Ref: Documento PDF, Sec 2.2)
    # ¡La aplicación principal del resalto!
    # La turbulencia disipa energía (calor, sonido).
    # delta_E = E1 - E2
    delta_E = E1 - E2

    # CONCEPTO TEÓRICO: LONGITUD DEL RESALTO (Lr)
    # (Ref: Documento PDF, Sec 5.2)
    # La longitud NO es teórica, se determina con FÓRMULAS EMPÍRICAS
    # (basadas en experimentos de laboratorio).
    # Usamos la aproximación común del USBR.
    Lr_usbr = 6.1 * y2 if tipo_canal == "Rectangular" else None  # USBR es para rectangulares

    # CONCEPTO TEÓRICO: CLASIFICACIÓN DEL RESALTO
    # (Ref: Documento PDF, Sec 4, Tabla 1)
    # El comportamiento del resalto (estabilidad, disipación)
    # depende directamente del Fr1.
    st.write("-> Clasificando el resalto según Fr1...")
    if 1 < Fr1 <= 1.7:
        tipo, nota = "Ondular", "Disipación de energía muy baja."
    elif Fr1 <= 2.5:
        tipo, nota = "Débil", "Baja disipación de energía."
    elif Fr1 <= 4.5:
        tipo, nota = "Oscilante", "ADVERTENCIA: Inestable, genera olas."
    elif Fr1 <= 9.0:
        tipo, nota = "Estable", "IDEAL PARA DISEÑO: Eficiente y predecible."
    else:
        tipo, nota = "Fuerte", "Muy alta disipación, pero muy turbulento."

    st.write("--- CÁLCULO COMPLETADO ---")

    # Devolvemos un "diccionario" con todos los resultados organizados.
    return {
        "y1": y1, "V1": V1, "Fr1": Fr1, "E1": E1,
        "y2": y2, "V2": V2, "Fr2": Fr2, "E2": E2,
        "delta_E": delta_E, "Lr_usbr": Lr_usbr,
        "tipo": tipo, "nota": nota
    }


# =========================================================================
# --- PASO 6: Interfaz de Usuario de Streamlit ---
#
# (Ref: Documento PDF, Sec 11)
# Esta sección construye la interfaz web.
# No contiene teoría hidráulica, solo organiza las entradas y salidas.
# =========================================================================

# Configuración de la página
st.set_page_config(page_title="Calculadora Didáctica de Resalto", layout="wide")
st.title("🛠️ Calculadora Didáctica de Resalto Hidráulico")
st.markdown(
    "Esta herramienta calcula y explica (paso a paso) el resalto hidráulico, integrando la teoría de hidráulica de canales.")

# --- Panel Lateral de Entradas ---
with st.sidebar:
    st.header("Parámetros de Entrada (Sección 1)")

    tipo_canal_input = st.selectbox("Tipo de Canal", ["Rectangular", "Trapezoidal"])

    Q_input = st.number_input("Caudal (Q) en m³/s:", min_value=0.1, value=18.0)
    b_input = st.number_input("Ancho de Solera (b) en m:", min_value=0.1, value=3.0)
    y1_input = st.number_input("Tirante aguas arriba (y1) en m:", min_value=0.01, value=1.0)

    z_input = 0.0  # Inicializar z_input
    if tipo_canal_input == "Trapezoidal":
        z_input = st.number_input("Talud (z) (1V:zH):", min_value=0.0, value=1.0, step=0.5)

# --- Lógica Principal de la Aplicación ---
if st.sidebar.button("Analizar Resalto"):

    # Creamos un contenedor para los pasos del cálculo
    log_container = st.expander("Ver Proceso de Cálculo (Paso a Paso)", expanded=False)

    with log_container:
        try:
            # ----------------------------------------------------
            # --- ¡AQUÍ SE EJECUTA TODA LA TEORÍA (PASO 5)! ---
            res = calcular_resalto_avanzado(Q_input, b_input, y1_input, tipo_canal_input, z_input)
            # ----------------------------------------------------

            st.header("📊 Resultados del Análisis")

            # --- 6.1: Mostrar Resultados Clave ---
            col_tipo, col_disip, col_lr = st.columns(3)
            col_tipo.metric("Tipo de Resalto (Sec 4)", res["tipo"])
            col_disip.metric("Disipación de Energía (ΔE) (Sec 2.2)", f"{res['delta_E']:.3f} m")

            lr_display = f"{res['Lr_usbr']:.2f} m (USBR)" if res['Lr_usbr'] else "N/A (Solo Rect.)"
            col_lr.metric("Longitud (Lr) (Empírica) (Sec 5.2)", lr_display)

            st.warning(f"**Nota (Ref: Sec 4, Tabla 1):** {res['nota']}")

            st.subheader("Detalle de Secciones Conjugadas")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Aguas Arriba (Sección 1)")
                st.dataframe({
                    "Parámetro": ["Tirante (y1)", "Velocidad (V1)", "Nº Froude (Fr1)", "Energía (E1)"],
                    "Valor": [f"{res['y1']:.3f} m", f"{res['V1']:.3f} m/s", f"{res['Fr1']:.3f}", f"{res['E1']:.3f} m"]
                })

            with col2:
                st.markdown("#### Aguas Abajo (Sección 2)")
                st.dataframe({
                    "Parámetro": ["Tirante Conjugado (y2)", "Velocidad (V2)", "Nº Froude (Fr2)", "Energía (E2)"],
                    "Valor": [f"{res['y2']:.3f} m", f"{res['V2']:.3f} m/s", f"{res['Fr2']:.3f}", f"{res['E2']:.3f} m"]
                })

            # --- 6.2: Visualización Gráfica ---
            # (Ref: Documento PDF, Sec 2.2 y 11.1)
            # Graficamos la Curva E-y para visualizar la pérdida de energía.

            st.subheader("Visualización: Curva de Energía Específica (E-y)")

            # Calculamos el rango del gráfico
            y_min_critico = (Q_input ** 2 / (g * b_input ** 2)) ** (1 / 3) if tipo_canal_input == "Rectangular" else \
            res['y1']  # Simplificación
            y_vals = np.linspace(y_min_critico * 0.5, res['y2'] * 1.5, 200)

            # Calculamos E para cada y
            if tipo_canal_input == "Rectangular":
                A_vals = b_input * y_vals
                E_vals = y_vals + (Q_input ** 2) / (2 * g * A_vals ** 2)
            else:
                E_vals = [y + (Q_input ** 2) / (2 * g * area_trap(y, b_input, z_input) ** 2) for y in y_vals]

            fig, ax = plt.subplots()

            # 1. Dibujar la curva E-y
            ax.plot(E_vals, y_vals, label="Curva E-y (E = y + V²/2g)")

            # 2. Marcar los puntos del resalto
            # (E1, y1) -> Supercrítico
            # (E2, y2) -> Subcrítico
            ax.plot(res['E1'], res['y1'], 'ro',
                    label=f"Sección 1 (Supercrítica)\ny1={res['y1']:.3f}, E1={res['E1']:.3f}")
            ax.plot(res['E2'], res['y2'], 'bo', label=f"Sección 2 (Subcrítica)\ny2={res['y2']:.3f}, E2={res['E2']:.3f}")

            # 3. Dibujar la pérdida de energía (la flecha horizontal)
            ax.plot([res['E1'], res['E2']], [res['y1'], res['y1']], 'r--',
                    label=f"Pérdida de Energía (ΔE) = {res['delta_E']:.3f} m")
            ax.annotate(
                "",
                xy=(res['E2'], res['y1']),
                xytext=(res['E1'], res['y1']),
                arrowprops=dict(arrowstyle="<->", color="red")
            )

            ax.axhline(res['y2'], color='blue', linestyle='--', linewidth=0.7)
            ax.axvline(res['E2'], color='blue', linestyle='--', linewidth=0.7)

            ax.set_xlabel("Energía Específica (E) [m]")
            ax.set_ylabel("Tirante (y) [m]")
            ax.set_title("Visualización de la Pérdida de Energía en el Resalto")
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)

            st.pyplot(fig)


        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
else:
    st.info("Configure los parámetros en la barra lateral y presione 'Analizar Resalto' para ejecutar el cálculo.")

