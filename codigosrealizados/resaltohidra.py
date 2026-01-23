# =========================================================================
# CALCULADORA AVANZADA DE RESALTO HIDRÁULICO
# Implementación con Python, Streamlit y SciPy
# =========================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- Constantes ---
g = 9.81  # Aceleración de la gravedad (m/s^2)


# --- Funciones de Cálculo para Geometría Trapezoidal ---
def area_trap(y, b, z):
    """Calcula el área de una sección trapezoidal."""
    return (b + z * y) * y


def centroide_trap(y, b, z):
    """Calcula el centroide de una sección trapezoidal desde la superficie."""
    if y == 0: return 0
    # Fórmula corregida para el centroide desde el fondo, luego se ajusta.
    # Esta fórmula parece ser la distancia desde el fondo.
    # Para la Fuerza Específica, necesitamos la distancia del centroide a la superficie libre.
    # Sin embargo, la fórmula de Fuerza Específica M = (Q^2 / gA) + A * y_bar
    # usa y_bar como la distancia desde la superficie libre al centroide del área.
    # La fórmula estándar para el centroide (y_c) desde la base es:
    y_c_base = (y / 3) * ((2 * b + z * y) / (b + z * y))

    # La Fuerza Específica se define a menudo con y_bar como la distancia
    # desde la *superficie libre* al centroide del área.
    # Pero si y_bar es la distancia desde la *base* al centroide, la Ecuación de Momento es diferente.

    # Asumamos que la fórmula de Fuerza Específica implementada
    # M = (Q^2 / gA) + A * y_bar
    # usa y_bar como la distancia desde la *base* al centroide.
    # Vamos a verificar la implementación de fuerza_especifica_trap.

    # Si y_bar es la distancia desde la superficie al centroide:
    # y_bar = y - y_c_base
    # y_bar = y - (y / 3) * ((2 * b + z * y) / (b + z * y))
    # y_bar = (y/3) * ( (3*(b+zy)) - (2b+zy) ) / (b+zy)
    # y_bar = (y/3) * ( b + 2*zy ) / (b+zy)

    # La fórmula que tenías: (y / 3) * ((3 * b + 2 * z * y) / (b + z * y))
    # parece una mezcla.

    # Usemos la fórmula estándar del centroide medido desde la BASE:
    if (b + z * y) == 0: return 0
    y_c_base = (y / 3) * ((2 * b + z * y) / (b + z * y))
    return y_c_base

    # Si la definición de Fuerza Específica (M) usa y_bar como la distancia
    # desde la SUPERFICIE al centroide:
    # A = (b + z*y) * y
    # T = b + 2*z*y
    # D = A/T = (b+zy)y / (b+2zy)
    # y_bar_superficie = (y/3) * ( (b+2*z*y) / (b+z*y) ) * ( (b+z*y)*y / (b+2*z*y) ) # Esto es D * (algo)

    # Volvamos a la definición estándar de M (Fuerza Específica o Función de Momento):
    # M = A * y_c + Q^2 / (g * A)
    # donde y_c es la distancia desde la base al centroide del área.
    # Esta es la implementación correcta.

    # Tu fórmula original: (y / 3) * ((3 * b + 2 * z * y) / (b + z * y))
    # No es la estándar para el centroide desde la base.
    # La estándar es: (y / 3) * ((2 * b + z * y) / (b + z * y))
    # Voy a usar la estándar.

    if (b + z * y) == 0: return 0
    return (y / 3) * ((2 * b + z * y) / (b + z * y))


def fuerza_especifica_trap(y, Q, b, z):
    """Calcula la Fuerza Específica (Función de Momento) para un trapecio."""
    if y <= 0:
        return np.inf  # Evitar división por cero o valores no físicos
    A = area_trap(y, b, z)
    if A == 0:
        return np.inf
    y_c = centroide_trap(y, b, z)  # Distancia de la base al centroide
    return (Q ** 2 / (g * A)) + A * y_c


# --- Función Principal de Cálculo ---
def calcular_resalto_avanzado(Q, b, y1, tipo_canal, z=0):
    """Calcula todos los parámetros de un resalto hidráulico."""

    # --- Condiciones Aguas Arriba (Sección 1) ---
    if tipo_canal == "Rectangular":
        A1 = b * y1
        T1 = b
    else:  # Trapezoidal
        A1 = area_trap(y1, b, z)
        T1 = b + 2 * z * y1

    if A1 == 0:
        raise ValueError("El área de flujo inicial (A1) no puede ser cero.")

    V1 = Q / A1
    D1 = A1 / T1  # Profundidad hidráulica
    Fr1 = V1 / np.sqrt(g * D1)
    E1 = y1 + V1 ** 2 / (2 * g)

    if Fr1 <= 1.0:
        raise ValueError(f"El flujo de entrada debe ser supercrítico (Fr > 1). Fr1 calculado: {Fr1:.2f}")

    # --- Cálculo del Tirante Conjugado (y2) ---
    if tipo_canal == "Rectangular":
        y2 = (y1 / 2) * (np.sqrt(1 + 8 * Fr1 ** 2) - 1)
    else:  # Trapezoidal - Solución numérica
        M1 = fuerza_especifica_trap(y1, Q, b, z)

        # Ecuación a resolver: M(y2) - M1 = 0
        func_a_resolver = lambda y2_guess: fuerza_especifica_trap(y2_guess, Q, b, z) - M1

        # Usar una estimación inicial más robusta.
        # La solución de canal rectangular es una buena primera suposición.
        y2_rect_guess = (y1 / 2) * (np.sqrt(1 + 8 * Fr1 ** 2) - 1)

        # fsolve necesita una estimación inicial.
        # y1 * (Fr1**2) es una estimación muy alta, pero fsolve debería manejarla.
        # Usemos la estimación rectangular que es más física.
        solucion = fsolve(func_a_resolver, x0=y2_rect_guess * 1.1, full_output=True)

        if not solucion[2] == 1:  # 'ier' no es 1, fsolve falló
            raise RuntimeError(f"El solucionador numérico (fsolve) no pudo converger. Mensaje: {solucion[3]}")

        y2 = solucion[0][0]  # El resultado es un array

    # --- Condiciones Aguas Abajo (Sección 2) ---
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

    # --- Parámetros del Resalto ---
    delta_E = E1 - E2
    Lr_usbr = 6.1 * y2 if tipo_canal == "Rectangular" else None  # USBR es para rectangulares

    # Clasificación del resalto
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

    return {
        "y1": y1, "V1": V1, "Fr1": Fr1, "E1": E1,
        "y2": y2, "V2": V2, "Fr2": Fr2, "E2": E2,
        "delta_E": delta_E, "Lr_usbr": Lr_usbr,
        "tipo": tipo, "nota": nota
    }


# --- Interfaz de Usuario de Streamlit ---
st.set_page_config(page_title="Calculadora Avanzada de Resalto", layout="wide")
st.title("🛠️ Calculadora Avanzada de Resalto Hidráulico")
st.markdown(
    "Esta herramienta calcula las propiedades del resalto hidráulico para canales **rectangulares** y **trapezoidales**.")

with st.sidebar:
    st.header("Parámetros de Entrada")

    # --- CORRECCIÓN ---
    # Se añade la lista de opciones obligatoria para st.selectbox
    tipo_canal_input = st.selectbox("Tipo de Canal", ["Rectangular", "Trapezoidal"])

    Q_input = st.number_input("Caudal (Q) en m³/s:", min_value=0.1, value=18.0)
    b_input = st.number_input("Ancho de Solera (b) en m:", min_value=0.1, value=3.0)
    y1_input = st.number_input("Tirante aguas arriba (y1) en m:", min_value=0.01, value=1.0)

    z_input = 0.0  # Inicializar z_input
    if tipo_canal_input == "Trapezoidal":
        z_input = st.number_input("Talud (z) (1V:zH):", min_value=0.0, value=1.0, step=0.5)

if st.sidebar.button("Analizar Resalto"):
    try:
        res = calcular_resalto_avanzado(Q_input, b_input, y1_input, tipo_canal_input, z_input)

        st.header("📊 Resultados del Análisis")

        col_tipo, col_disip, col_lr = st.columns(3)
        col_tipo.metric("Tipo de Resalto", res["tipo"])
        col_disip.metric("Disipación de Energía (ΔE)", f"{res['delta_E']:.3f} m")

        lr_display = f"{res['Lr_usbr']:.2f} m (USBR)" if res['Lr_usbr'] else "N/A (Solo Rect.)"
        col_lr.metric("Longitud (Lr)", lr_display)

        st.warning(f"**Nota:** {res['nota']}")

        st.subheader("Detalle de Secciones")

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

        # --- Visualización Gráfica ---
        st.subheader("Perfil del Resalto y Energía")

        # 1. Gráfico de Energía Específica vs Tirante
        y_min_critico = (Q_input ** 2 / (g * b_input ** 2)) ** (1 / 3) if tipo_canal_input == "Rectangular" else res[
            'y1']  # Simplificación
        y_vals = np.linspace(y_min_critico * 0.5, res['y2'] * 1.5, 200)

        if tipo_canal_input == "Rectangular":
            A_vals = b_input * y_vals
            E_vals = y_vals + (Q_input ** 2) / (2 * g * A_vals ** 2)
        else:
            E_vals = [y + (Q_input ** 2) / (2 * g * area_trap(y, b_input, z_input) ** 2) for y in y_vals]

        fig, ax = plt.subplots()
        ax.plot(E_vals, y_vals, label="Energía Específica (E)")

        # Puntos del resalto
        ax.plot([res['E1'], res['E2']], [res['y1'], res['y1']], 'r--', label=f"ΔE = {res['delta_E']:.3f} m")
        ax.plot(res['E1'], res['y1'], 'ro', label=f"Sección 1 (y1={res['y1']:.3f}, E1={res['E1']:.3f})")
        ax.plot(res['E2'], res['y2'], 'bo', label=f"Sección 2 (y2={res['y2']:.3f}, E2={res['E2']:.3f})")
        ax.axhline(res['y2'], color='blue', linestyle='--', linewidth=0.7)
        ax.axvline(res['E2'], color='blue', linestyle='--', linewidth=0.7)

        ax.set_xlabel("Energía Específica (E) [m]")
        ax.set_ylabel("Tirante (y) [m]")
        ax.set_title("Curva E-y y Disipación de Energía")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

        st.pyplot(fig)


    except Exception as e:
        st.error(f"Error en el cálculo: {e}")
else:
    st.info("Configure los parámetros en la barra lateral y presione 'Analizar Resalto' para ejecutar el cálculo.")
