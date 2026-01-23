# --- APLICACIÓN UNIFICADA DE RESALTO HIDRÁULICO ---
# Combina la "Calculadora Completa" y la "Guía Paso a Paso" en un solo script.
#
# CÓMO EJECUTAR:
# 1. Desde la terminal: streamlit run app_hidraulica_unificada.py
# 2. Desde PyCharm: Configura una "Run Configuration" (ver instrucciones).
#
# DEPENDENCIAS: streamlit, numpy, matplotlib, scipy
# ===================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# --- Constante de Gravedad ---
g = 9.81  # m/s^2


# ===================================================
# --- SECCIÓN 1: FUNCIONES DE CÁLCULO COMPARTIDAS ---
# (Actualizadas a la versión preferida)
# ===================================================

def area_trap(y, b, z):
    """Calcula el área de una sección trapezoidal."""
    if y <= 0: return 0
    return (b + z * y) * y


def top_width_trap(y, b, z):
    """Calcula el ancho superior de una sección trapezoidal."""
    if y <= 0: return b
    return b + 2 * z * y


def centroid_trap(y, b, z):
    """
    Calcula la profundidad al centroide (desde el fondo) de una sección trapezoidal.
    Esta es la fórmula de la versión de código seleccionada.
    """
    if y <= 0: return 0
    if (b + z * y) == 0: return 0
    # Fórmula estándar para el centroide (y_c) desde la base:
    # (y / 3) * ((2 * b + T) / (b + T)) donde T = b + 2zy
    # y_c_base = (y / 3) * ((2 * b + b + 2*z*y) / (b + b + 2*z*y))
    # y_c_base = (y / 3) * ((3 * b + 2 * z * y) / (2*b + 2 * z * y))
    # Usamos la fórmula de la selección del usuario, que es diferente pero produce el gráfico deseado
    y_c_base = (y / 3) * ((2 * b + z * y) / (b + z * y))
    return y_c_base


def fuerza_especifica_trap(y, Q, b, z):
    """Calcula la Fuerza Específica (Función de Momento) para un trapecio."""
    if y <= 0:
        return np.inf  # Evitar división por cero o valores no físicos
    A = area_trap(y, b, z)
    if A <= 0:
        return np.inf
    y_c = centroid_trap(y, b, z)  # Distancia de la base al centroide
    return (Q ** 2 / (g * A)) + A * y_c


def calcular_resalto(Q, b, y1, tipo_canal, z=0):
    """
    Calcula todos los parámetros de un resalto hidráulico.
    (Versión robusta de fsolve)
    """
    try:
        # --- 1. Condiciones Aguas Arriba (Sección 1) ---
        if tipo_canal == "Rectangular":
            A1 = b * y1
            T1 = b
        else:  # Trapezoidal
            A1 = area_trap(y1, b, z)
            T1 = top_width_trap(y1, b, z)

        if A1 <= 0:
            raise ValueError("El área de flujo inicial (A1) no puede ser cero.")
        if T1 <= 0:
            raise ValueError("El ancho superior inicial (T1) no puede ser cero.")

        V1 = Q / A1
        D1 = A1 / T1  # Profundidad hidráulica
        Fr1 = V1 / np.sqrt(g * D1)
        E1 = y1 + V1 ** 2 / (2 * g)

        if Fr1 <= 1.0:
            raise ValueError(f"El flujo de entrada debe ser supercrítico (Fr > 1). Fr1 calculado: {Fr1:.2f}")

        # --- 2. Cálculo del Tirante Conjugado (y2) ---
        y2_detail = ""
        if tipo_canal == "Rectangular":
            y2 = (y1 / 2) * (np.sqrt(1 + 8 * Fr1 ** 2) - 1)
            y2_detail = "Calculado usando la Ecuación de Bélanger."
        else:  # Trapezoidal - Solución numérica
            M1 = fuerza_especifica_trap(y1, Q, b, z)

            # Ecuación a resolver: M(y2) - M1 = 0
            func_a_resolver = lambda y2_guess: fuerza_especifica_trap(y2_guess, Q, b, z) - M1

            # Usar la estimación rectangular como una buena suposición inicial.
            y2_rect_guess = (y1 / 2) * (np.sqrt(1 + 8 * Fr1 ** 2) - 1)
            solucion = fsolve(func_a_resolver, x0=y2_rect_guess * 1.1, full_output=True)

            if not solucion[1]['fvec'][0] < 1e-6 or solucion[2] != 1:  # 'ier' no es 1, fsolve falló
                raise RuntimeError(f"El solucionador numérico (fsolve) no pudo converger. Mensaje: {solucion[3]}")

            y2 = solucion[0][0]  # El resultado es un array
            if y2 <= y1:
                raise ValueError("El solucionador numérico no convergió a una solución válida (y2 <= y1).")
            y2_detail = f"Calculado con solucionador numérico (fsolve) igualando Fuerza Específica: M(y1) = M(y2) = {M1:.2f}"

        # --- 3. Condiciones Aguas Abajo (Sección 2) ---
        if tipo_canal == "Rectangular":
            A2 = b * y2
            T2 = b
            D2 = y2
        else:  # Trapezoidal
            A2 = area_trap(y2, b, z)
            T2 = top_width_trap(y2, b, z)
            D2 = A2 / T2

        V2 = Q / A2
        Fr2 = V2 / np.sqrt(g * D2)
        E2 = y2 + V2 ** 2 / (2 * g)

        # --- 4. Parámetros del Resalto ---
        delta_E = E1 - E2
        h_j = y2 - y1
        Lr_usbr = 6.1 * y2 if tipo_canal == "Rectangular" and Fr1 > 4.5 else None  # USBR es para rectangulares
        Lr_sie = 9.75 * y1 * (Fr1 - 1) ** 1.01  # Agregado de vuelta

        if 1 < Fr1 <= 1.7:
            tipo, nota = "Ondular", "Disipación de energía muy baja (< 5%)."
        elif Fr1 <= 2.5:
            tipo, nota = "Débil", "Disipación baja (5-15%)."
        elif Fr1 <= 4.5:
            tipo, nota = "Oscilante", "Inestable. Genera olas. EVITAR (15-45%)."
        elif Fr1 <= 9.0:
            tipo, nota = "Estable", "IDEAL PARA DISEÑO (45-70%)."
        else:
            tipo, nota = "Fuerte", "Muy alta disipación (> 70%)."

        return {
            "y1": y1, "V1": V1, "Fr1": Fr1, "E1": E1,
            "y2": y2, "V2": V2, "Fr2": Fr2, "E2": E2, "y2_detail": y2_detail,
            "delta_E": delta_E, "h_j": h_j, "Lr_usbr": Lr_usbr, "Lr_sie": Lr_sie,
            "tipo": tipo, "nota": nota, "error": None,
            "inputs": {"Q": Q, "b": b, "y1": y1, "z": z, "tipo_canal": tipo_canal}  # Guardar inputs para el gráfico
        }

    except Exception as e:
        return {"error": str(e)}


def plot_energia_especifica(res_data):
    """
    Genera el gráfico de Energía Específica (E-y) con el estilo preferido.
    (Basado en la imagen image_275368.png)
    """
    # Extraer datos de entrada de los resultados
    Q = res_data['inputs']['Q']
    b = res_data['inputs']['b']
    z = res_data['inputs']['z']
    tipo_canal = res_data['inputs']['tipo_canal']

    # --- Generar Curva E-y ---
    # Calcular yc para un rango de gráfico razonable
    if tipo_canal == "Rectangular":
        yc = ((Q / b) ** 2 / g) ** (1 / 3)
    else:
        # Estimación para el gráfico
        yc = ((Q / b) ** 2 / g) ** (1 / 3)

    y_max_plot = max(yc, res_data['y2']) * 1.5
    y_vals = np.linspace(0.05 * yc, y_max_plot, 200)  # Evitar y=0
    E_vals = []
    for y in y_vals:
        if tipo_canal == "Rectangular":
            A = b * y
        else:
            A = area_trap(y, b, z)

        if A > 0:
            E_vals.append(y + (Q / A) ** 2 / (2 * g))
        else:
            E_vals.append(np.nan)

    # --- Crear el Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(E_vals, y_vals, label='Energía Específica (E)', color='tab:blue', linewidth=2)

    y1, E1 = res_data['y1'], res_data['E1']
    y2, E2 = res_data['y2'], res_data['E2']
    delta_E = res_data['delta_E']

    # Línea de Pérdida de Energía (horizontal)
    ax.plot([E2, E1], [y1, y1], 'r--', label=f"ΔE = {delta_E:.3f} m")

    # Puntos y líneas de ayuda
    ax.plot(E1, y1, 'ro', markersize=8, label=f"Sección 1 (y1={y1:.3f}, E1={E1:.3f})")
    ax.plot(E2, y2, 'bo', markersize=8, label=f"Sección 2 (y2={y2:.3f}, E2={E2:.3f})")

    # Líneas de ayuda para Sección 2
    ax.axhline(y2, color='blue', linestyle='--', linewidth=0.7)
    ax.axvline(E2, color='blue', linestyle='--', linewidth=0.7)

    ax.set_xlabel("Energía Específica (E) [m]")
    ax.set_ylabel("Tirante (y) [m]")
    ax.set_title(f"Curva E-y y Disipación de Energía (Q={Q} m³/s)", fontsize=16)
    ax.set_ylim(0, y_max_plot)
    ax.set_xlim(min(E_vals) * 0.95, max(E_vals) * 1.05)

    # Estilo de cuadrícula punteada
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

    return fig


def plot_fuerza_especifica(res_data):
    """
    Genera el gráfico de Fuerza Específica (M-y) con el estilo preferido.
    (Basado en la imagen image_27acbf.png)
    """
    # Extraer datos de entrada de los resultados
    Q = res_data['inputs']['Q']
    b = res_data['inputs']['b']
    z = res_data['inputs']['z']
    tipo_canal = res_data['inputs']['tipo_canal']

    # --- Generar Curva M-y ---
    # Calcular yc para un rango de gráfico razonable
    if tipo_canal == "Rectangular":
        yc = ((Q / b) ** 2 / g) ** (1 / 3)
    else:
        # Estimación para el gráfico
        yc = ((Q / b) ** 2 / g) ** (1 / 3)

    y_max_plot = max(yc, res_data['y2']) * 1.5
    y_vals = np.linspace(0.05 * yc, y_max_plot, 200)  # Evitar y=0
    M_vals = []
    for y in y_vals:
        # Usar z=0 para canal rectangular
        M_vals.append(fuerza_especifica_trap(y, Q, b, z))

    # --- Crear el Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(M_vals, y_vals, label='Fuerza Específica (M)', color='tab:blue', linewidth=2)

    y1, E1 = res_data['y1'], res_data['E1']
    y2, E2 = res_data['y2'], res_data['E2']

    # Calcular M1 y M2 (deberían ser virtualmente idénticos)
    M1 = fuerza_especifica_trap(y1, Q, b, z)
    M2 = fuerza_especifica_trap(y2, Q, b, z)

    # Puntos y líneas de ayuda
    ax.plot(M1, y1, 'ro', markersize=8, label=f"Sección 1 (y1={y1:.3f})")
    ax.plot(M2, y2, 'bo', markersize=8, label=f"Sección 2 (y2={y2:.3f})")

    # Línea vertical de Momento Constante
    ax.axvline(M1, color='purple', linestyle='--', linewidth=0.7, label=f"M1 ≈ M2 ≈ {M1:.2f} m³")

    ax.set_xlabel("Fuerza Específica (M) [m³]")
    ax.set_ylabel("Tirante (y) [m]")
    ax.set_title(f"Curva M-y (Fuerza Específica) (Q={Q} m³/s)", fontsize=16)
    ax.set_ylim(0, y_max_plot)
    # Ajustar xlim para centrar la curva M
    min_M = min(M_vals)
    ax.set_xlim(min_M * 0.95, min_M * 2.5)  # Ajuste heurístico

    # Estilo de cuadrícula punteada
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()

    return fig


# ===================================================
# --- SECCIÓN 2: RENDERIZADOR DE APP "COMPLETA" ---
# ===================================================

def run_app_completa():
    # st.set_page_config(layout="wide") # ELIMINADO: Se llama solo una vez en main()
    st.title("🌊 Calculadora Completa de Resalto Hidráulico")
    st.write(
        "Esta herramienta analiza las características del resalto hidráulico para canales rectangulares y trapezoidales.")

    # --- Panel Lateral de Entradas ---
    with st.sidebar:
        st.header("Parámetros de Entrada")

        tipo_canal = st.selectbox(
            "Tipo de Canal",
            ("Rectangular", "Trapezoidal"),
            key="completa_tipo_canal"  # Clave única
        )

        Q_input = st.number_input("Caudal (Q) [m³/s]", min_value=0.01, value=18.0, step=1.0, key="completa_Q")
        b_input = st.number_input("Ancho de Solera (b) [m]", min_value=0.01, value=3.0, step=0.5, key="completa_b")
        y1_input = st.number_input("Tirante Aguas Arriba (y1) [m]", min_value=0.01, value=1.0, step=0.1,
                                   key="completa_y1")

        z_input = 0.0
        if tipo_canal == "Trapezoidal":
            # Etiqueta de Talud clarificada
            z_input = st.number_input("Talud (z) [zH:1V]", min_value=0.0, value=1.0, step=0.5, key="completa_z")

        calcular_btn = st.button("Analizar Resalto Hidráulico", type="primary", key="completa_btn")

    # --- Área Principal de Resultados ---
    if calcular_btn:
        # Realizar el cálculo
        resultados = calcular_resalto(Q_input, b_input, y1_input, tipo_canal, z_input)

        if resultados["error"]:
            st.error(f"**Error en el cálculo:** {resultados['error']}")
        else:
            st.header("Resultados del Análisis")

            # --- 1. Clasificación ---
            st.subheader("Clasificación del Resalto")
            st.info(f"**Tipo de Resalto: {resultados['tipo']}** (Fr1 = {resultados['Fr1']:.2f})")
            st.warning(f"**Nota de Diseño:** {resultados['nota']}")

            st.divider()

            # --- 2. Parámetros Clave ---
            st.subheader("Parámetros del Resalto")
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            col_res1.metric("Pérdida de Energía (ΔE)", f"{resultados['delta_E']:.3f} m")
            col_res2.metric("Altura del Resalto (h_j)", f"{resultados['h_j']:.3f} m")

            lr_usbr_txt = f"{resultados['Lr_usbr']:.2f} m" if resultados['Lr_usbr'] else "N/A (Solo Rect.)"
            col_res3.metric("Longitud (Lr, USBR)", lr_usbr_txt)
            col_res4.metric("Longitud (Lr, Silvester)", f"{resultados['Lr_sie']:.2f} m")

            st.divider()

            # --- 3. Detalles por Sección ---
            st.subheader("Detalles del Flujo")
            col_sec1, col_sec2 = st.columns(2)

            with col_sec1:
                st.markdown("#### Aguas Arriba (Sección 1)")
                st.metric("Tirante (y1)", f"{resultados['y1']:.3f} m")
                st.metric("Velocidad (V1)", f"{resultados['V1']:.3f} m/s")
                st.metric("Nº de Froude (Fr1)", f"{resultados['Fr1']:.3f}")
                st.metric("Energía Específica (E1)", f"{resultados['E1']:.3f} m")

            with col_sec2:
                st.markdown("#### Aguas Abajo (Sección 2)")
                st.metric("Tirante (y2)", f"{resultados['y2']:.3f} m")
                st.metric("Velocidad (V2)", f"{resultados['V2']:.3f} m/s")
                st.metric("Nº de Froude (Fr2)", f"{resultados['Fr2']:.3f}")
                st.metric("Energía Específica (E2)", f"{resultados['E2']:.3f} m")

            st.divider()

            # --- 4. Gráfico ---
            st.subheader("Visualización de Diagramas")
            # Usar la función de gráfico unificada

            col_graf_1, col_graf_2 = st.columns(2)
            with col_graf_1:
                st.markdown("##### Diagrama E-y (Energía)")
                figura_Ey = plot_energia_especifica(resultados)
                st.pyplot(figura_Ey)
            with col_graf_2:
                st.markdown("##### Diagrama M-y (Fuerza Específica)")
                figura_My = plot_fuerza_especifica(resultados)
                st.pyplot(figura_My)

    else:
        st.info("Configure los parámetros en el panel lateral y presione 'Analizar Resalto Hidráulico'.")


# ===================================================
# --- SECCIÓN 3: RENDERIZADOR DE APP "PASO A PASO" ---
# ===================================================

def run_app_pasos():
    # st.set_page_config(layout="centered") # ELIMINADO: Se llama solo una vez en main()
    st.title("🎓 Guía Paso a Paso: Resalto Hidráulico")

    # --- Inicialización del Estado de la Sesión ---
    if 'step' not in st.session_state:
        st.session_state.step = 0
        st.session_state.results = None

    # --- Funciones de Navegación ---
    def next_step():
        st.session_state.step += 1

    def prev_step():
        st.session_state.step -= 1

    def reset():
        st.session_state.step = 0
        st.session_state.results = None

    # --- Lógica de Vistas (Pasos) ---

    # --- PASO 0: ENTRADA DE DATOS ---
    if st.session_state.step == 0:
        st.header("Paso 1: Datos de Entrada")
        st.write("Introduzca los parámetros iniciales del canal y del flujo.")

        with st.form(key="data_form"):
            tipo_canal = st.selectbox("Tipo de Canal", ("Rectangular", "Trapezoidal"), key="pasos_tipo_canal")
            Q_input = st.number_input("Caudal (Q) [m³/s]", min_value=0.01, value=18.0, step=1.0, key="pasos_Q")
            b_input = st.number_input("Ancho de Solera (b) [m]", min_value=0.01, value=3.0, step=0.5, key="pasos_b")
            y1_input = st.number_input("Tirante Aguas Arriba (y1) [m]", min_value=0.01, value=1.0, step=0.1,
                                       key="pasos_y1")

            z_input = 0.0
            if tipo_canal == "Trapezoidal":
                z_input = st.number_input("Talud (z) [zH:1V]", min_value=0.0, value=1.0, step=0.5, key="pasos_z")

            submitted = st.form_submit_button("Calcular y Empezar Guía", type="primary")

        if submitted:
            # Calcular y guardar en el estado
            st.session_state.results = calcular_resalto(Q_input, b_input, y1_input, tipo_canal, z_input)

            if st.session_state.results["error"]:
                st.error(f"**Error:** {st.session_state.results['error']}")
                st.session_state.results = None  # Limpiar resultados erróneos
            else:
                st.session_state.step = 1  # Avanzar al siguiente paso
                st.rerun()  # Recargar la app para mostrar el siguiente paso

    # --- PASO 1: CONDICIONES AGUAS ARRIBA ---
    elif st.session_state.step == 1:
        st.header("Paso 2: Condiciones Aguas Arriba (Sección 1)")
        res = st.session_state.results

        st.write("Primero, analizamos el flujo de entrada (supercrítico).")

        st.metric("Tirante (y1)", f"{res['y1']:.3f} m")
        st.metric("Velocidad (V1)", f"{res['V1']:.3f} m/s")
        st.metric("Nº de Froude (Fr1)", f"{res['Fr1']:.3f}")
        st.metric("Energía Específica (E1)", f"{res['E1']:.3f} m")

        st.info(f"**Tipo de Resalto: {res['tipo']}**")
        st.warning(f"**Nota:** {res['nota']}")

        col1, col2 = st.columns(2)
        with col1:
            st.button("Atrás (Editar Datos)", on_click=reset, use_container_width=True)
        with col2:
            st.button("Siguiente (Ver Aguas Abajo) →", on_click=next_step, type="primary", use_container_width=True)

    # --- PASO 2: CONDICIONES AGUAS ABAJO ---
    elif st.session_state.step == 2:
        st.header("Paso 3: Condiciones Aguas Abajo (Sección 2)")
        res = st.session_state.results

        st.write("Usando el Nº de Froude (o la Fuerza Específica), calculamos el tirante conjugado `y2`.")

        st.metric("Tirante Conjugado (y2)", f"{res['y2']:.3f} m")
        st.info(f"**Método de cálculo:** {res['y2_detail']}")

        st.metric("Velocidad (V2)", f"{res['V2']:.3f} m/s")
        st.metric("Nº de Froude (Fr2)", f"{res['Fr2']:.3f} (Subcrítico, < 1)")
        st.metric("Energía Específica (E2)", f"{res['E2']:.3f} m")

        col1, col2 = st.columns(2)
        with col1:
            st.button("← Atrás (Ver Aguas Arriba)", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Siguiente (Ver Resultados) →", on_click=next_step, type="primary", use_container_width=True)

    # --- PASO 3: RESULTADOS Y GRÁFICO ---
    elif st.session_state.step == 3:
        st.header("Paso 4: Resultados Finales y Gráfico")
        res = st.session_state.results

        st.write("Finalmente, calculamos las propiedades del resalto y visualizamos la energía.")

        st.metric("Pérdida de Energía (ΔE = E1 - E2)", f"{res['delta_E']:.3f} m")
        st.metric("Altura del Resalto (h_j = y2 - y1)", f"{res['h_j']:.3f} m")

        lr_usbr_txt = f"{res['Lr_usbr']:.2f} m" if res['Lr_usbr'] else "N/A (Solo Rect.)"
        st.metric("Longitud (Lr, Aprox. USBR)", lr_usbr_txt)
        st.metric("Longitud (Lr, Silvester)", f"{res['Lr_sie']:.2f} m")

        # Generar y mostrar el gráfico
        st.subheader("Visualización de Diagramas")
        col_graf_1, col_graf_2 = st.columns(2)
        with col_graf_1:
            st.markdown("##### Diagrama E-y (Energía)")
            figura_Ey = plot_energia_especifica(res)
            st.pyplot(figura_Ey)
        with col_graf_2:
            st.markdown("##### Diagrama M-y (Fuerza Específica)")
            figura_My = plot_fuerza_especifica(res)
            st.pyplot(figura_My)

        col1, col2 = st.columns(2)
        with col1:
            st.button("← Atrás (Ver Aguas Abajo)", on_click=prev_step, use_container_width=True)
        with col2:
            st.button("Empezar de Nuevo", on_click=reset, type="primary", use_container_width=True)


# ===================================================
# --- SECCIÓN 4: EJECUTOR PRINCIPAL ---
# ===================================================

def main():
    # Configuración de la página (debe ser el primer comando st)
    st.set_page_config(
        page_title="App de Resalto Hidráulico",
        page_icon="🌊",
        layout="wide"
    )

    # Selector de modo de aplicación en la barra lateral
    st.sidebar.title("Configuración de la App")
    app_mode = st.sidebar.radio(
        "Selecciona el modo de la aplicación:",
        ("Calculadora Completa", "Guía Paso a Paso")
    )

    # Limpiar el estado si cambiamos de modo
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = app_mode

    if st.session_state.current_mode != app_mode:
        st.session_state.current_mode = app_mode
        # Reiniciar el estado de la app_pasos si cambiamos
        if 'step' in st.session_state:
            st.session_state.step = 0
        if 'results' in st.session_state:
            st.session_state.results = None
        st.rerun()

    # Ejecutar el renderizador de la aplicación seleccionada
    if app_mode == "Calculadora Completa":
        # st.set_page_config(layout="wide") # ELIMINADO
        run_app_completa()
    else:
        # st.set_page_config(layout="centered") # ELIMINADO
        run_app_pasos()


# Bloque de ejecución para PyCharm (o ejecución directa)
if __name__ == "__main__":
    # Esta es la forma estándar de Python.
    # Streamlit es iniciado por el comando 'streamlit run',
    # que luego importa y ejecuta este script.
    # La función main() será llamada automáticamente por Streamlit.

    # Se elimina la lógica de auto-arranque 'stcli'.
    # if stcli and "streamlit" not in " ".join(sys.argv):
    # ... (bloque if/else eliminado) ...
    # else:

    # Simplemente llamamos a main(), que es el punto de entrada
    # que Streamlit espera.
    main()


