"""
HERRAMIENTA INTERACTIVA PARA ANÁLISIS DE COMPUERTA CON FLUJO MODULAR
Aplicación web con Streamlit para análisis hidráulico iterativo

Autor: Emanuel - Análisis Estructural & Machine Learning
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Compuerta - Flujo Modular",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


class CompuertaCalculadora:
    """Clase para cálculos de compuerta con flujo modular"""

    def __init__(self, Q, b, Cc):
        self.Q = Q
        self.b = b
        self.Cc = Cc
        self.g = 9.81

    def calcular_parametros(self, a, y1):
        """Calcula todos los parámetros para un valor dado de a y y1"""
        # Relación fundamental
        y2 = self.Cc * a

        # Área del orificio
        Ao = a * self.b

        # Coeficiente de velocidad
        Cv = 1 / np.sqrt(1 + y2/y1)

        # Coeficiente de descarga
        Cd = self.Cc * Cv

        # Velocidad teórica
        v_teorica = np.sqrt(2 * self.g * (y1 - y2))

        # Caudal calculado
        Q_calc = Cd * Ao * v_teorica

        # Diferencia
        Diff = Q_calc - self.Q

        # Velocidad aguas abajo
        v2 = Q_calc / (self.b * y2)

        # Número de Froude
        F = v2 / np.sqrt(self.g * y2)

        # Tirante conjugado
        y3 = y2/2 * (-1 + np.sqrt(1 + 8*F**2))

        # Longitud de resalto
        L = 6 * (y3 - y2)

        return {
            'a': a,
            'y1': y1,
            'y2': y2,
            'Ao': Ao,
            'Cv': Cv,
            'Cd': Cd,
            'v_teorica': v_teorica,
            'Q_calc': Q_calc,
            'Diff': Diff,
            'v2': v2,
            'F': F,
            'y3': y3,
            'L': L,
            'y1/a': y1/a,
            '2/3*y1': (2/3)*y1,
            'cumple_y1_a': y1/a > 1.35,
            'cumple_flujo_modular': (2/3)*y1 > a
        }

    def newton_raphson_iteracion(self, a, y1_inicial, tol=1e-6, max_iter=50):
        """Ejecuta Newton-Raphson y devuelve todas las iteraciones"""
        y1 = y1_inicial
        y2 = self.Cc * a

        if y1 <= y2:
            y1 = y2 + 1.0

        iteraciones = []

        for i in range(max_iter):
            params = self.calcular_parametros(a, y1)

            iteraciones.append({
                'iteracion': i + 1,
                'y1': y1,
                'y2': params['y2'],
                'Cd': params['Cd'],
                'Q_calc': params['Q_calc'],
                'Diff': params['Diff'],
                'F': params['F']
            })

            if abs(params['Diff']) < tol:
                return y1, params, iteraciones, True

            # Derivada numérica
            h = 0.001
            params_plus = self.calcular_parametros(a, y1 + h)
            params_minus = self.calcular_parametros(a, y1 - h)
            df = (params_plus['Diff'] - params_minus['Diff']) / (2 * h)

            if abs(df) < 1e-15:
                break

            # Actualización
            y1_nuevo = y1 - params['Diff'] / df

            if y1_nuevo <= y2:
                y1_nuevo = y2 + 0.1

            y1 = y1_nuevo

        params = self.calcular_parametros(a, y1)
        return y1, params, iteraciones, False


def plot_convergencia(df_iter):
    """Crea gráfico de convergencia con Plotly"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Evolución de y1', 'Error (|Diff|)',
                       'Evolución de Cd', 'Caudal calculado'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Gráfico 1: y1
    fig.add_trace(
        go.Scatter(x=df_iter['iteracion'], y=df_iter['y1'],
                  mode='lines+markers', name='y1',
                  line=dict(color='#1E88E5', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )

    # Gráfico 2: Error
    fig.add_trace(
        go.Scatter(x=df_iter['iteracion'], y=np.abs(df_iter['Diff']),
                  mode='lines+markers', name='|Diff|',
                  line=dict(color='#E53935', width=3),
                  marker=dict(size=8)),
        row=1, col=2
    )
    fig.update_yaxes(type="log", row=1, col=2)

    # Gráfico 3: Cd
    fig.add_trace(
        go.Scatter(x=df_iter['iteracion'], y=df_iter['Cd'],
                  mode='lines+markers', name='Cd',
                  line=dict(color='#43A047', width=3),
                  marker=dict(size=8)),
        row=2, col=1
    )

    # Gráfico 4: Q_calc
    fig.add_trace(
        go.Scatter(x=df_iter['iteracion'], y=df_iter['Q_calc'],
                  mode='lines+markers', name='Q calculado',
                  line=dict(color='#FB8C00', width=3),
                  marker=dict(size=8)),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Iteración", row=1, col=1)
    fig.update_xaxes(title_text="Iteración", row=1, col=2)
    fig.update_xaxes(title_text="Iteración", row=2, col=1)
    fig.update_xaxes(title_text="Iteración", row=2, col=2)

    fig.update_yaxes(title_text="y1 (m)", row=1, col=1)
    fig.update_yaxes(title_text="|Diff| (m³/s)", row=1, col=2)
    fig.update_yaxes(title_text="Cd", row=2, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="Proceso de Convergencia Newton-Raphson",
        title_font_size=20
    )

    return fig


def plot_perfil_hidraulico(params):
    """Crea perfil hidráulico del flujo"""
    y1 = params['y1']
    y2 = params['y2']
    a = params['a']
    y3 = params['y3']
    L = params['L']

    fig = go.Figure()

    # Fondo del canal
    fig.add_shape(type="rect",
                  x0=0, x1=20, y0=-0.2, y1=0,
                  fillcolor="brown", opacity=0.3,
                  line_width=0)

    # Sección aguas arriba
    fig.add_shape(type="rect",
                  x0=0, x1=5, y0=0, y1=y1,
                  fillcolor="lightblue", opacity=0.6,
                  line=dict(color="blue", width=2))

    # Compuerta
    fig.add_shape(type="rect",
                  x0=5, x1=5.3, y0=0, y1=a,
                  fillcolor="gray", opacity=0.8,
                  line=dict(color="black", width=2))

    # Chorro
    x_chorro = np.linspace(5.3, 8, 30)
    y_chorro = a - (a - y2) * ((x_chorro - 5.3)/(8 - 5.3))**1.5
    fig.add_trace(go.Scatter(x=x_chorro, y=y_chorro,
                            fill='tonexty', fillcolor='rgba(100, 181, 246, 0.6)',
                            line=dict(color='blue', width=2),
                            name='Chorro'))

    # Sección aguas abajo
    fig.add_shape(type="rect",
                  x0=8, x1=15, y0=0, y1=y2,
                  fillcolor="darkblue", opacity=0.6,
                  line=dict(color="blue", width=2))

    # Resalto hidráulico
    fig.add_shape(type="rect",
                  x0=15, x1=15+abs(L), y0=0, y1=y3,
                  fillcolor="lightgreen", opacity=0.5,
                  line=dict(color="green", width=2, dash="dash"))

    # Anotaciones
    fig.add_annotation(x=2.5, y=y1/2, text=f"y1={y1:.3f}m",
                      showarrow=True, arrowhead=2, font=dict(size=12, color="blue"))
    fig.add_annotation(x=11, y=y2/2, text=f"y2={y2:.3f}m",
                      showarrow=True, arrowhead=2, font=dict(size=12, color="blue"))
    fig.add_annotation(x=5.15, y=a/2, text=f"a={a:.2f}m",
                      font=dict(size=10, color="white"))

    fig.update_layout(
        title="Perfil Hidráulico del Flujo",
        xaxis_title="Distancia (m)",
        yaxis_title="Elevación (m)",
        height=500,
        showlegend=False,
        xaxis=dict(range=[-1, 20]),
        yaxis=dict(range=[-0.5, max(y1, y3) + 0.5])
    )

    return fig


def main():
    """Función principal de la aplicación"""

    # Título principal
    st.markdown('<p class="main-header">🌊 Análisis de Compuerta con Flujo Modular</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar con parámetros
    with st.sidebar:
        st.markdown("## ⚙️ Parámetros de Diseño")

        Q = st.number_input("Caudal Q (m³/s)", value=0.8, min_value=0.1,
                           max_value=5.0, step=0.1, format="%.2f")
        b = st.number_input("Ancho b (m)", value=0.2, min_value=0.1,
                           max_value=2.0, step=0.05, format="%.2f")
        Cc = st.number_input("Coeficiente Cc", value=0.61, min_value=0.5,
                            max_value=0.9, step=0.01, format="%.2f")

        st.markdown("---")
        st.markdown("## 🎯 Parámetro de Iteración")

        # Opciones predefinidas del Excel
        casos_excel = {
            "Caso 1: a = 1.15 m": 1.15,
            "Caso 2: a = 1.20 m": 1.20,
            "Caso 3: a = 1.35 m": 1.35,
            "Caso 4: a = 1.40 m": 1.40,
            "Personalizado": None
        }

        caso_seleccionado = st.selectbox("Seleccionar caso:", list(casos_excel.keys()))

        if casos_excel[caso_seleccionado] is not None:
            a = casos_excel[caso_seleccionado]
            st.info(f"📏 Altura de compuerta: **{a} m**")
        else:
            a = st.number_input("Altura a (m)", value=1.15, min_value=0.5,
                               max_value=3.0, step=0.05, format="%.2f")

        st.markdown("---")
        st.markdown("## 🔧 Parámetros de Convergencia")

        y1_inicial = st.number_input("y1 inicial (m)", value=3.0, min_value=1.0,
                                     max_value=10.0, step=0.5, format="%.2f")
        tol = st.select_slider("Tolerancia",
                              options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                              value=1e-6,
                              format_func=lambda x: f"{x:.0e}")

        calcular = st.button("🚀 CALCULAR", type="primary", use_container_width=True)

    # Área principal
    if calcular:
        # Crear calculadora
        calc = CompuertaCalculadora(Q, b, Cc)

        # Ejecutar Newton-Raphson
        with st.spinner('Calculando...'):
            y1_sol, params, iteraciones, convergio = calc.newton_raphson_iteracion(
                a, y1_inicial, tol
            )

        # Crear DataFrame de iteraciones
        df_iter = pd.DataFrame(iteraciones)

        # Tabs para organizar resultados
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Resumen", "🔄 Iteraciones", "📈 Convergencia",
            "🌊 Perfil Hidráulico", "📋 Comparación Excel"
        ])

        with tab1:
            st.markdown("### 🎯 Resultados Finales")

            if convergio:
                st.success(f"✅ Convergió en {len(iteraciones)} iteraciones")
            else:
                st.warning("⚠️ No convergió completamente")

            # Métricas principales en columnas
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("y1 (m)", f"{params['y1']:.4f}")
                st.metric("y2 (m)", f"{params['y2']:.4f}")

            with col2:
                st.metric("Cd", f"{params['Cd']:.4f}")
                st.metric("Cv", f"{params['Cv']:.4f}")

            with col3:
                st.metric("Q calc (m³/s)", f"{params['Q_calc']:.6f}")
                st.metric("Diff", f"{params['Diff']:.2e}")

            with col4:
                st.metric("F (Froude)", f"{params['F']:.3f}")
                st.metric("v2 (m/s)", f"{params['v2']:.3f}")

            st.markdown("---")

            # Parámetros adicionales
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📐 Geometría")
                geo_data = {
                    'Parámetro': ['Altura compuerta (a)', 'Área orificio (Ao)',
                                 'Tirante conjugado (y3)', 'Longitud resalto (L)'],
                    'Valor': [f"{params['a']:.3f} m", f"{params['Ao']:.3f} m²",
                             f"{params['y3']:.3f} m", f"{params['L']:.3f} m"]
                }
                st.dataframe(pd.DataFrame(geo_data), hide_index=True, use_container_width=True)

            with col2:
                st.markdown("### ✅ Verificación")

                check1 = "✅" if params['cumple_y1_a'] else "❌"
                check2 = "✅" if params['cumple_flujo_modular'] else "❌"

                verif_data = {
                    'Condición': ['y1/a > 1.35', '2/3·y1 > a'],
                    'Valor': [f"{params['y1/a']:.3f}", f"{params['2/3*y1']:.3f}"],
                    'Estado': [check1, check2]
                }
                st.dataframe(pd.DataFrame(verif_data), hide_index=True, use_container_width=True)

                if params['cumple_y1_a'] and params['cumple_flujo_modular']:
                    st.success("🎯 FLUJO MODULAR VERIFICADO")
                else:
                    st.error("⚠️ NO CUMPLE CONDICIONES DE FLUJO MODULAR")

        with tab2:
            st.markdown("### 🔄 Tabla de Iteraciones")

            # Formatear DataFrame
            df_display = df_iter.copy()
            df_display['y1'] = df_display['y1'].apply(lambda x: f"{x:.6f}")
            df_display['y2'] = df_display['y2'].apply(lambda x: f"{x:.6f}")
            df_display['Cd'] = df_display['Cd'].apply(lambda x: f"{x:.6f}")
            df_display['Q_calc'] = df_display['Q_calc'].apply(lambda x: f"{x:.6f}")
            df_display['Diff'] = df_display['Diff'].apply(lambda x: f"{x:.2e}")
            df_display['F'] = df_display['F'].apply(lambda x: f"{x:.6f}")

            st.dataframe(df_display, use_container_width=True, height=400)

            # Botón para descargar
            csv = df_iter.to_csv(index=False)
            st.download_button(
                label="📥 Descargar iteraciones (CSV)",
                data=csv,
                file_name=f"iteraciones_a{a:.2f}.csv",
                mime="text/csv"
            )

        with tab3:
            st.markdown("### 📈 Gráficos de Convergencia")
            fig_conv = plot_convergencia(df_iter)
            st.plotly_chart(fig_conv, use_container_width=True)

        with tab4:
            st.markdown("### 🌊 Perfil Hidráulico del Flujo")
            fig_perfil = plot_perfil_hidraulico(params)
            st.plotly_chart(fig_perfil, use_container_width=True)

            # Información adicional
            col1, col2, col3 = st.columns(3)

            with col1:
                tipo_flujo = "Supercrítico (F>1)" if params['F'] > 1 else "Subcrítico (F<1)"
                st.info(f"**Tipo de flujo:** {tipo_flujo}")

            with col2:
                E1 = params['y1'] + params['v2']**2/(2*calc.g)
                E2 = params['y2'] + params['v2']**2/(2*calc.g)
                st.info(f"**Pérdida de energía:** {E1-E2:.3f} m")

            with col3:
                potencia = calc.g * Q * (E1 - E2) * 1000  # en W
                st.info(f"**Potencia disipada:** {potencia:.1f} W")

        with tab5:
            st.markdown("### 📋 Comparación con Datos del Excel")

            # Datos del Excel
            datos_excel = {
                1.15: {'y1': 2.775418, 'y2': 0.7015, 'Cd': 0.545000, 'F': 2.173622},
                1.20: {'y1': 2.669476, 'y2': 0.7320, 'Cd': 0.540392, 'F': 2.039196},
                1.35: {'y1': 2.432090, 'y2': 0.8235, 'Cd': 0.527236, 'F': 1.708953},
                1.40: {'y1': 2.373526, 'y2': 0.8540, 'Cd': 0.523109, 'F': 1.618224}
            }

            if a in datos_excel:
                excel = datos_excel[a]

                comp_data = {
                    'Parámetro': ['y1 (m)', 'y2 (m)', 'Cd', 'F'],
                    'Python': [f"{params['y1']:.6f}", f"{params['y2']:.6f}",
                              f"{params['Cd']:.6f}", f"{params['F']:.6f}"],
                    'Excel': [f"{excel['y1']:.6f}", f"{excel['y2']:.6f}",
                             f"{excel['Cd']:.6f}", f"{excel['F']:.6f}"],
                    'Error': [f"{abs(params['y1']-excel['y1'])*1000:.3f} mm",
                             f"{abs(params['y2']-excel['y2'])*1000:.3f} mm",
                             f"{abs(params['Cd']-excel['Cd']):.6f}",
                             f"{abs(params['F']-excel['F']):.6f}"]
                }

                st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

                # Análisis de precisión
                error_y1 = abs(params['y1'] - excel['y1']) * 1000
                error_Cd = abs(params['Cd'] - excel['Cd']) / excel['Cd'] * 100

                col1, col2 = st.columns(2)
                with col1:
                    if error_y1 < 2:
                        st.success(f"✅ Error en y1: {error_y1:.3f} mm (Excelente)")
                    elif error_y1 < 5:
                        st.info(f"ℹ️ Error en y1: {error_y1:.3f} mm (Bueno)")
                    else:
                        st.warning(f"⚠️ Error en y1: {error_y1:.3f} mm")

                with col2:
                    if error_Cd < 0.1:
                        st.success(f"✅ Error en Cd: {error_Cd:.3f}% (Excelente)")
                    elif error_Cd < 1:
                        st.info(f"ℹ️ Error en Cd: {error_Cd:.3f}% (Bueno)")
                    else:
                        st.warning(f"⚠️ Error en Cd: {error_Cd:.3f}%")
            else:
                st.info("ℹ️ No hay datos de comparación del Excel para este valor de 'a'")
                st.markdown("Los casos disponibles son: 1.15, 1.20, 1.35, 1.40 m")

    else:
        # Mensaje inicial
        st.info("👈 Configura los parámetros en el panel lateral y presiona **CALCULAR**")

        st.markdown("### 📚 Acerca de esta herramienta")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Características:**
            - ✅ Método Newton-Raphson con derivada numérica
            - ✅ Visualización en tiempo real del proceso iterativo
            - ✅ Comparación automática con datos del Excel
            - ✅ Gráficos interactivos de convergencia
            - ✅ Perfil hidráulico del flujo
            - ✅ Exportación de resultados
            """)

        with col2:
            st.markdown("""
            **Casos predefinidos del Excel:**
            - Caso 1: a = 1.15 m
            - Caso 2: a = 1.20 m
            - Caso 3: a = 1.35 m
            - Caso 4: a = 1.40 m
            - Personalizado: cualquier valor
            """)

        st.markdown("---")
        st.markdown("""
        ### 🔬 Fundamento Matemático
        
        **Ecuaciones principales:**
        - Relación geométrica: `y2 = Cc × a`
        - Coeficiente de velocidad: `Cv = 1/sqrt(1 + y2/y1)`
        - Coeficiente de descarga: `Cd = Cc × Cv`
        - Caudal: `Q = Cd × Ao × sqrt(2g(y1-y2))`
        
        **Método de solución:**
        - Newton-Raphson con derivada numérica
        - Convergencia típica: 3-4 iteraciones
        - Precisión: < 1e-6 m³/s
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    📊 Desarrollado por Emanuel | Análisis Estructural & Machine Learning<br>
    🌊 Herramienta para análisis hidráulico profesional
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()