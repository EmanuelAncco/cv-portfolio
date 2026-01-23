import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calcular_step1_elastico(D, t, E, tau_u, q_u, PGD, beta_deg, L1, L2):
    """
    Cálculo simplificado del Paso 1 (puramente elástico) del Apéndice A.
    Se asumen las siguientes simplificaciones:
      - Ángulos de rotación: phiA, phiB
      - Elongación requerida = DeltaL_req
      - Sigma_axial usando ecuación sencilla (sqrt(E*tau_u*DeltaL_req/A))
      - No se incluyen momentos ni cortantes explícitos.
    Retorna: phiA, phiB, sigma_ax, elong_req, x2D, y2D, x3D, y3D, z3D
    para graficar la deformada en 2D y 3D.
    """

    # Conversión de ángulo a radianes
    beta = np.radians(beta_deg)

    # Componentes de desplazamiento según el ángulo
    Dx = PGD * np.cos(beta)
    Dy = PGD * np.sin(beta)

    # Área y momento de inercia (simplificado)
    A = np.pi * (D - t) * t

    # Paso 1.1: suposición inicial de rotaciones
    # (en un método más avanzado se itera, aquí fijamos un approach súper simple)
    # Aproximamos phiB "mayor" cuando la falla está cerca de B;
    # phiA "pequeño" si la falla se ubica en B o en la segunda junta
    phiA_guess = 0.2 * np.pi / 180.0  # ~0.2 grados
    phiB_guess = 3.5 * np.pi / 180.0  # ~3.5 grados

    # Paso 1.2: calcular elongación requerida (Ec. 16, simplificada)
    def elong_required(phiA, phiB):
        cA = np.cos(phiA) if abs(np.cos(phiA)) > 1e-10 else 1e-10
        cB = np.cos(phiA + phiB) if abs(np.cos(phiA + phiB)) > 1e-10 else 1e-10
        return Dx + 2 * ((L1 / cA - L1) + (L2 / cB - L2))

    dL_req = elong_required(phiA_guess, phiB_guess)
    if dL_req < 0:
        dL_req = 0  # Si el cruce es oblicuo en compresión, en este ejemplo simple lo forzamos a 0

    # Paso 1.3: calcular sigma axial (Ecuación 20 simplificada)
    # sigma_ax = sqrt( E * tau_u * dL_req / A ), asumiendo < sigma_y
    val = E * tau_u * dL_req / A
    sigma_ax = np.sqrt(val) if val > 0 else 0

    # Para graficar la deformada 2D:
    # supondremos (x, y) en el plano, con 2 tramos: AB (largo L1) y BC (largo L2).
    # y_AB = phiA_guess*x (lineal). y_BC = y_AB(L1) + (x-L1)*(phiA+phiB).
    npts = 20
    xAB = np.linspace(0, L1, npts)
    yAB = phiA_guess * xAB  # lineal
    if L2 > 0:
        xBC = np.linspace(L1, L1 + L2, npts)
        yBC = yAB[-1] + (xBC - L1) * (phiA_guess + phiB_guess)
        x2D = np.concatenate([xAB, xBC])
        y2D = np.concatenate([yAB, yBC])
    else:
        # falla en B => xBC = L1, yBC = yAB[-1]
        x2D = xAB
        y2D = yAB

    # Para un gráfico 3D sencillo, podemos trazar la tubería como una curva en X-Y
    # con Z=0, y añadir la componente vertical de Dy para "ilustrar" algo,
    # o simplemente mostrar la misma deformada en un plano 3D.
    # Aquí haremos que Z=0 y la "altura" (Z) no cambie, solo para que sea "3D" en la visualización.
    z2D = np.zeros_like(x2D)  # la tubería en "altura 0"

    # Retornamos resultados
    return (phiA_guess, phiB_guess, sigma_ax, dL_req, x2D, y2D, z2D, Dx, Dy)


def main():
    st.title("Ejemplo: Paso 1 (Apéndice A) - Tubería con Juntas Flexibles")

    # Texto en español del Paso 1 (resumen/traducción libre)
    paso1_text = r"""
**Paso 1 (Análisis Elástico Inicial)**

En este paso, se supone un comportamiento puramente elástico de la tubería.  
Se definen suposiciones iniciales para las rotaciones en las juntas flexibles 
(\(\varphi_A\) y \(\varphi_B\)) y se calcula la **elongación requerida** 
(\(\Delta L_{\mathrm{req}}\)) a partir de la componente axial de la falla 
(\(\Delta x\)) y la contribución geométrica de los tramos AB y BC. 

Luego, se estima la **tensión axial** (\(\sigma_\alpha\)) asumiendo que 
\(\sigma_\alpha < \sigma_y\). Si la elongación o tensión calculada supera valores 
críticos, se reajustan las suposiciones, avanzando a los pasos siguientes donde 
se contemplan efectos no lineales y verificación de momentos, cortantes y 
compatibilidad más rigurosa.

*(Ver Ecs. (16)–(20) del artículo para más detalles.)*
    """

    # Desplegamos en un "expander" para no saturar la vista principal
    with st.expander("Texto (en español) del Paso 1 del Apéndice A"):
        st.markdown(paso1_text)

    st.sidebar.title("Parámetros de Entrada")

    # Entradas de usuario
    D = st.sidebar.slider("Diámetro [m]", 0.2, 1.0, 0.762, 0.01)
    t = st.sidebar.slider("Espesor [m]", 0.005, 0.05, 0.0125, 0.001)
    E = st.sidebar.number_input("Módulo Elástico E [Pa]", value=2.10e11, format="%.2e")
    tau_u = st.sidebar.number_input("Fuerza fricción (tau_u) [N/m]", value=2.275e4, format="%.2e")
    q_u = st.sidebar.number_input("Fuerza lateral (q_u) [N/m]", value=1.3445e5, format="%.2e")
    PGD = st.sidebar.slider("Desplazamiento total (PGD) [m]", 0.0, 3.0, 1.5, 0.1)
    beta = st.sidebar.slider("Ángulo de cruce beta [°]", 0.0, 90.0, 60.0, 1.0)
    L1 = st.sidebar.slider("Long. tramo AB [m]", 1.0, 20.0, 8.0, 0.5)
    L2 = st.sidebar.slider("Long. tramo BC [m]", 0.0, 20.0, 0.0, 0.5)

    # Llamamos la función de cálculo
    phiA, phiB, sigma_ax, dL_req, x2D, y2D, z2D, Dx, Dy = calcular_step1_elastico(
        D, t, E, tau_u, q_u, PGD, beta, L1, L2
    )

    # Resultados en la interfaz
    st.subheader("Resultados (Paso 1 Simplificado)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rotación estimada (phiA)** = {np.degrees(phiA):.4f}°")
        st.write(f"**Rotación estimada (phiB)** = {np.degrees(phiB):.4f}°")
        st.write(f"**Elongación requerida** = {dL_req:.6f} m")

    with col2:
        st.write(f"**Tensión axial (sigma_ax)** = {sigma_ax * 1e-6:.3f} MPa")
        st.write(f"**Despl. falla**: Dx={Dx:.3f} m, Dy={Dy:.3f} m")

    # --- GRAFICO 2D ---
    fig2d, ax2d = plt.subplots(figsize=(6, 4))
    ax2d.plot(x2D, y2D, "b-o", label="Tubería (deformada 2D)")
    ax2d.set_title("Deformada aproximada en 2D")
    ax2d.set_xlabel("Longitud a lo largo del eje (m)")
    ax2d.set_ylabel("Desplazamiento transversal (m)")
    ax2d.grid(True)
    ax2d.legend()
    st.pyplot(fig2d)

    # --- GRAFICO 3D ---
    fig3d = plt.figure(figsize=(6, 4))
    ax3d = fig3d.add_subplot(111, projection='3d')
    # Trazamos la curva
    ax3d.plot(x2D, y2D, z2D, "r-o", label="Tubería 3D (z=0)")

    ax3d.set_title("Visualización 3D (Vista en Perspectiva)")
    ax3d.set_xlabel("Eje X (m)")
    ax3d.set_ylabel("Eje Y (m)")
    ax3d.set_zlabel("Eje Z (m)")
    # Ajuste de los límites para verse mejor
    max_range = max(L1 + L2, max(abs(y2D))) + 1
    ax3d.set_xlim(0, max(L1 + L2, 0.1))
    ax3d.set_ylim(-max_range / 2, max_range / 2)
    ax3d.set_zlim(-max_range / 2, max_range / 2)
    ax3d.legend()

    st.pyplot(fig3d)

    st.markdown("""
    ---
    **Nota**: Este ejemplo no integra todas las ecuaciones de
    momento, cortante y compatibilidad axial del artículo. 
    Para resultados idénticos a los del Apéndice A, 
    implementa todas las ecuaciones y el método iterativo
    recomendado (Newton-Raphson multivariable).
    """)


if __name__ == "__main__":
    main()
