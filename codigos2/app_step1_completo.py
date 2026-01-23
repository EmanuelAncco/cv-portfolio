import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------

def elongacion_requerida(Dx, L1, L2, phiA, phiB):
    """
    Ecuación (16) del artículo (Paso 1)
    Delta L_req = Dx + 2 [ (L1/cos(phiA) - L1) + (L2/cos(phiA+phiB) - L2) ]
    """
    cA = np.cos(phiA) if abs(np.cos(phiA)) > 1e-9 else 1e-9
    cB = np.cos(phiA + phiB) if abs(np.cos(phiA + phiB)) > 1e-9 else 1e-9
    return Dx + 2.0 * ((L1 / cA - L1) + (L2 / cB - L2))


def elongacion_disponible(elastic_mod, tau_u, A, DLreq, sigma_y=None):
    """
    Paso 1: asumiendo que sigma < sigma_y (o no se supera plasticidad),
    Ecuación (20) => sigma = sqrt(E * tau_u * DLreq / A)
    N = sigma * A
    (Si DLreq =>0 se pone 0)
    Retorna (sigma, N)
    """
    if DLreq <= 0.0:
        return (0.0, 0.0)
    # Si hubiera que chequear si sigma > sigma_y, habría que pasar a Eq. (23).
    val = elastic_mod * tau_u * DLreq / A
    sigma_calc = np.sqrt(val) if val > 0 else 0
    # Chequeo simple (por si excede el limite elástico, no implementamos Step II aquí)
    if sigma_y and sigma_calc > sigma_y:
        sigma_calc = sigma_y
    N_calc = sigma_calc * A
    return (sigma_calc, N_calc)


def calcular_tramos(phiA, phiB, L1, L2, Dy):
    """
    - Divide la tubería en 4 tramos con simetría respecto a la falla:
      1) A->B  (izquierda)
      2) B->Falla
      3) Falla->B' (simétrico)
      4) B'->A' (simétrico)
    - Para simplificar, llamamos "B->Falla" al tramo de longitud L2
      con rotación (phiA+phiB).
    - Se asume la misma deformada en espejo a la derecha de la falla.
    - Devuelve coordenadas (x,y) de la tubería y un array con la longitud total discretizada.
    """
    # Se asume y=0 hasta llegar a A->B y B->Falla con rotación
    # EJEMPLO (abajo se hace muy simple: lineal en cada tramo)
    npts = 21

    # 1) Tramo A->B: rotación ~ phiA
    xAB = np.linspace(0, L1, npts)
    yAB = phiA * xAB  # lineal, como aproximación
    # 2) Tramo B->Falla: rotación ~ phiA+phiB
    xBC = np.linspace(L1, L1 + L2, npts)
    yBC = yAB[-1] + (xBC - L1) * (phiA + phiB)

    # Concatenamos lado izquierdo
    x_izq = np.concatenate([xAB, xBC])
    y_izq = np.concatenate([yAB, yBC])

    # La falla la ubicamos al final del xBC => xFalla = L1+L2
    # (yFalla = yBC[-1], debería ser ~ Dy/2 si se cumple la eq. (14))

    # Por simetría, lado derecho:
    # Lo repetimos de L1+L2 -> L1+L2+L2 -> L1+L2+L2+L1
    xCD = np.linspace(L1 + L2, L1 + L2 + L2, npts)
    # rotación (phiA+phiB) pero y sube
    yCD = yBC[-1] + (xCD - (L1 + L2)) * (phiA + phiB)  # continuamos
    xDA = np.linspace(L1 + L2 + L2, L1 + L2 + L2 + L1, npts)
    # rotación phiA
    yDA = yCD[-1] + (xDA - (L1 + L2 + L2)) * phiA

    x_der = np.concatenate([xCD, xDA])
    y_der = np.concatenate([yCD, yDA])

    # Juntamos todo: A->B->Falla->B'->A'
    x_tot = np.concatenate([x_izq, x_der])
    y_tot = np.concatenate([y_izq, y_der])

    return x_tot, y_tot


def calcular_fuerzas_normales(phiA, phiB, N_falla, L1, L2, tau_u):
    """
    A modo de ejemplo, usamos Ecs. (24a) y (24b) (N1(x) = NB - (L1-x)*tau_u)
    para estimar la variación de la fuerza normal (axial) a lo largo de AB, BC.
    Supongamos NB ~ N_falla, etc.
    De forma muy resumida y lineal.

    Devuelve x y N(x) en 4 tramos (A->B, B->Falla, falla->B', B'->A').
    """
    # Para la izquierda:
    # N_falla = N en la falla. A->B => se parte de N_B = N_falla?
    # (en realidad, eq. 24a => N1(x)= NB - (L1 - x)*tau_u)
    # análogamente para B->Falla => eq. 24b.
    # Por simetría, lado derecho igual.

    # Simplificaremos asumiendo N_B = N_falla, N_A = N_B + L1*tau_u.
    NB = N_falla
    NA = NB + L1 * tau_u

    # Discretización para A->B
    npts = 21
    xAB = np.linspace(0, L1, npts)
    # eq. 24a => N1(x)= NB - (L1 - x)*tau_u
    NAB = NB - (L1 - xAB) * tau_u

    # B->Falla
    # eq. 24b => N2(x) = NC - (L2 - x)*tau_u
    # Asumimos NC= N_falla. x corre 0..L2
    xBC = np.linspace(0, L2, npts)
    NBC = N_falla - (L2 - xBC) * tau_u

    # Lado derecho (simetría)
    # A'->B' y B'->Falla. Mismo N
    # Los definimos como espejo:
    xABd = np.linspace(0, L1, npts)
    NABd = NB - (L1 - xABd) * tau_u
    xBCd = np.linspace(0, L2, npts)
    NBCd = N_falla - (L2 - xBCd) * tau_u

    return (xAB, NAB, xBC, NBC, xABd, NABd, xBCd, NBCd)


# -----------------------------------------------------------
# APLICACIÓN PRINCIPAL STREAMLIT
# -----------------------------------------------------------
def main():
    st.title("Tubería con Juntas Flexibles")
    st.subheader("Versión Completa del Paso 1 (Apéndice A)")

    # Texto explicativo (en español) del Paso 1
    texto_paso1 = r"""
**Resumen del Paso 1 (Apéndice A) - Ecuaciones Principales:**

1. Se asume tubería en **régimen elástico** (sin plastificación).  
2. Se plantean las rotaciones \(\varphi_A\) y \(\varphi_B\) en las juntas más cercanas a la falla.  
3. Se satisfacen las condiciones:  
   \[
     M_1(L_1)=0, \quad M_2(L_2)=0, 
     \quad \delta_2(L_2)=\frac{D_y}{2}, 
     \quad \Delta L_\text{req} = \Delta L_\text{av}.
   \]
   (Ver Ecs. (12)-(15).)  

4. Para la componente axial, la **elongación requerida** \(\Delta L_{\mathrm{req}}\) se calcula con la (Ec. 16):  
   \[
     \Delta L_{\text{req}} 
     = \Delta x
     + 2 \left(\frac{L_1}{\cos\varphi_A} - L_1\right)
     + 2 \left(\frac{L_2}{\cos(\varphi_A+\varphi_B)} - L_2\right).
   \]

5. La **elongación disponible** en Paso 1 (sin plastificar) se obtiene de la (Ec. 20) 
   asumiendo \(\sigma_\alpha = \sqrt{\frac{E\,\tau_u\,\Delta L_{\text{req}}}{A}}\).  
   Si \(\sigma_\alpha\) excede \(\sigma_y\), se pasaría a Paso II.  

6. Se discretizan 4 tramos por **simetría**:
   - A->B (longitud \(L_1\))
   - B->Falla (longitud \(L_2\))
   - Falla->B' (longitud \(L_2\))
   - B'->A' (longitud \(L_1\))

*(Nota: Aquí se hace una aproximación lineal al momento y rotación en cada tramo. Para resultados exactos, aplicar iteración Newton-Raphson con Ecs. (12)-(15).)*
    """
    with st.expander("Texto (en español) del Paso 1 del Apéndice A"):
        st.markdown(texto_paso1)

    # BARRA LATERAL DE PARÁMETROS
    st.sidebar.title("Parámetros de Entrada")

    D = st.sidebar.slider("Diámetro [m]", 0.2, 1.0, 0.76, 0.01)
    t = st.sidebar.slider("Espesor [m]", 0.01, 0.05, 0.012, 0.001)
    E = st.sidebar.number_input("Módulo Elástico E [Pa]", value=2.10e11, format="%.2e")
    tau_u = st.sidebar.number_input("Fuerza fricción (tau_u) [N/m]", value=2.28e4, format="%.2e")
    q_u = st.sidebar.number_input("Fuerza lateral (q_u) [N/m]", value=1.34e5, format="%.2e")
    PGD = st.sidebar.slider("Desplazamiento total (PGD) [m]", 0.0, 3.0, 1.5, 0.1)
    beta_deg = st.sidebar.slider("Ángulo de cruce beta [°]", 0.0, 90.0, 60.0, 1.0)
    L1 = st.sidebar.slider("Long. tramo AB [m]", 1.0, 20.0, 8.0, 0.5)
    L2 = st.sidebar.slider("Long. tramo BC [m]", 0.0, 20.0, 8.0, 0.5)
    sigma_y = st.sidebar.number_input("Esf. fluencia (sigma_y) [Pa]", value=4.90e8, format="%.2e")

    # COMPONENTES DEL DESPLAZAMIENTO DE LA FALLA
    rad = np.radians(beta_deg)
    Dx = PGD * np.cos(rad)
    Dy = PGD * np.sin(rad)

    # GEOMETRÍA
    A = np.pi * (D - t) * t  # area ~ pi*(Dext - t)*t
    # Se omite I en este ejemplo, pero se usaría en eq. (12)-(15)

    # -----------------------------------------------
    # Paso 1: SUPOSICION de phiA y phiB (aquí fijas o se hace una mini-iteración)
    # Para hacerlo "dinámico", haremos un slider para phiA, phiB
    phiA_deg = st.sidebar.slider("Rotación en Junta A [°]", 0.0, 5.0, 0.2, 0.1)
    phiB_deg = st.sidebar.slider("Rotación en Junta B [°]", 0.0, 10.0, 3.5, 0.1)
    phiA = np.radians(phiA_deg)
    phiB = np.radians(phiB_deg)

    # -----------------------------------------------
    # 1) Calcular elongación requerida (Eq. 16)
    dL_req = elongacion_requerida(Dx, L1, L2, phiA, phiB)

    # 2) Calcular sigma axial asumiendo eq. (20) (condición elástica)
    sigma_ax, N_falla = elongacion_disponible(E, tau_u, A, dL_req, sigma_y=sigma_y)

    # -----------------------------------------------
    # 3) "Construir" la tubería en 4 tramos (A->B->Falla->B'->A') en 2D
    x2D, y2D = calcular_tramos(phiA, phiB, L1, L2, Dy)

    # 4) Calcular fuerzas normales (axiales) N(x) a lo largo de cada tramo
    xAB, NAB, xBC, NBC, xABd, NABd, xBCd, NBCd = calcular_fuerzas_normales(phiA, phiB, N_falla, L1, L2, tau_u)

    # -----------------------------------------------
    # MOSTRAR RESULTADOS
    st.subheader("Resultados (Paso 1 Completo - Simplificado)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Rotación (phiA)** = {phiA_deg:.4f}°")
        st.write(f"**Rotación (phiB)** = {phiB_deg:.4f}°")
    with col2:
        st.write(f"**Tensión axial (sigma_ax)** = {sigma_ax * 1e-6:.3f} MPa")
        st.write(f"**N en la falla** = {N_falla * 1e-3:.2f} kN")
    with col3:
        st.write(f"**Elong. requerida** = {dL_req:.6f} m")
        st.write(f"Despl. falla: Dx={Dx:.3f} m, Dy={Dy:.3f} m")

    # ------------------------------------------------
    # GRAFICO 2D DEFORMADA + flechas fuerza normal
    fig2d, ax2d = plt.subplots(figsize=(7, 4))
    ax2d.plot(x2D, y2D, "b-o", label="Tubería (deformada)")

    # Para dibujar "flechas" (quiver) representando la fuerza normal axial en cada tramo
    # Tomamos algunos puntos en la tubería (solo lado izq. si se quiere) y dibujamos flechas horizontales:
    # Asumimos la dirección axial (aprox. tangente a la tubería). Para simplificar, usaremos flechas horizontales si la tubería ~ axis-x.

    # A->B
    # xAB: [0..L1], NAB: (fuerza axial)
    # para cada x, flecha en dirección +x si NAB>0
    skip = 5
    x_sel_AB = xAB[::skip]
    N_sel_AB = NAB[::skip]
    # "dx" de flecha ~ 0.5
    # la magnitud de la flecha la normalizamos
    scaleF = 1e-4  # factor "visual"
    dx_AB = np.sign(N_sel_AB) * 0.5
    dy_AB = np.zeros_like(dx_AB)

    # quiver -> x, y, U, V
    ax2d.quiver(x_sel_AB, phiA * x_sel_AB, dx_AB, dy_AB, color="red",
                angles="xy", scale_units="xy",
                scale=1 / scaleF, width=0.003, label="Fuerza Axial A->B")

    # B->Falla
    x_sel_BC = xBC[::skip]
    N_sel_BC = NBC[::skip]
    dx_BC = np.sign(N_sel_BC) * 0.5
    dy_BC = np.zeros_like(dx_BC)
    # La posicion en y2D => yB + phiA+phiB*(x-someRef)
    # Como simplificación, la base es yB + ...
    yBase_BC = phiA * L1 + (x_sel_BC) * (phiA + phiB) - (phiA + phiB) * 0  # ajustado
    ax2d.quiver(L1 + x_sel_BC, yBase_BC, dx_BC, dy_BC, color="green",
                angles="xy", scale_units="xy",
                scale=1 / scaleF, width=0.003, label="Fuerza Axial B->Falla")

    # (Podrías hacer algo similar para el lado derecho.)

    ax2d.set_xlabel("Longitud a lo largo del eje (m)")
    ax2d.set_ylabel("Desplazamiento transversal (m)")
    ax2d.set_title("Deformada (4 tramos) y fuerzas normales (flechas)")
    ax2d.grid(True)
    ax2d.legend()
    st.pyplot(fig2d)

    # ------------------------------------------------
    # GRAFICO 3D
    fig3d = plt.figure(figsize=(6, 4))
    ax3d = fig3d.add_subplot(111, projection='3d')
    # Para 3D, usamos x2D, y2D, z=0
    z2D = np.zeros_like(x2D)
    ax3d.plot(x2D, y2D, z2D, "b-o", label="Tubería 3D (z=0)")

    ax3d.set_title("Visualización 3D (Perspectiva)")
    ax3d.set_xlabel("Eje X (m)")
    ax3d.set_ylabel("Eje Y (m)")
    ax3d.set_zlabel("Eje Z (m)")

    max_range = max(L1 + L2, max(abs(y2D))) + 1
    ax3d.set_xlim(0, (L1 + L2) * 2 + 1)
    ax3d.set_ylim(-max_range / 2, max_range / 2)
    ax3d.set_zlim(-max_range / 2, max_range / 2)
    ax3d.legend()

    st.pyplot(fig3d)

    st.markdown("""
    ---
    **Nota**: Este *ejemplo extendido* del Paso 1 muestra 4 tramos, 
    calcula \(\Delta L_{\mathrm{req}}\) y la fuerza axial \((N)\) 
    según ecuaciones (16)–(20), y dibuja flechas 
    para visualizar la dirección de la fuerza normal.  
    Para una **solución idéntica** a la del artículo, 
    es necesario resolver completamente las ecuaciones 
    (12)–(15) (momento nulo, condición \(\delta_2(L_2)=D_y/2\), etc.) 
    con un método **iterativo** (Newton-Raphson) 
    que ajuste \(\varphi_A, \varphi_B, V_A\) y verifique 
    la compatibilidad axial \(\Delta L_{\mathrm{req}} = \Delta L_{\mathrm{av}}\).
    """)


# -----------------------------------------------------------
# EJECUCIÓN
# -----------------------------------------------------------
if __name__ == "__main__":
    main()
