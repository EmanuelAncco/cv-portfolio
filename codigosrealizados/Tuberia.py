import numpy as np

def calculate_pipeline_stress(
    # Propiedades de la Tubería
    D,          # Diámetro exterior (m)
    t,          # Espesor de pared (m)
    E,          # Módulo de Young del acero (Pa)
    nu,         # Coeficiente de Poisson del acero (adimensional)
    alpha_T,    # Coeficiente de expansión térmica lineal del acero (°C⁻¹)
    Sy,         # Límite de Fluencia Específico Mínimo (SMYS) del acero (Pa)

    # Propiedades del Suelo (H es la única usada directamente en cálculos de esfuerzo en este modelo simplificado)
    H,          # Profundidad desde la superficie hasta el eje de la tubería (m)

    # Cargas Operacionales
    p,          # Presión interna máxima de operación (Pa)
    delta_T,    # Diferencia de temperatura máxima entre operación e instalación (°C)

    # Carga de Tráfico (simplificada según documento)
    W_traffic,  # Carga por rueda o peso total del vehículo/equipo (N) - Usado como carga puntual simplificada
    A_contacto, # Área de contacto de la carga (m²) - Usado para calcular presión superficial (aunque no directamente en la fórmula final del documento)
    If,         # Factor de impacto (DAF) (adimensional)
    # Hc calculado a partir de H y D

    # Parámetros Sísmicos (TGD)
    PGV,        # Velocidad Pico del Suelo en la dirección de propagación (m/s)
    C,          # Velocidad aparente de propagación de la onda sísmica a lo largo de la tubería (m/s)
    alpha_seismic # Factor de relación deformación-velocidad (adimensional, 1.0 según documento)
):
    """
    Calcula los esfuerzos en una tubería enterrada bajo cargas combinadas.

    Basado en el modelo analítico simplificado descrito en el documento
    "Gasoductos_ Modelo Analítico 2021-2025_.docx" y la lógica de la interfaz.

    Args:
        D (float): Diámetro exterior de la tubería (m).
        t (float): Espesor de pared de la tubería (m).
        E (float): Módulo de Young del acero (Pa).
        nu (float): Coeficiente de Poisson del acero (adimensional).
        alpha_T (float): Coeficiente de expansión térmica lineal del acero (°C⁻¹).
        Sy (float): Límite de Fluencia Específico Mínimo (SMYS) del acero (Pa).
        H (float): Profundidad desde la superficie hasta el eje de la tubería (m).
        p (float): Presión interna máxima de operación (Pa).
        delta_T (float): Diferencia de temperatura máxima entre operación e instalación (°C).
        W_traffic (float): Carga por rueda o peso total del vehículo/equipo (N).
        A_contacto (float): Área de contacto de la carga (m²).
        If (float): Factor de impacto (DAF) (adimensional).
        PGV (float): Velocidad Pico del Suelo (m/s).
        C (float): Velocidad aparente de propagación de la onda sísmica (m/s).
        alpha_seismic (float): Factor de relación deformación-velocidad (adimensional).

    Returns:
        dict: Un diccionario que contiene los esfuerzos calculados y el ratio de esfuerzo.
    """

    # --- Ecuaciones del Modelo Analítico ---

    # Esfuerzo Circunferencial por Presión Interna (sigma_h)
    # Fuente: Ecuación de Barlow (Documento 5.4.3)
    sigma_h = (p * D) / (2 * t)

    # Esfuerzo Axial por Presión Interna (sigma_a_p)
    # Fuente: Efecto Poisson en tubería restringida (Documento 5.4.3)
    sigma_a_p = nu * sigma_h

    # Esfuerzo Axial por Cambio de Temperatura (sigma_a_T)
    # Fuente: Tubería restringida (Documento 5.4.3)
    sigma_a_T = E * alpha_T * delta_T

    # Esfuerzo Longitudinal por Carga de Tráfico (sigma_L_traf)
    # Basado en la formulación simplificada del documento (Boussinesq puntual + viga simplificada)
    Hc = H - D / 2.0 # Profundidad a la corona (m)

    # Presión vertical en la corona (Qd) - Usando Boussinesq simplificado para carga puntual
    # Fuente: Adaptado de Boussinesq/Newmark (simplificación del documento 5.5.2)
    # Nota: El documento usa W_traffic como carga puntual en el ejemplo
    if Hc <= 0:
        # Evitar división por cero o resultados no físicos para tuberías muy superficiales
        Qd = 0.0
    else:
         Qd = (3 * W_traffic) / (2 * np.pi * Hc**2)

    # Carga lineal Wt
    # Fuente: Adaptado del documento (If * Qd * D) (Documento 5.5.2)
    Wt = If * Qd * D # N/m

    # Longitud característica X y factor km (valores de ejemplo del documento 5.5.2)
    X = 2 * Hc # m
    km = 10    # adimensional (intermedio)

    # Módulo de sección aproximado Zpipe
    Zpipe = (np.pi * D**2 * t) / 4.0 # m^3

    sigma_L_traf = 0.0
    if km != 0 and Zpipe != 0:
         # Momento flector por tráfico (simplificado)
        M_traf = (Wt * X**2) / km # Nm
        # Esfuerzo longitudinal por tráfico
        # Fuente: Adaptado del documento (M_traf / Zpipe) (Documento 5.5.2)
        sigma_L_traf = M_traf / Zpipe # Pa


    # Esfuerzo Axial por TGD (sigma_a_w)
    # Fuente: Modelo de Newmark simplificado (Documento 5.4.3)
    sigma_a_w = 0.0
    if C != 0:
        sigma_a_w = E * alpha_seismic * (PGV / C)


    # Esfuerzo Longitudinal Total (sigma_L)
    # Fuente: Principio de superposición (Documento 5.4.3)
    sigma_L = sigma_a_p + sigma_a_T + sigma_L_traf + sigma_a_w

    # Esfuerzo Circunferencial Total (sigma_h_total)
    # Fuente: Simplificación del modelo (solo presión interna) (Documento 5.4.3)
    sigma_h_total = sigma_h

    # Esfuerzo Equivalente de Von Mises (sigma_VM)
    # Fuente: Criterio de fluencia de Von Mises (asumiendo tau=0) (Documento 5.4.3)
    sigma_VM = np.sqrt(sigma_L**2 - sigma_L * sigma_h_total + sigma_h_total**2)

    # Ratio de Esfuerzo
    # Fuente: Comparación con el límite de fluencia (Documento 5.4.3)
    ratio = float('inf') # Evitar división por cero si Sy es cero
    if Sy != 0:
        ratio = sigma_VM / Sy

    results = {
        "sigma_h": sigma_h,
        "sigma_a_p": sigma_a_p,
        "sigma_a_T": sigma_a_T,
        "sigma_L_traf": sigma_L_traf,
        "sigma_a_w": sigma_a_w,
        "sigma_L_total": sigma_L,
        "sigma_h_total": sigma_h_total,
        "sigma_VM": sigma_VM,
        "Ratio_VM_Sy": ratio
    }

    return results

# --- Valores por defecto (Caso Miraflores de la interfaz) ---

# Parámetros de la Tubería
default_pipe_params = {
    "D": 0.61,          # m (24 pulgadas)
    "t": 0.0095,        # m (espesor estándar)
    "E": 2.07e11,       # Pa (Módulo de Young del acero)
    "nu": 0.3,          # adimensional (Coeficiente de Poisson del acero)
    "alpha_T": 1.2e-5,  # °C⁻¹ (Coeficiente de expansión térmica lineal del acero)
    "Sy": 4.48e8,       # Pa (448 MPa, SMYS del acero API 5L X65)
}

# Propiedades del Suelo (solo H es relevante para los cálculos en este modelo)
default_soil_params = {
    "H": 1.5,   # m (Profundidad al eje)
}

# Cargas Operacionales
default_operational_params = {
    "p": 7e6,           # Pa (70 bar)
    "delta_T": -15,     # °C (Operación más fría que instalación)
}

# Carga de Tráfico
default_traffic_params = {
    "W_traffic": 35500, # N (Carga por rueda, 35.5 kN)
    "A_contacto": 0.1,  # m² (Área de contacto simplificada)
    "If": 1.5,          # adimensional (Factor de impacto)
}

# Parámetros Sísmicos (TGD)
default_seismic_params = {
    "PGV": 0.40, # m/s (Miraflores)
    "C": 800,    # m/s (Miraflores)
    "alpha_seismic": 1.0, # adimensional (Factor de relación deformación-velocidad)
}

# Combinar todos los parámetros por defecto
default_params = {
    **default_pipe_params,
    **default_soil_params,
    **default_operational_params,
    **default_traffic_params,
    **default_seismic_params
}

# --- Ejecutar el cálculo con los valores por defecto ---
print("--- Resultados con Valores por Defecto (Caso Miraflores) ---")

# Realizar los cálculos con los parámetros por defecto
results = calculate_pipeline_stress(**default_params)

# Imprimir resultados (convertir a MPa para mejor legibilidad)
print(f"σh (Presión): {results['sigma_h']/1e6:.1f} MPa")
print(f"σa,p (Axial Presión): {results['sigma_a_p']/1e6:.1f} MPa")
print(f"σa,T (Axial Temperatura): {results['sigma_a_T']/1e6:.1f} MPa")
print(f"σL,traf (Longitudinal Tráfico): {results['sigma_L_traf']/1e6:.1f} MPa")
print(f"σa,w (Axial TGD): {results['sigma_a_w']/1e6:.1f} MPa")
print(f"σL Total: {results['sigma_L_total']/1e6:.1f} MPa")
print(f"σh Total: {results['sigma_h_total']/1e6:.1f} MPa")
print(f"σVM (Von Mises): {results['sigma_VM']/1e6:.1f} MPa")
print(f"Ratio σVM/Sy: {results['Ratio_VM_Sy']:.2f}")
print("-" * 30)

# --- Ejemplo de cómo usar la función con otros valores (ej. La Molina) ---
print("--- Ejemplo con Valores de La Molina ---")

la_molina_params = {
    **default_params, # Empezar con los valores por defecto
    "PGV": 0.50, # m/s (La Molina)
    "C": 400,    # m/s (La Molina)
    # Otros parámetros de suelo específicos de La Molina (gamma, phi, etc.)
    # no se usan en los cálculos de esfuerzo directo en este modelo simplificado,
    # pero podrías incluirlos si la función calculate_pipeline_stress los usara.
    # Por ejemplo: "gamma": 18000, "phi": 35, ...
}

la_molina_results = calculate_pipeline_stress(**la_molina_params)

print(f"σh (Presión): {la_molina_results['sigma_h']/1e6:.1f} MPa")
print(f"σa,p (Axial Presión): {la_molina_results['sigma_a_p']/1e6:.1f} MPa")
print(f"σa,T (Axial Temperatura): {la_molina_results['sigma_a_T']/1e6:.1f} MPa")
print(f"σL,traf (Longitudinal Tráfico): {la_molina_results['sigma_L_traf']/1e6:.1f} MPa")
print(f"σa,w (Axial TGD): {la_molina_results['sigma_a_w']/1e6:.1f} MPa")
print(f"σL Total: {la_molina_results['sigma_L_total']/1e6:.1f} MPa")
print(f"σh Total: {la_molina_results['sigma_h_total']/1e6:.1f} MPa")
print(f"σVM (Von Mises): {la_molina_results['sigma_VM']/1e6:.1f} MPa")
print(f"Ratio σVM/Sy: {la_molina_results['Ratio_VM_Sy']:.2f}")
print("-" * 30)
