import numpy as np


def compute_pipeline_parameters(D, t, E1, E2, sigma_y, tau_u, Delta_x, L1, L2, qu, Kspr):
    """
    Calcula las fuerzas axiales Nc, NB y NA, desplazamientos laterales y momentos para una tubería enterrada con juntas flexibles.
    """
    # Área de la sección transversal
    A = np.pi * (D - t) * t

    # Momento de inercia
    I = (np.pi / 64) * (D ** 4 - (D - 2 * t) ** 4)

    # Longitud anclada (Lanch) según ecuación (18)
    Lanch = (sigma_y * A) / tau_u

    # Elongación requerida (Delta_Lreq) según ecuación (16)
    Delta_Lreq = Delta_x + 2 * (L1 / np.cos(0) - L1) + 2 * (L2 / np.cos(0) - L2)

    # Elongación disponible en la fase elástica según ecuación (19)
    Delta_Lav = (sigma_y ** 2 * A) / (E1 * tau_u)

    if Delta_Lreq <= Delta_Lav:
        # Estrés axial en fase elástica (Ecuación 20)
        sigma_alpha = np.sqrt(E1 * tau_u * Delta_Lreq / A)
    else:
        # Estrés axial en fase plástica (Ecuación 23)
        sigma_alpha = (sigma_y * (E1 - E2) + np.sqrt(
            sigma_y ** 2 * (E2 ** 2 - E1 * E2) + E1 ** 2 * E2 * Delta_Lreq * tau_u * A)) / E1

    # Cálculo de las fuerzas axiales según ecuaciones (24a) y (24b)
    NB = sigma_alpha * A
    NC = NB - (L1 * tau_u)
    NA = NC - (L2 * tau_u)

    # Cálculo de los momentos máximos en los tramos AB y BC
    M_AB = (qu * L1 ** 2) / 8
    M_BC = (qu * L2 ** 2) / 8

    # Cálculo de los desplazamientos laterales
    delta_1 = (M_AB * L1 ** 2) / (2 * E1 * I)
    delta_2 = (M_BC * L2 ** 2) / (2 * E1 * I)

    # Cálculo de la reacción del suelo en L1
    qu_star = min(qu, Kspr * L1 * np.tan(Delta_Lreq / (L1 + L2)))

    return NB, NC, NA, M_AB, M_BC, delta_1, delta_2, qu_star


# Parámetros del ejemplo del Apéndice A
D = 0.762  # Diámetro de la tubería en metros
t = 0.0125  # Espesor de la tubería en metros
E1 = 210e9  # Módulo de Young elástico en Pascales
E2 = 1.088e9  # Módulo de Young plástico en Pascales
sigma_y = 490e6  # Esfuerzo de fluencia en Pascales
tau_u = 22.75e3  # Fuerza de fricción axial en N/m
Delta_x = 1.5  # Desplazamiento horizontal en metros
L1 = 8  # Longitud entre juntas en metros
L2 = 8  # Longitud entre juntas en metros
qu = 134.45e3  # Fuerza lateral del suelo en N/m
Kspr = 1e6  # Rigidez del resorte del suelo en N/m

NB, NC, NA, M_AB, M_BC, delta_1, delta_2, qu_star = compute_pipeline_parameters(D, t, E1, E2, sigma_y, tau_u, Delta_x,
                                                                                L1, L2, qu, Kspr)

print(f"Fuerza axial NB: {NB:.2f} N")
print(f"Fuerza axial NC: {NC:.2f} N")
print(f"Fuerza axial NA: {NA:.2f} N")
print(f"Momento máximo M_AB: {M_AB:.2f} Nm")
print(f"Momento máximo M_BC: {M_BC:.2f} Nm")
print(f"Desplazamiento lateral delta_1: {delta_1:.6f} m")
print(f"Desplazamiento lateral delta_2: {delta_2:.6f} m")
print(f"Reacción del suelo en L1 (qu_star): {qu_star:.2f} N/m")
