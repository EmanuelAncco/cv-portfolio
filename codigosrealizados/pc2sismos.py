"""
PROBLEMA 3 - ANÁLISIS DINÁMICO MODAL ESPECTRAL CON GRÁFICOS
Edificio de oficinas de concreto armado - 2 pisos
Ubicación: San Isidro, Lima - Suelo S2
Norma E.030 - Diseño Sismorresistente
Autor: Emanuel Ancco
Fecha: 2025-01-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor visualización
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

class Problema3_AnalisisModalEspectral:
    """
    Análisis Dinámico Modal Espectral - Edificio 2 pisos
    Datos proporcionados:
    - h1 = 3.0078 m, h2 = 2.65 m
    - Suelo tipo S2
    - Columnas empotradas en la base
    """

    def __init__(self):
        self.g = 9.81  # m/s² (gravedad)

        # Datos del problema
        self.h1 = 3.0078  # m
        self.h2 = 2.65    # m

        # Matriz de Masa (Tnf·s²/m)
        self.M = np.array([
            [17.8389, 0],
            [0, 13.8634]
        ])

        # Matriz de Rigidez (Tnf/m)
        self.K = np.array([
            [92178.4778, -54740.0498],
            [-54740.0498, 54740.0498]
        ])

        # Frecuencias circulares (s⁻¹)
        self.omega1 = 32.0017  # rad/s
        self.omega2 = 89.9538  # rad/s

        # Periodos (s)
        self.T1 = 0.1963  # s
        self.T2 = 0.0698  # s

        # Frecuencias cíclicas (Hz)
        self.f1 = 5.0932   # Hz
        self.f2 = 14.3166  # Hz

        # Vectores propios (modos de vibración normalizados)
        # Modo 1
        self.X1 = np.array([0.7406, 1.0])

        # Modo 2
        self.X2 = np.array([-1.0493, 1.0])

        # Masas generalizadas
        self.m1_star = 23.6488  # Tnf·s²/m
        self.m2_star = 33.5043  # Tnf·s²/m

        # Rigideces generalizadas
        self.K1_star = 24218.8856  # Tnf/m
        self.K2_star = 271106.1534  # Tnf/m

        # Masas participantes
        self.L1 = 27.0756  # Tnf·s²/m
        self.L2 = -4.8548  # Tnf·s²/m

        # Factores de participación modal
        self.r1 = 1.1449
        self.r2 = -0.1449

    def resolver_completo(self):
        """Resuelve el problema completo con todas las preguntas"""

        resultados = []
        resultados.append("="*100)
        resultados.append("PROBLEMA 3: ANÁLISIS DINÁMICO MODAL ESPECTRAL")
        resultados.append("Edificio de Oficinas - 2 Pisos - San Isidro, Lima")
        resultados.append("="*100)

        # Datos iniciales
        resultados.append("\n📋 DATOS DEL PROBLEMA:")
        resultados.append("-" * 100)
        resultados.append(f"Altura primer piso: h₁ = {self.h1} m")
        resultados.append(f"Altura segundo piso: h₂ = {self.h2} m")
        resultados.append(f"Altura total: H = {self.h1 + self.h2:.4f} m")
        resultados.append(f"Ubicación: San Isidro, Lima (Zona 4)")
        resultados.append(f"Tipo de suelo: S2")
        resultados.append(f"Material: Concreto armado")
        resultados.append(f"Condición: Columnas empotradas en la base")

        # Mostrar matrices
        resultados.append("\n📊 MATRICES DEL SISTEMA:")
        resultados.append("-" * 100)
        resultados.append("\n🔹 MATRIZ DE MASA [M] (Tnf·s²/m):")
        resultados.append(f"    ┌                              ┐")
        resultados.append(f"M = │ {self.M[0,0]:10.4f}    {self.M[0,1]:10.4f} │")
        resultados.append(f"    │ {self.M[1,0]:10.4f}    {self.M[1,1]:10.4f} │")
        resultados.append(f"    └                              ┘")

        resultados.append("\n🔹 MATRIZ DE RIGIDEZ [K] (Tnf/m):")
        resultados.append(f"    ┌                                      ┐")
        resultados.append(f"K = │ {self.K[0,0]:11.4f}   {self.K[0,1]:11.4f} │")
        resultados.append(f"    │ {self.K[1,0]:11.4f}   {self.K[1,1]:11.4f} │")
        resultados.append(f"    └                                      ┘")

        # Propiedades modales
        resultados.append("\n📈 PROPIEDADES MODALES:")
        resultados.append("-" * 100)

        resultados.append(f"\n🔹 MODO 1:")
        resultados.append(f"   ω₁ = {self.omega1:.4f} rad/s")
        resultados.append(f"   T₁ = 2π/ω₁ = 2π/{self.omega1:.4f} = {self.T1:.4f} s")
        resultados.append(f"   f₁ = 1/T₁ = 1/{self.T1:.4f} = {self.f1:.4f} Hz")
        resultados.append(f"   Vector propio X₁ = [{self.X1[0]:.4f}, {self.X1[1]:.4f}]ᵀ")

        resultados.append(f"\n🔹 MODO 2:")
        resultados.append(f"   ω₂ = {self.omega2:.4f} rad/s")
        resultados.append(f"   T₂ = 2π/ω₂ = 2π/{self.omega2:.4f} = {self.T2:.4f} s")
        resultados.append(f"   f₂ = 1/T₂ = 1/{self.T2:.4f} = {self.f2:.4f} Hz")
        resultados.append(f"   Vector propio X₂ = [{self.X2[0]:.4f}, {self.X2[1]:.4f}]ᵀ")

        # Pregunta a) ZUCS
        resultados.append("\n" + "="*100)
        resultados.append("a) CÁLCULO DE COEFICIENTES DE SITIO (ZUCS) - 2 puntos")
        resultados.append("="*100)

        zucs_results = self.calcular_ZUCS()
        resultados.extend(zucs_results)

        # Pregunta b) Factor R
        resultados.append("\n" + "="*100)
        resultados.append("b) CÁLCULO DEL FACTOR DE REDUCCIÓN R - 1 punto")
        resultados.append("="*100)

        r_results = self.calcular_factor_R()
        resultados.extend(r_results)

        # Pregunta c) Fuerzas inerciales
        resultados.append("\n" + "="*100)
        resultados.append("c) FUERZAS INERCIALES Y CORTANTE BASAL - 3 puntos")
        resultados.append("="*100)

        fuerzas_results = self.calcular_fuerzas_inerciales()
        resultados.extend(fuerzas_results)

        # Pregunta d) Derivas
        resultados.append("\n" + "="*100)
        resultados.append("d) DERIVAS DE PISO - 3 puntos")
        resultados.append("="*100)

        derivas_results = self.calcular_derivas()
        resultados.extend(derivas_results)

        return "\n".join(resultados)

    def calcular_ZUCS(self):
        """a) Cálculo de coeficientes de sitio (ZUCS) según E.030"""
        resultados = []

        resultados.append("\n📐 PASO 1: Identificar parámetros sísmicos")
        resultados.append("-" * 100)

        # Zona sísmica
        resultados.append("\n🔹 FACTOR DE ZONA (Z):")
        resultados.append("   Ubicación: San Isidro, Lima")
        resultados.append("   Lima se encuentra en Zona Sísmica 4 (sismicidad muy alta)")
        resultados.append("   Según Tabla N°1 de E.030:")
        Z = 0.45
        resultados.append(f"   ✓ Z = {Z}")

        # Factor de uso
        resultados.append("\n🔹 FACTOR DE USO (U):")
        resultados.append("   Edificación: Oficinas (uso común)")
        resultados.append("   Según Tabla N°5 de E.030:")
        resultados.append("   Categoría: C - Edificaciones comunes")
        U = 1.0
        resultados.append(f"   ✓ U = {U}")

        # Factor de suelo
        resultados.append("\n🔹 FACTOR DE SUELO (S) y PERIODOS Tp, TL:")
        resultados.append("   Tipo de suelo: S2 (Suelos intermedios)")
        resultados.append("   Según Tabla N°3 de E.030:")
        resultados.append("   Para suelo S2:")
        S = 1.05
        Tp = 0.6  # s
        TL = 2.0  # s
        resultados.append(f"   ✓ S = {S}")
        resultados.append(f"   ✓ Tp = {Tp} s (Periodo que define la plataforma del espectro)")
        resultados.append(f"   ✓ TL = {TL} s (Periodo que define inicio de zona descendente)")

        # Factor de amplificación sísmica C
        resultados.append("\n📐 PASO 2: Calcular Factor de Amplificación Sísmica (C)")
        resultados.append("-" * 100)

        resultados.append("\nSegún E.030, el factor C se calcula como:")
        resultados.append("   • Si T < Tp:        C = 2.5")
        resultados.append("   • Si Tp ≤ T ≤ TL:   C = 2.5 × (Tp/T)")
        resultados.append("   • Si T > TL:        C = 2.5 × (Tp×TL/T²)")

        # Para Modo 1
        resultados.append(f"\n🔹 MODO 1 (T₁ = {self.T1:.4f} s):")
        resultados.append(f"   Comparación: T₁ = {self.T1:.4f} s < Tp = {Tp} s")
        resultados.append(f"   Por lo tanto, usamos: C₁ = 2.5")
        C1 = 2.5
        resultados.append(f"   ✓ C₁ = {C1}")

        # Para Modo 2
        resultados.append(f"\n🔹 MODO 2 (T₂ = {self.T2:.4f} s):")
        resultados.append(f"   Comparación: T₂ = {self.T2:.4f} s < Tp = {Tp} s")
        resultados.append(f"   Por lo tanto, usamos: C₂ = 2.5")
        C2 = 2.5
        resultados.append(f"   ✓ C₂ = {C2}")

        # ZUCS
        resultados.append("\n📐 PASO 3: Calcular ZUCS (sin R)")
        resultados.append("-" * 100)

        ZUCS1 = Z * U * C1 * S
        ZUCS2 = Z * U * C2 * S

        resultados.append(f"\n🔹 MODO 1:")
        resultados.append(f"   ZUCS₁ = Z × U × C₁ × S")
        resultados.append(f"   ZUCS₁ = {Z} × {U} × {C1} × {S}")
        resultados.append(f"   ✓ ZUCS₁ = {ZUCS1:.6f}")

        resultados.append(f"\n🔹 MODO 2:")
        resultados.append(f"   ZUCS₂ = Z × U × C₂ × S")
        resultados.append(f"   ZUCS₂ = {Z} × {U} × {C2} × {S}")
        resultados.append(f"   ✓ ZUCS₂ = {ZUCS2:.6f}")

        resultados.append("\n✅ RESUMEN:")
        resultados.append(f"   Z = {Z} (Zona 4)")
        resultados.append(f"   U = {U} (Edificación común)")
        resultados.append(f"   C₁ = {C1} (T₁ < Tp)")
        resultados.append(f"   C₂ = {C2} (T₂ < Tp)")
        resultados.append(f"   S = {S} (Suelo S2)")
        resultados.append(f"   ZUCS₁ = {ZUCS1:.6f}")
        resultados.append(f"   ZUCS₂ = {ZUCS2:.6f}")

        # Guardar valores para uso posterior
        self.Z = Z
        self.U = U
        self.S = S
        self.Tp = Tp
        self.TL = TL
        self.C1 = C1
        self.C2 = C2
        self.ZUCS1 = ZUCS1
        self.ZUCS2 = ZUCS2

        return resultados

    def calcular_factor_R(self):
        """b) Cálculo del factor de reducción de fuerza sísmica R"""
        resultados = []

        resultados.append("\n📐 DETERMINACIÓN DEL FACTOR R")
        resultados.append("-" * 100)

        resultados.append("\n🔹 CARACTERÍSTICAS ESTRUCTURALES:")
        resultados.append("   • Material: Concreto armado")
        resultados.append("   • Número de pisos: 2")
        resultados.append("   • Columnas: Sección transversal igual")
        resultados.append("   • Condición: Empotradas en la base")
        resultados.append("   • Uso: Edificio de oficinas")

        resultados.append("\n📊 ANÁLISIS DEL SISTEMA ESTRUCTURAL:")
        resultados.append("-" * 100)

        resultados.append("\nSegún la información proporcionada:")
        resultados.append("   1. Todas las columnas son de igual sección → Sistema regular")
        resultados.append("   2. Columnas empotradas en base → Sistema de pórticos")
        resultados.append("   3. No se mencionan muros de corte o placas")
        resultados.append("   4. Edificio de 2 pisos (baja altura)")

        resultados.append("\n🔍 DETERMINACIÓN DEL SISTEMA:")
        resultados.append("\nSistema de Pórticos de Concreto Armado:")
        resultados.append("   • La resistencia sísmica está dada principalmente por los pórticos")
        resultados.append("   • No hay muros de corte que absorban más del 70% del cortante")
        resultados.append("   • Sistema estructural: PÓRTICOS")

        resultados.append("\n📋 TABLA N°7 DE E.030 - SISTEMAS ESTRUCTURALES:")
        resultados.append("-" * 100)
        resultados.append("   Sistema                                    | R básico")
        resultados.append("   ------------------------------------------|----------")
        resultados.append("   Pórticos de concreto armado               | R = 8")
        resultados.append("   Dual de concreto armado                   | R = 7")
        resultados.append("   Muros de concreto armado                  | R = 6")
        resultados.append("   Muros de ductilidad limitada              | R = 4")

        resultados.append("\n📐 VERIFICACIÓN DE IRREGULARIDADES:")
        resultados.append("-" * 100)

        resultados.append("\n🔹 IRREGULARIDAD EN ALTURA:")
        resultados.append("   • Edificio de solo 2 pisos")
        resultados.append("   • Columnas de igual sección en ambos niveles")
        resultados.append("   • No hay piso blando ni irregularidad de masa significativa")
        resultados.append("   ✓ Estructura REGULAR en altura")

        resultados.append("\n🔹 IRREGULARIDAD EN PLANTA:")
        resultados.append("   • No se menciona asimetría o torsión excesiva")
        resultados.append("   • Sistema simétrico de pórticos")
        resultados.append("   ✓ Estructura REGULAR en planta")

        resultados.append("\n✅ FACTOR R RESULTANTE:")
        resultados.append("-" * 100)

        R = 8.0

        resultados.append(f"\nSistema estructural: PÓRTICOS DE CONCRETO ARMADO")
        resultados.append(f"Regularidad: REGULAR (sin penalizaciones)")
        resultados.append(f"\n✓ R = {R}")

        resultados.append("\n📝 JUSTIFICACIÓN:")
        resultados.append("   El factor R = 8 se aplica porque:")
        resultados.append("   1. Es un sistema de pórticos de concreto armado")
        resultados.append("   2. La estructura es regular en altura y planta")
        resultados.append("   3. No hay factores de penalización por irregularidades")
        resultados.append("   4. Cumple con los requisitos de ductilidad del código")

        # Guardar para uso posterior
        self.R = R

        return resultados

    def calcular_fuerzas_inerciales(self):
        """c) Cálculo de fuerzas inerciales, cortante basal y comparación"""
        resultados = []

        resultados.append("\n📐 ANÁLISIS DINÁMICO MODAL ESPECTRAL")
        resultados.append("-" * 100)

        resultados.append("\n🔹 MÉTODO: Combinación Modal 25/75")
        resultados.append("   Según E.030, para estructuras regulares:")
        resultados.append("   Respuesta total = 0.25 × |Modo1 + Modo2| + 0.75 × √(Modo1² + Modo2²)")
        resultados.append("   Esto es: 0.25 × ABS + 0.75 × SRSS")

        # Aceleraciones espectrales
        resultados.append("\n📐 PASO 1: Calcular Aceleraciones Espectrales")
        resultados.append("-" * 100)

        Sa1 = (self.ZUCS1 / self.R) * self.g  # m/s²
        Sa2 = (self.ZUCS2 / self.R) * self.g  # m/s²

        resultados.append(f"\n🔹 MODO 1:")
        resultados.append(f"   Sa₁ = (ZUCS₁/R) × g")
        resultados.append(f"   Sa₁ = ({self.ZUCS1:.6f}/{self.R}) × {self.g} m/s²")
        resultados.append(f"   ✓ Sa₁ = {Sa1:.6f} m/s²")

        resultados.append(f"\n🔹 MODO 2:")
        resultados.append(f"   Sa₂ = (ZUCS₂/R) × g")
        resultados.append(f"   Sa₂ = ({self.ZUCS2:.6f}/{self.R}) × {self.g} m/s²")
        resultados.append(f"   ✓ Sa₂ = {Sa2:.6f} m/s²")

        # Fuerzas modales
        resultados.append("\n📐 PASO 2: Calcular Fuerzas Modales por Piso")
        resultados.append("-" * 100)

        resultados.append("\nLas fuerzas inerciales modales se calculan como:")
        resultados.append("   Fi,mode = mi × Xi,mode × Γmode × Sa_mode / g")

        # Masas
        m1 = self.M[0, 0]  # Piso 1
        m2 = self.M[1, 1]  # Piso 2

        # Fuerzas Modo 1
        F1_modo1_piso1 = m1 * self.X1[0] * self.r1 * Sa1 / self.g
        F1_modo1_piso2 = m2 * self.X1[1] * self.r1 * Sa1 / self.g

        resultados.append(f"\n🔹 FUERZAS MODO 1:")
        resultados.append(f"   Piso 1: F₁,₁ = {m1:.4f} × {self.X1[0]:.4f} × {self.r1:.4f} × {Sa1:.6f} / {self.g}")
        resultados.append(f"   ✓ F₁,₁ = {F1_modo1_piso1:.4f} Tnf")
        resultados.append(f"   Piso 2: F₂,₁ = {m2:.4f} × {self.X1[1]:.4f} × {self.r1:.4f} × {Sa1:.6f} / {self.g}")
        resultados.append(f"   ✓ F₂,₁ = {F1_modo1_piso2:.4f} Tnf")

        # Fuerzas Modo 2
        F2_modo2_piso1 = m1 * self.X2[0] * self.r2 * Sa2 / self.g
        F2_modo2_piso2 = m2 * self.X2[1] * self.r2 * Sa2 / self.g

        resultados.append(f"\n🔹 FUERZAS MODO 2:")
        resultados.append(f"   Piso 1: F₁,₂ = {m1:.4f} × {self.X2[0]:.4f} × {self.r2:.4f} × {Sa2:.6f} / {self.g}")
        resultados.append(f"   ✓ F₁,₂ = {F2_modo2_piso1:.4f} Tnf")
        resultados.append(f"   Piso 2: F₂,₂ = {m2:.4f} × {self.X2[1]:.4f} × {self.r2:.4f} × {Sa2:.6f} / {self.g}")
        resultados.append(f"   ✓ F₂,₂ = {F2_modo2_piso2:.4f} Tnf")

        # Combinación 25/75
        resultados.append("\n📐 PASO 3: Aplicar Combinación Modal 25/75")
        resultados.append("-" * 100)

        # Piso 1
        abs_piso1 = abs(F1_modo1_piso1) + abs(F2_modo2_piso1)
        srss_piso1 = np.sqrt(F1_modo1_piso1**2 + F2_modo2_piso1**2)
        F_total_piso1 = 0.25 * abs_piso1 + 0.75 * srss_piso1

        resultados.append(f"\n🔹 PISO 1:")
        resultados.append(f"   ABS₁ = {abs_piso1:.4f} Tnf")
        resultados.append(f"   SRSS₁ = {srss_piso1:.4f} Tnf")
        resultados.append(f"   ✓ F₁,total = {F_total_piso1:.4f} Tnf")

        # Piso 2
        abs_piso2 = abs(F1_modo1_piso2) + abs(F2_modo2_piso2)
        srss_piso2 = np.sqrt(F1_modo1_piso2**2 + F2_modo2_piso2**2)
        F_total_piso2 = 0.25 * abs_piso2 + 0.75 * srss_piso2

        resultados.append(f"\n🔹 PISO 2:")
        resultados.append(f"   ABS₂ = {abs_piso2:.4f} Tnf")
        resultados.append(f"   SRSS₂ = {srss_piso2:.4f} Tnf")
        resultados.append(f"   ✓ F₂,total = {F_total_piso2:.4f} Tnf")

        # Cortantes
        V_piso2 = F_total_piso2
        V_piso1 = F_total_piso1 + V_piso2
        V_basal = V_piso1

        resultados.append("\n📐 PASO 4: Calcular Cortantes")
        resultados.append("-" * 100)
        resultados.append(f"   V₂ = {V_piso2:.4f} Tnf")
        resultados.append(f"   V₁ = {V_piso1:.4f} Tnf")
        resultados.append(f"   ✓ V_basal = {V_basal:.4f} Tnf")

        # Comparación
        W_total = (m1 + m2) * self.g
        V_estatico = (self.ZUCS1 / self.R) * W_total
        V_minimo = 0.80 * V_estatico

        resultados.append("\n📐 PASO 5: Comparación con 80% Estático")
        resultados.append("-" * 100)
        resultados.append(f"   V_estático = {V_estatico:.4f} Tnf")
        resultados.append(f"   V_mínimo (80%) = {V_minimo:.4f} Tnf")
        resultados.append(f"   V_dinámico = {V_basal:.4f} Tnf")

        if V_basal >= V_minimo:
            resultados.append(f"   ✓ CUMPLE: {V_basal:.4f} ≥ {V_minimo:.4f}")
        else:
            resultados.append(f"   ✗ NO CUMPLE: Requiere escalar")

        # Guardar
        self.Sa1 = Sa1
        self.Sa2 = Sa2
        self.F_modo1_piso1 = F1_modo1_piso1
        self.F_modo1_piso2 = F1_modo1_piso2
        self.F_modo2_piso1 = F2_modo2_piso1
        self.F_modo2_piso2 = F2_modo2_piso2
        self.F_piso1 = F_total_piso1
        self.F_piso2 = F_total_piso2
        self.V_basal = V_basal
        self.V_piso1 = V_piso1
        self.V_piso2 = V_piso2

        return resultados

    def calcular_derivas(self):
        """d) Cálculo de derivas por piso"""
        resultados = []

        resultados.append("\n📐 CÁLCULO DE DERIVAS")
        resultados.append("-" * 100)

        # Desplazamientos espectrales
        Sd1 = self.Sa1 * (self.T1**2) / (4 * np.pi**2)
        Sd2 = self.Sa2 * (self.T2**2) / (4 * np.pi**2)

        resultados.append(f"\n📐 PASO 1: Desplazamientos Espectrales")
        resultados.append(f"   Sd₁ = {Sd1*1000:.4f} mm")
        resultados.append(f"   Sd₂ = {Sd2*1000:.4f} mm")

        # Desplazamientos modales
        delta1_piso1_modo1 = self.X1[0] * self.r1 * Sd1 * 1000
        delta1_piso2_modo1 = self.X1[1] * self.r1 * Sd1 * 1000
        delta1_piso1_modo2 = self.X2[0] * self.r2 * Sd2 * 1000
        delta1_piso2_modo2 = self.X2[1] * self.r2 * Sd2 * 1000

        resultados.append(f"\n📐 PASO 2: Desplazamientos por Modo")
        resultados.append(f"   Modo 1 - Piso 1: {delta1_piso1_modo1:.4f} mm")
        resultados.append(f"   Modo 1 - Piso 2: {delta1_piso2_modo1:.4f} mm")
        resultados.append(f"   Modo 2 - Piso 1: {delta1_piso1_modo2:.4f} mm")
        resultados.append(f"   Modo 2 - Piso 2: {delta1_piso2_modo2:.4f} mm")

        # Combinación 25/75
        abs_desp_piso1 = abs(delta1_piso1_modo1) + abs(delta1_piso1_modo2)
        srss_desp_piso1 = np.sqrt(delta1_piso1_modo1**2 + delta1_piso1_modo2**2)
        delta_total_piso1 = 0.25 * abs_desp_piso1 + 0.75 * srss_desp_piso1

        abs_desp_piso2 = abs(delta1_piso2_modo1) + abs(delta1_piso2_modo2)
        srss_desp_piso2 = np.sqrt(delta1_piso2_modo1**2 + delta1_piso2_modo2**2)
        delta_total_piso2 = 0.25 * abs_desp_piso2 + 0.75 * srss_desp_piso2

        resultados.append(f"\n📐 PASO 3: Desplazamientos Totales (25/75)")
        resultados.append(f"   δ₁ = {delta_total_piso1:.4f} mm")
        resultados.append(f"   δ₂ = {delta_total_piso2:.4f} mm")

        # Derivas
        deriva_piso1 = delta_total_piso1
        deriva_piso2 = delta_total_piso2 - delta_total_piso1

        resultados.append(f"\n📐 PASO 4: Derivas de Entrepiso")
        resultados.append(f"   Δ₁ = {deriva_piso1:.4f} mm")
        resultados.append(f"   Δ₂ = {deriva_piso2:.4f} mm")

        # Distorsiones
        h1_mm = self.h1 * 1000
        h2_mm = self.h2 * 1000

        distorsion_piso1 = deriva_piso1 / h1_mm
        distorsion_piso2 = deriva_piso2 / h2_mm

        factor_amplif = 0.75 * self.R
        distorsion_ine_piso1 = factor_amplif * distorsion_piso1
        distorsion_ine_piso2 = factor_amplif * distorsion_piso2

        resultados.append(f"\n📐 PASO 5: Distorsiones Inelásticas")
        resultados.append(f"   γ₁,ine = {distorsion_ine_piso1:.6f}")
        resultados.append(f"   γ₂,ine = {distorsion_ine_piso2:.6f}")

        limite = 0.007
        resultados.append(f"\n📐 PASO 6: Verificación (Límite = {limite})")

        if distorsion_ine_piso1 <= limite:
            resultados.append(f"   Piso 1: ✓ CUMPLE")
        else:
            resultados.append(f"   Piso 1: ✗ NO CUMPLE")

        if distorsion_ine_piso2 <= limite:
            resultados.append(f"   Piso 2: ✓ CUMPLE")
        else:
            resultados.append(f"   Piso 2: ✗ NO CUMPLE")

        # Guardar para gráficos
        self.delta_total_piso1 = delta_total_piso1
        self.delta_total_piso2 = delta_total_piso2
        self.deriva_piso1 = deriva_piso1
        self.deriva_piso2 = deriva_piso2
        self.distorsion_ine_piso1 = distorsion_ine_piso1
        self.distorsion_ine_piso2 = distorsion_ine_piso2

        return resultados

    def generar_graficos_completos(self):
        """Genera todos los gráficos del análisis"""

        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

        # Título principal
        fig.suptitle('PROBLEMA 3: ANÁLISIS DINÁMICO MODAL ESPECTRAL\nEdificio 2 Pisos - San Isidro, Lima',
                    fontsize=16, fontweight='bold', y=0.98)

        # 1. Espectro de Respuesta
        ax1 = fig.add_subplot(gs[0, :2])
        self.graficar_espectro(ax1)

        # 2. Modos de vibración
        ax2 = fig.add_subplot(gs[0, 2:])
        self.graficar_modos_vibracion(ax2)

        # 3. Fuerzas por piso
        ax3 = fig.add_subplot(gs[1, :2])
        self.graficar_fuerzas(ax3)

        # 4. Cortantes por piso
        ax4 = fig.add_subplot(gs[1, 2:])
        self.graficar_cortantes(ax4)

        # 5. Desplazamientos
        ax5 = fig.add_subplot(gs[2, :2])
        self.graficar_desplazamientos(ax5)

        # 6. Derivas
        ax6 = fig.add_subplot(gs[2, 2:])
        self.graficar_derivas(ax6)

        # 7. Distorsiones
        ax7 = fig.add_subplot(gs[3, :2])
        self.graficar_distorsiones(ax7)

        # 8. Edificio deformado
        ax8 = fig.add_subplot(gs[3, 2:])
        self.graficar_edificio_deformado(ax8)

        plt.savefig('Problema3_Graficos_Completos.png', dpi=300, bbox_inches='tight')
        print("✓ Gráficos guardados en: Problema3_Graficos_Completos.png")

        plt.show()

    def graficar_espectro(self, ax):
        """Gráfica del espectro de respuesta"""
        T_range = np.linspace(0.01, 2.0, 1000)
        Sa_values = []

        for T in T_range:
            if T < self.Tp:
                C = 2.5
            elif self.Tp <= T <= self.TL:
                C = 2.5 * (self.Tp / T)
            else:
                C = 2.5 * (self.Tp * self.TL / T**2)

            Sa = (self.Z * self.U * C * self.S / self.R) * self.g
            Sa_values.append(Sa)

        ax.plot(T_range, Sa_values, 'b-', linewidth=2, label='Espectro E.030')
        ax.plot(self.T1, self.Sa1, 'ro', markersize=10, label=f'Modo 1 (T={self.T1:.4f}s)')
        ax.plot(self.T2, self.Sa2, 'go', markersize=10, label=f'Modo 2 (T={self.T2:.4f}s)')

        ax.axvline(self.Tp, color='orange', linestyle='--', alpha=0.5, label=f'Tp={self.Tp}s')
        ax.axvline(self.TL, color='red', linestyle='--', alpha=0.5, label=f'TL={self.TL}s')

        ax.set_xlabel('Periodo T (s)')
        ax.set_ylabel('Aceleración Espectral Sa (m/s²)')
        ax.set_title('Espectro de Respuesta - Suelo S2')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def graficar_modos_vibracion(self, ax):
        """Gráfica de modos de vibración"""
        alturas = [0, self.h1, self.h1 + self.h2]

        # Modo 1
        desp_modo1 = [0, self.X1[0], self.X1[1]]
        ax.plot(desp_modo1, alturas, 'o-', linewidth=2, markersize=8,
               label='Modo 1', color='blue')

        # Modo 2
        desp_modo2 = [0, self.X2[0], self.X2[1]]
        ax.plot(desp_modo2, alturas, 's-', linewidth=2, markersize=8,
               label='Modo 2', color='red')

        # Línea vertical en 0
        ax.axvline(0, color='black', linestyle='-', linewidth=1)

        ax.set_xlabel('Amplitud Normalizada')
        ax.set_ylabel('Altura (m)')
        ax.set_title('Modos de Vibración')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def graficar_fuerzas(self, ax):
        """Gráfica de fuerzas por piso"""
        pisos = ['Piso 1', 'Piso 2']
        x = np.arange(len(pisos))
        width = 0.25

        modo1 = [self.F_modo1_piso1, self.F_modo1_piso2]
        modo2 = [self.F_modo2_piso1, self.F_modo2_piso2]
        total = [self.F_piso1, self.F_piso2]

        ax.bar(x - width, modo1, width, label='Modo 1', color='skyblue')
        ax.bar(x, modo2, width, label='Modo 2', color='lightcoral')
        ax.bar(x + width, total, width, label='Total (25/75)', color='gold')

        ax.set_xlabel('Nivel')
        ax.set_ylabel('Fuerza (Tnf)')
        ax.set_title('Fuerzas Inerciales por Piso')
        ax.set_xticks(x)
        ax.set_xticklabels(pisos)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def graficar_cortantes(self, ax):
        """Gráfica de cortantes por piso"""
        pisos = ['Piso 1', 'Piso 2', 'Basal']
        cortantes = [self.V_piso1, self.V_piso2, self.V_basal]
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        bars = ax.barh(pisos, cortantes, color=colors, edgecolor='black', linewidth=1.5)

        # Añadir valores
        for bar, val in zip(bars, cortantes):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f} Tnf', va='center', fontweight='bold')

        ax.set_xlabel('Cortante (Tnf)')
        ax.set_title('Cortantes por Piso')
        ax.grid(True, alpha=0.3, axis='x')

    def graficar_desplazamientos(self, ax):
        """Gráfica de desplazamientos"""
        alturas = [0, self.h1, self.h1 + self.h2]
        desplaz = [0, self.delta_total_piso1, self.delta_total_piso2]

        ax.plot(desplaz, alturas, 'o-', linewidth=3, markersize=10,
               color='purple', label='Desplazamientos')

        # Añadir valores
        ax.text(self.delta_total_piso1 + 0.1, self.h1,
               f'{self.delta_total_piso1:.2f} mm', fontsize=9)
        ax.text(self.delta_total_piso2 + 0.1, self.h1 + self.h2,
               f'{self.delta_total_piso2:.2f} mm', fontsize=9)

        ax.set_xlabel('Desplazamiento (mm)')
        ax.set_ylabel('Altura (m)')
        ax.set_title('Desplazamientos Laterales')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def graficar_derivas(self, ax):
        """Gráfica de derivas"""
        pisos = ['Piso 1', 'Piso 2']
        derivas = [self.deriva_piso1, self.deriva_piso2]
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(pisos, derivas, color=colors, edgecolor='black', linewidth=1.5)

        for bar, val in zip(bars, derivas):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                   f'{val:.2f} mm', ha='center', fontweight='bold')

        ax.set_ylabel('Deriva (mm)')
        ax.set_title('Derivas de Entrepiso')
        ax.grid(True, alpha=0.3, axis='y')

    def graficar_distorsiones(self, ax):
        """Gráfica de distorsiones"""
        pisos = ['Piso 1', 'Piso 2']
        distorsiones = [self.distorsion_ine_piso1 * 1000,
                       self.distorsion_ine_piso2 * 1000]
        limite = 0.007 * 1000

        colors = ['green' if d <= limite else 'red' for d in distorsiones]
        bars = ax.bar(pisos, distorsiones, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # Línea de límite
        ax.axhline(limite, color='red', linestyle='--', linewidth=2,
                  label=f'Límite E.030 ({limite:.1f}‰)')

        # Valores
        for bar, val in zip(bars, distorsiones):
            status = '✓' if val <= limite else '✗'
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                   f'{val:.2f}‰\n{status}', ha='center', fontweight='bold')

        ax.set_ylabel('Distorsión Inelástica (‰)')
        ax.set_title('Distorsiones Inelásticas vs Límite E.030')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def graficar_edificio_deformado(self, ax):
        """Gráfica del edificio deformado"""
        # Escala de deformación
        escala = 50

        # Posiciones originales
        x_orig = [0, 0, 0]
        y_orig = [0, self.h1, self.h1 + self.h2]

        # Posiciones deformadas
        x_def = [0,
                self.delta_total_piso1 * escala / 1000,
                self.delta_total_piso2 * escala / 1000]

        # Edificio original (línea punteada)
        ax.plot(x_orig, y_orig, 'k--', linewidth=2, alpha=0.5,
               label='Original')

        # Edificio deformado
        ax.plot(x_def, y_orig, 'r-', linewidth=3, marker='o', markersize=10,
               label=f'Deformado (×{escala})')

        # Columnas
        for i in range(len(y_orig)):
            ax.plot([x_orig[i], x_def[i]], [y_orig[i], y_orig[i]],
                   'b--', alpha=0.3)

        # Niveles
        for y in y_orig:
            ax.axhline(y, color='gray', linestyle=':', alpha=0.3)

        ax.set_xlabel('Desplazamiento (m)')
        ax.set_ylabel('Altura (m)')
        ax.set_title('Configuración Deformada del Edificio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')


# ============================================================================
# EJECUCIÓN COMPLETA
# ============================================================================

def main():
    """Función principal"""
    print("\n" + "="*100)
    print(" "*30 + "PROBLEMA 3 - 9 PUNTOS")
    print(" "*20 + "ANÁLISIS DINÁMICO MODAL ESPECTRAL")
    print("="*100)

    # Crear instancia
    problema = Problema3_AnalisisModalEspectral()

    # Resolver problema completo
    print("\n🔄 Resolviendo problema completo...\n")
    resultados = problema.resolver_completo()

    # Mostrar resultados
    print(resultados)

    # Guardar en archivo
    with open("Problema3_Solucion_Completa.txt", "w", encoding="utf-8") as f:
        f.write(resultados)

    print("\n" + "="*100)
    print("✓ Solución guardada en: Problema3_Solucion_Completa.txt")
    print("="*100)

    # Generar gráficos
    print("\n🎨 Generando gráficos completos...")
    problema.generar_graficos_completos()

    print("\n" + "="*100)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*100)
    print("\nArchivos generados:")
    print("  1. Problema3_Solucion_Completa.txt")
    print("  2. Problema3_Graficos_Completos.png")
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()