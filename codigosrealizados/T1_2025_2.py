# ==============================================================================
# TALLER DE CLASIFICACIÓN CON REDES NEURONALES - IDI UPC
# Problema: Predicción de la Calidad del Concreto
# ==============================================================================

# Paso 1: Importar las herramientas necesarias
# ------------------------------------------------------------------------------
# Le pedimos a Python que cargue nuestras herramientas de trabajo.
# TensorFlow/Keras: para construir y entrenar nuestra red neuronal.
# Scikit-learn: para dividir nuestros datos y **NUEVO: para escalar nuestros datos**.
# Matplotlib: para visualizar nuestros resultados como ingenieros.
# Numpy: para manejar los datos numéricos de forma eficiente.
# ------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # <-- NUEVA HERRAMIENTA IMPORTADA
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

print("Librerías cargadas. Listos para empezar.")


# ==============================================================================
# Paso 2: El Generador de Datos Sintéticos
# ------------------------------------------------------------------------------
# Esta función simula los resultados de un laboratorio de materiales.
# Genera datos de mezclas de concreto, algunas que fallan (Clase 0)
# y otras que pasan (Clase 1) la prueba de compresión.
# Cada equipo recibirá datos con características distintas.
# ------------------------------------------------------------------------------
def generar_datos_concreto(n_muestras, centros, std_dev, balance):
    """
    Genera un dataset sintético para el problema de clasificación de concreto.
    - n_muestras: Número total de puntos de datos.
    - centros: Coordenadas de los centroides para las clases.
    - std_dev: Desviación estándar (dispersión de los datos).
    - balance: Proporción de muestras para la primera clase (e.g., [0.5, 0.5] para balanceado).
    """
    X, y = make_blobs(n_samples=n_muestras, centers=centros, cluster_std=std_dev,
                      random_state=42, n_features=2)

    # Simular un desbalance si es necesario
    if balance[0] != 0.5:
        # Esta es una forma simple de crear desbalance
        clase_0_idx = np.where(y == 0)[0]
        clase_1_idx = np.where(y == 1)[0]
        n_clase_0 = int(n_muestras * balance[0])
        n_clase_1 = n_muestras - n_clase_0

        idx_0_sample = np.random.choice(clase_0_idx, n_clase_0, replace=False)
        idx_1_sample = np.random.choice(clase_1_idx, n_clase_1, replace=False)

        indices = np.concatenate([idx_0_sample, idx_1_sample])
        X = X[indices, :]
        y = y[indices]

    return X, y

# ==============================================================================
# Paso 3: Visualización y Definición del Modelo
# ------------------------------------------------------------------------------
# Funciones de ayuda para graficar nuestros datos y los resultados.
# También definimos la arquitectura de nuestra red neuronal aquí.
# Es un modelo simple, pero potente para este problema.
# ------------------------------------------------------------------------------

def visualizar_datos(X, y, titulo):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Falla (Clase 0)')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Pasa (Clase 1)')
    plt.title(titulo)
    plt.xlabel('Característica 1 (Escalada)')
    plt.ylabel('Característica 2 (Escalada)')
    plt.legend()
    plt.grid(True)
    plt.show()

def crear_modelo():
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(2,)), # Capa oculta 1
        keras.layers.Dense(8, activation='relu'),                  # Capa oculta 2
        keras.layers.Dense(1, activation='sigmoid')                # Capa de salida
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def visualizar_historia(historia):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(historia.history['accuracy'], label='Precisión (Entrenamiento)')
    plt.plot(historia.history['val_accuracy'], label='Precisión (Validación)')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Precisión a lo largo del Entrenamiento')

    plt.subplot(1, 2, 2)
    plt.plot(historia.history['loss'], label='Pérdida (Entrenamiento)')
    plt.plot(historia.history['val_loss'], label='Pérdida (Validación)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdida a lo largo del Entrenamiento')
    plt.show()

# ==============================================================================
# INSTRUCCIONES PARA LA SESIÓN
# ==============================================================================
#
# 1. Ejecuten todas las celdas de código hasta este punto.
# 2. Cada equipo debe ir a su sección designada más abajo.
# 3. NO ejecuten el código de otros equipos.
# 4. Dentro de su sección:
#    a. Lean los comentarios para entender su desafío específico.
#    b. Ejecuten el código para generar y visualizar sus datos.
#    c. **NUEVO**: Observen el bloque de código que escala los datos.
#    d. Ejecuten el código para entrenar el modelo. Verán el progreso en vivo.
#    e. Comparen los nuevos gráficos con los anteriores. ¿Ven la diferencia?
#    f. Ejecuten la evaluación final y anoten su precisión en el set de prueba.
#
# ¡QUE COMIENCE EL TALLER!
#
# ==============================================================================


# ==============================================================================
# SECCIÓN DEL EQUIPO 1: El Caso de Alta Calidad (Realista)
# ------------------------------------------------------------------------------
# Su desafío: Tienen datos de laboratorio de alta calidad, pero realistas.
# Las clases están bien definidas, pero con una mínima superposición.
# Su modelo debería alcanzar una precisión muy alta, pero no un 100% perfecto.
# ------------------------------------------------------------------------------
print("\n--- INICIANDO TRABAJO DEL EQUIPO 1 ---")
# Generación de datos con un ligero traslape para ser más realista
X1, y1 = generar_datos_concreto(n_muestras=300, centros=[(0.4, 2), (0.6, 4)], std_dev=0.1, balance=[0.5, 0.5])

# División de datos: 70% entrenamiento, 15% validación, 15% prueba
X1_train, X1_temp, y1_train, y1_temp = train_test_split(X1, y1, test_size=0.3, random_state=42)
X1_val, X1_test, y1_val, y1_test = train_test_split(X1_temp, y1_temp, test_size=0.5, random_state=42)

# **NUEVO Y CRÍTICO**: Escalando los datos
print("\nEscalando los datos del Equipo 1...")
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train) # Aprender la escala SÓLO de los datos de entrenamiento
X1_val_scaled = scaler1.transform(X1_val)         # Aplicar la misma escala a la validación
X1_test_scaled = scaler1.transform(X1_test)       # Aplicar la misma escala a la prueba
visualizar_datos(X1_train_scaled, y1_train, "Equipo 1: Datos Escalados (Alta Calidad)")

# Crear y entrenar el modelo con los datos escalados
modelo1 = crear_modelo()
print("\nEntrenando Modelo del Equipo 1 (Epoch por Epoch)...")
historia1 = modelo1.fit(X1_train_scaled, y1_train, epochs=30, validation_data=(X1_val_scaled, y1_val), verbose=1)
print("Entrenamiento completo.")
visualizar_historia(historia1)

# Evaluación final
loss1, acc1 = modelo1.evaluate(X1_test_scaled, y1_test)
print(f"\nPrecisión final del Equipo 1 en datos de prueba: {acc1*100:.2f}%")
print("--- FIN DEL TRABAJO DEL EQUIPO 1 ---")
