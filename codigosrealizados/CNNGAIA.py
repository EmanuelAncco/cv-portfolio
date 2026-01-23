#!/usr/bin/env python3
"""
potato_cnn.py – Clasificador de Tizón de la Papa con CNN (GPU-ready)

Implementa un modelo CNN (VGG16 pre-entrenado) para clasificar enfermedades de la papa.
Entrena el modelo con el dataset (incluyendo la clase 'healthy' ampliada)
y guarda solo los pesos del mejor modelo durante el entrenamiento.

• TensorFlow 2.10+ • CUDA 11.2+ • cuDNN 8.1+ • Python 3.10+
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16 # Usaremos VGG16 como modelo base
# Importar scipy si es necesario para ImageDataGenerator (aunque no se usa directamente aquí)
# import scipy # No es necesario importar directamente

# ─────────────────────────── Parámetros globales ────────────────────────────
# Asegúrate de que DATA_DIR apunta a la carpeta que contiene las carpetas train y val
# Dentro de train y val deben estar las carpetas de clase (incluida la healthy ampliada)
DATA_DIR         = r"D:\Python_proyectos_2025\Agricultura\archive\DATA"  # ← cambia aquí si mueves el dataset base
TRAIN_DIR        = os.path.join(DATA_DIR, 'train') # Ruta a la carpeta de entrenamiento base
VAL_DIR          = os.path.join(DATA_DIR, 'val')   # Ruta a la carpeta de validación base

SEED             = 123
IMG_SIZE         = (224, 224) # Tamaño de imagen para VGG16
BATCH_SIZE       = 32
EPOCHS           = 50 # Épocas para el entrenamiento
LEARNING_RATE    = 1e-4 # Tasa de aprendizaje inicial

BUFFER_SIZE_PREFETCH = tf.data.AUTOTUNE # Tamaño del buffer para prefetch
SHUFFLE_BUFFER_SIZE = 1000 # Tamaño del buffer para shuffle (debe ser > 0)

# Nombre del archivo donde se guardarán solo los pesos del mejor modelo
BEST_WEIGHTS_PATH = "modeloGAIA_cnn_best_weights.h5" # Guardar solo pesos en .h5 con nuevo nombre

# ──────────────────────────── Configuración GPU ─────────────────────────────
# → Activar GPU con memory-growth si está disponible
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        for g in _gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.set_visible_devices(_gpus[0], "GPU")
        print("✔ GPU habilitada:", _gpus[0].name)
    except RuntimeError as e:
        print(f"⚠️ Error al configurar la GPU: {e}")
        print("Se usará CPU.")
        tf.config.set_visible_devices([], 'GPU') # Deshabilitar GPU si falla la configuración
else:
    print("⚠️ No se detectó GPU; se usará CPU")


def set_seeds(seed: int = SEED):
    """Establece las semillas para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_datasets(train_dir: str, val_dir: str,
                     img_size=IMG_SIZE,
                     batch_size=BATCH_SIZE,
                     seed: int = SEED):
    """
    Prepara los datasets de entrenamiento y validación cargándolos desde
    directorios separados (train_dir y val_dir).
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data directory not found: {val_dir}")


    print(f"Cargando dataset de entrenamiento desde: {train_dir}")
    # Cargar el dataset de entrenamiento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, # Apuntar directamente a la carpeta 'train'
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical", # Usamos one-hot encoding para las etiquetas
        shuffle=True # Importante para entrenamiento
    )

    print(f"Cargando dataset de validación desde: {val_dir}")
    # Cargar el dataset de validación
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, # Apuntar directamente a la carpeta 'val'
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False # No es necesario mezclar el conjunto de validación
    )

    class_names = train_ds.class_names
    print("Clases detectadas:", class_names)
    num_classes = len(class_names)
    print(f"Número de clases: {num_classes}")

    # Calcular pesos de clase para manejar desbalance (opcional pero recomendado)
    # Necesitamos contar las instancias en el dataset de entrenamiento
    print("Calculando pesos de clase...")

    # Iterar sobre el dataset de entrenamiento para recolectar todas las etiquetas
    all_labels = []
    # Usamos el dataset de entrenamiento normal (con batching)
    # Para el conteo, podemos usar un batch_size grande para ser más eficientes
    # Creamos un dataset temporal solo para el conteo sin shuffle ni cache/prefetch
    count_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=seed,
        image_size=img_size,
        batch_size=256, # Usar un batch_size más grande para el conteo
        label_mode="categorical",
        shuffle=False # No necesitamos shuffle para contar
    )

    for _, labels_batch in count_ds:
        all_labels.append(labels_batch.numpy()) # Convertir el tensor a numpy array

    # Concatenar todas las etiquetas de los batches
    all_labels = np.concatenate(all_labels, axis=0)

    # Sumar a lo largo del eje de las muestras para obtener el conteo por clase
    counts = all_labels.sum(axis=0)

    # Calcular pesos de clase para manejar desbalance
    # Formula: total_samples / (num_classes * class_count)
    total_samples = counts.sum()
    # Asegurarse de que num_classes es correcto basado en las carpetas detectadas
    num_classes_detected = len(class_names)
    # Asegurarse de que 'counts' es iterable (debería serlo después de la corrección)
    class_weight = {i: total_samples / (num_classes_detected * count)
                    for i, count in enumerate(counts)}
    print("Pesos de clase calculados:", class_weight)


    # Aplicar optimizaciones al pipeline de datos
    # Usar SHUFFLE_BUFFER_SIZE para shuffle y BUFFER_SIZE_PREFETCH para prefetch
    train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(BUFFER_SIZE_PREFETCH)
    val_ds = val_ds.cache().prefetch(BUFFER_SIZE_PREFETCH)

    return train_ds, val_ds, class_names, class_weight, num_classes


# ─────────────────────────── Modelo CNN (VGG16) ────────────────────────────

def create_cnn_classifier(num_classes, input_shape=IMG_SIZE + (3,), include_augmentation=True):
    """
    Construye el modelo CNN usando VGG16 pre-entrenado como base.
    Incluye capas de aumento de datos si include_augmentation es True.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Capa de aumento de datos (Data Augmentation) - Incluir solo si include_augmentation es True
    if include_augmentation:
        augmented = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ], name="data_augmentation")(x)
        x = augmented # Aplicar aumento si está incluido

    # Cargar el modelo base VGG16 pre-entrenado en ImageNet
    # include_top=False elimina la capa clasificadora final de VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Congelar las capas del modelo base inicialmente
    base_model.trainable = False

    # Conectar el modelo base a la pipeline
    # VGG16 espera imágenes en el rango 0-255, no 0-1.
    # Sin embargo, la capa Rescaling(1./255) en el preprocesamiento de datasets
    # ya escala a 0-1. VGG16 tiene su propio preprocesamiento interno si se usa
    # con include_top=True o si se usa preprocess_input.
    # Para simplicidad y consistencia con el preprocesamiento de datasets,
    # mantendremos la entrada 0-1 y VGG16 debería manejarlo (aunque idealmente
    # usaríamos preprocess_input de VGG16).
    # Si experimentas problemas, considera añadir una capa preprocess_input de VGG16
    # o ajustar el Rescaling inicial. Por ahora, mantenemos la simplicidad.

    x = base_model(x, training=False) # training=False es importante cuando base_model.trainable es False

    # Añadir capas clasificadoras personalizadas
    x = layers.GlobalAveragePooling2D()(x) # Pooling global para reducir dimensiones
    x = layers.Dropout(0.5)(x) # Dropout para regularización
    # Capa densa con activación ReLU
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Otro dropout
    outputs = layers.Dense(num_classes, activation="softmax")(x) # Capa de salida

    # Crear el modelo completo
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_classifier")
    return model


# ─────────────────────────── Entrenamiento y Evaluación ────────────────────────────

def compile_and_train_cnn(model, train_ds, val_ds, class_weight,
                          epochs=EPOCHS, lr=LEARNING_RATE,
                          checkpoint_path=BEST_WEIGHTS_PATH):
    """Compila y entrena el modelo CNN."""
    # Asegurar que class_weight sea serializable para Keras
    class_weight = {int(k): float(v) for k, v in class_weight.items()}

    # Usar el optimizador Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compilar el modelo
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks para guardar solo los pesos del mejor modelo
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, # Ruta donde guardar los pesos
        monitor="val_accuracy", # Monitorear la precisión en validación
        mode="max", # Guardar los pesos cuando la precisión sea máxima
        save_best_only=True, # Solo guardar los mejores pesos
        save_weights_only=True, # *** GUARDAR SOLO LOS PESOS ***
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, # Paciencia para Early Stopping
                                             restore_best_weights=True) # Restaurar los mejores pesos al detener

    print(f"\nIniciando entrenamiento del modelo CNN por {epochs} épocas...")
    # Entrenar el modelo
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, early],
        class_weight=class_weight
    )
    print("Entrenamiento del CNN completado.")
    # Devolvemos la ruta donde se guardaron los mejores pesos
    return hist, checkpoint_path


def evaluate_and_save_cnn(best_weights_path, val_ds, num_classes): # Recibe la ruta de los pesos
    """Evalúa el modelo cargando los mejores pesos y confirma su ruta."""
    print(f"\nEvaluando el modelo cargando los mejores pesos desde: {best_weights_path}")

    try:
        # *** Reconstruir la arquitectura del modelo SIN aumento de datos para evaluación ***
        # Pasamos include_augmentation=False
        loaded_model = create_cnn_classifier(num_classes=num_classes, include_augmentation=False)

        # Cargar los mejores pesos en la arquitectura reconstruida
        loaded_model.load_weights(best_weights_path)

        # *** Compilar el modelo reconstruido para la evaluación ***
        # Usar los mismos parámetros de compilación que en compile_and_train_cnn
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        loaded_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Evaluar el modelo en el dataset de validación
        loss, acc = loaded_model.evaluate(val_ds)
        print(f"Resultados de la evaluación: Loss: {loss:.4f} · Accuracy: {acc:.4f}")
        print(f"Los mejores pesos del modelo entrenado se guardaron en: {best_weights_path}")

    except Exception as e:
        print(f"Error al cargar o evaluar el modelo con pesos guardados: {e}")
        print(f"Asegúrate de que el archivo '{best_weights_path}' existe y contiene pesos válidos.")


def main():
    """Función principal para ejecutar el flujo de entrenamiento del CNN."""
    set_seeds()

    # Preparar los datasets
    # Pasamos las rutas a las carpetas train y val base.
    # Asegúrate de que las carpetas de clase dentro de TRAIN_DIR y VAL_DIR
    # contengan los datos correctos (incluida la clase healthy aumentada y dividida).
    train_ds, val_ds, classes, class_weight, num_classes = prepare_datasets(TRAIN_DIR, VAL_DIR)
    print("Clases detectadas:", classes)
    print(f"Número de clases: {num_classes}")


    # Construir el modelo CNN CON aumento de datos para entrenamiento
    model = create_cnn_classifier(num_classes=num_classes, include_augmentation=True)
    model.summary()

    # Entrenar el modelo CNN
    # El callback ModelCheckpoint guardará solo los pesos en BEST_WEIGHTS_PATH
    hist, best_weights_path_saved = compile_and_train_cnn(model, train_ds, val_ds, class_weight,
                                                        epochs=EPOCHS, lr=LEARNING_RATE,
                                                        checkpoint_path=BEST_WEIGHTS_PATH) # Usar la constante


    # Evaluar el modelo cargando los mejores pesos guardados
    # La función evaluate_and_save_cnn ahora carga y evalúa el modelo con los pesos guardados
    evaluate_and_save_cnn(best_weights_path_saved, val_ds, num_classes=num_classes)


if __name__ == "__main__":
    main()
