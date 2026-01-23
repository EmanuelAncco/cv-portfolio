#!/usr/bin/env python3
"""
potato_resnet_transfer_learning.py – Clasificador de Tizón de Papa con Transfer Learning (ResNet50V2).

Implementa un modelo ResNet50V2 pre-entrenado y lo adapta al dataset de papa.
Entrena principalmente la cabeza clasificadora añadida.
Guarda solo los pesos del mejor modelo durante el entrenamiento.

Basado en la estrategia de Transfer Learning común y similar a la vista en:
https://www.kaggle.com/code/adnanyaramis/potato-blight-disease

• TensorFlow 2.10+ • CUDA 11.2+ • cuDNN 8.1+ • Python 3.10+
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

# ─────────────────────────── Parámetros globales ────────────────────────────
# Asegúrate de que DATA_DIR apunta a la carpeta que contiene las carpetas train y val
DATA_DIR         = r"D:\Python_proyectos_2025\Agricultura\archive\DATA"  # ← Cambia si es necesario
TRAIN_DIR        = os.path.join(DATA_DIR, 'train')
VAL_DIR          = os.path.join(DATA_DIR, 'val')

SEED             = 123
IMG_SIZE         = (224, 224) # Tamaño de imagen estándar para ResNet50V2
BATCH_SIZE       = 32
EPOCHS           = 25 # Menos épocas suelen ser suficientes para transfer learning (ajustar según sea necesario)
LEARNING_RATE    = 1e-3 # Tasa de aprendizaje inicial (puede ser un poco más alta para la cabeza)

BUFFER_SIZE_PREFETCH = tf.data.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 1000

# Nombre del archivo donde se guardarán los pesos del mejor modelo
BEST_WEIGHTS_PATH = "resnet_transfer_best_weights.h5" # Nuevo nombre para los pesos

# ──────────────────────────── Configuración GPU ─────────────────────────────
# (Código de configuración de GPU igual que antes)
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
        tf.config.set_visible_devices([], 'GPU')
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
    Prepara los datasets de entrenamiento y validación.
    (Función idéntica a la original, calcula class_weight).
    """
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data directory not found: {val_dir}")

    print(f"Cargando dataset de entrenamiento desde: {train_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    print(f"Cargando dataset de validación desde: {val_dir}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Clases detectadas:", class_names)
    num_classes = len(class_names)
    print(f"Número de clases: {num_classes}")

    print("Calculando pesos de clase...")
    all_labels = []
    count_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, seed=seed, image_size=img_size, batch_size=256, label_mode="categorical", shuffle=False
    )
    for _, labels_batch in count_ds:
        all_labels.append(labels_batch.numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    counts = all_labels.sum(axis=0)
    total_samples = counts.sum()
    num_classes_detected = len(class_names)
    class_weight = {i: total_samples / (num_classes_detected * count)
                    for i, count in enumerate(counts) if count > 0} # Evitar división por cero si una clase no tiene muestras
    print("Pesos de clase calculados:", class_weight)

    # Optimizar datasets
    train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(BUFFER_SIZE_PREFETCH)
    val_ds = val_ds.cache().prefetch(BUFFER_SIZE_PREFETCH)

    # IMPORTANTE para ResNetV2: Aplicar la función de preprocesamiento específica
    # O incluir la capa Rescaling en el modelo. Optaremos por la capa en el modelo.
    # train_ds = train_ds.map(lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.map(lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names, class_weight, num_classes


# ─────────────────── Modelo con Transfer Learning (ResNet50V2) ───────────────────

def create_transfer_model(num_classes, input_shape=IMG_SIZE + (3,),
                          include_augmentation=True):
    """
    Construye el modelo usando ResNet50V2 pre-entrenado.
    """
    # --- Capa de Entrada ---
    inputs = keras.Input(shape=input_shape)

    # --- Capa de Aumento de Datos (Opcional pero recomendado) ---
    x = inputs
    if include_augmentation:
        # Usar aumentos ligeros para no distorsionar demasiado las características aprendidas
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        # Nota: ResNet fue entrenado con imágenes [0, 255], pero su función
        # preprocess_input normaliza a [-1, 1]. Usaremos Rescaling(1./127.5, offset=-1)
        # para lograr una normalización similar si no usamos preprocess_input.
        # O más simple, escalar a [0, 1] que también funciona bien.

    # --- Capa de Reescalado ---
    # Normaliza los píxeles de [0, 255] a [0, 1]
    x = layers.Rescaling(1./255)(x)
    # Alternativa para normalizar a [-1, 1] como preprocess_input:
    # x = layers.Rescaling(1./127.5, offset=-1)(x)

    # --- Modelo Base Pre-entrenado (ResNet50V2) ---
    # include_top=False: No incluir la capa clasificadora original de ImageNet
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=x, # Pasar tensor preprocesado
                            input_shape=input_shape) # input_shape es redundante si se usa input_tensor

    # --- Congelar el Modelo Base ---
    # Evita que los pesos pre-entrenados se modifiquen durante el entrenamiento inicial
    base_model.trainable = False

    # --- Nueva Cabeza Clasificadora ---
    # Añadir capas personalizadas encima del modelo base
    # Usar la salida del modelo base congelado
    x = base_model.output # Obtener la salida del modelo base
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x) # Agrupar características espacialmente
    x = layers.BatchNormalization()(x) # Normalizar antes de Dropout/Dense
    x = layers.Dropout(0.3, name="top_dropout")(x) # Regularización Dropout
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x) # Capa final

    # --- Construir el Modelo Completo ---
    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50V2_Transfer")

    return model, base_model # Devolver base_model para posible afinamiento posterior

# ─────────────────────────── Entrenamiento y Evaluación ────────────────────────────

def compile_and_train_model(model, train_ds, val_ds, class_weight,
                            epochs=EPOCHS, lr=LEARNING_RATE,
                            checkpoint_path=BEST_WEIGHTS_PATH):
    """Compila y entrena el modelo (Fase 1: Entrenar la cabeza)."""
    # Asegurar que class_weight sea serializable
    class_weight = {int(k): float(v) for k, v in class_weight.items()}

    # Usar optimizador Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compilar el modelo
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, # Menor paciencia puede ser útil aquí
        restore_best_weights=True, verbose=1
    )
    # Opcional: Reducir LR si la mejora se estanca
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )

    print(f"\nIniciando entrenamiento (Fase 1: Cabeza) por hasta {epochs} épocas...")
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, early, reduce_lr],
        class_weight=class_weight
    )
    print("Entrenamiento (Fase 1) completado.")
    return hist, checkpoint_path

# --- Opcional: Función para afinamiento (Fine-tuning) ---
def fine_tune_model(model, base_model, train_ds, val_ds, class_weight,
                    initial_hist, fine_tune_lr=1e-5, fine_tune_epochs=10,
                    checkpoint_path=BEST_WEIGHTS_PATH):
    """Descongela capas del base_model y entrena con LR baja."""
    # Descongelar el modelo base
    base_model.trainable = True

    # Opcional: Congelar las primeras capas (BatchNormalization a veces da problemas)
    # O descongelar solo un número limitado de capas superiores
    print(f"Número total de capas en el modelo base: {len(base_model.layers)}")
    fine_tune_at = 100 # Descongelar desde la capa 100 hacia arriba (ajustar)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        # ¡Importante! Mantener las capas BatchNormalization congeladas si se descongela
        if isinstance(layer, layers.BatchNormalization):
             layer.trainable = False


    # Recompilar el modelo con una tasa de aprendizaje muy baja
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Modelo recompilado para afinamiento (fine-tuning).")
    model.summary() # Mostrar qué capas son entrenables ahora

    # Continuar entrenamiento
    total_epochs = initial_hist.epoch[-1] + 1 + fine_tune_epochs
    print(f"\nIniciando afinamiento (Fase 2) por {fine_tune_epochs} épocas más...")

    # Usar los mismos callbacks, pero monitorizar val_accuracy podría ser mejor aquí
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, # O monitor='val_accuracy'
        restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1 # LR mínima más baja
    )

    hist_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_hist.epoch[-1] + 1, # Continuar desde donde terminó
        callbacks=[ckpt, early, reduce_lr],
        class_weight=class_weight
    )
    print("Afinamiento (Fase 2) completado.")
    return hist_fine, checkpoint_path


def evaluate_model(weights_path, val_ds, num_classes):
    """Carga los mejores pesos y evalúa el modelo."""
    print(f"\nEvaluando el modelo cargando los mejores pesos desde: {weights_path}")

    try:
        # Reconstruir la arquitectura (SIN aumento para evaluación)
        loaded_model, _ = create_transfer_model(num_classes=num_classes, include_augmentation=False)

        # Cargar los mejores pesos
        loaded_model.load_weights(weights_path)

        # Compilar para evaluación (necesario para .evaluate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # LR no importa mucho aquí
        loaded_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Evaluar
        loss, acc = loaded_model.evaluate(val_ds)
        print(f"Resultados de la evaluación: Loss: {loss:.4f} · Accuracy: {acc:.4f}")
        print(f"Los mejores pesos del modelo entrenado se guardaron en: {weights_path}")

    except Exception as e:
        print(f"Error al cargar o evaluar el modelo con pesos guardados: {e}")
        print(f"Asegúrate de que el archivo '{weights_path}' existe y es válido.")

# ─────────────────────────── Ejecución Principal ────────────────────────────
if __name__ == "__main__":
    set_seeds()

    # 1. Preparar los datasets
    train_ds, val_ds, classes, class_weight, num_classes = prepare_datasets(TRAIN_DIR, VAL_DIR)
    print("Clases detectadas para el modelo:", classes)
    print(f"Número de clases: {num_classes}")

    # 2. Construir el modelo con Transfer Learning (CON aumento para entrenamiento)
    model, base_model = create_transfer_model(num_classes=num_classes, include_augmentation=True)
    print("Arquitectura del Modelo Inicial (Cabeza Entrenable):")
    model.summary()

    # 3. Compilar y Entrenar (Fase 1: Cabeza)
    hist, best_weights_path_saved = compile_and_train_model(
        model, train_ds, val_ds, class_weight,
        epochs=EPOCHS, lr=LEARNING_RATE, checkpoint_path=BEST_WEIGHTS_PATH
    )

    # --- OPCIONAL: Fase de Afinamiento (Fine-tuning) ---
    # Descomenta las siguientes líneas si quieres intentar el afinamiento
    # print("\n--- Iniciando Fase de Afinamiento Opcional ---")
    # hist_fine, best_weights_path_saved = fine_tune_model(
    #     model, base_model, train_ds, val_ds, class_weight,
    #     initial_hist=hist, fine_tune_lr=1e-5, fine_tune_epochs=10, # Ajusta LR y épocas
    #     checkpoint_path=BEST_WEIGHTS_PATH
    # )
    # ----------------------------------------------------

    # 4. Evaluar el modelo final cargando los mejores pesos guardados
    evaluate_model(best_weights_path_saved, val_ds, num_classes=num_classes)

