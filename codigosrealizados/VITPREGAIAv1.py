#!/usr/bin/env python3
"""
potato_vit_transfer_learning.py – Clasificador de Tizón de Papa con Transfer Learning (ViT).

Implementa un modelo Vision Transformer (ViT) pre-entrenado desde TensorFlow Hub
y lo adapta al dataset de papa.
Entrena principalmente la cabeza clasificadora añadida.
Guarda solo los pesos del mejor modelo durante el entrenamiento.

Requiere: pip install tensorflow_hub

• TensorFlow 2.10+ • CUDA 11.2+ • cuDNN 8.1+ • Python 3.10+ • tensorflow_hub
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub # Necesario para cargar el modelo ViT

# ─────────────────────────── Parámetros globales ────────────────────────────
DATA_DIR         = r"D:\Python_proyectos_2025\Agricultura\archive\DATA"  # ← Cambia si es necesario
TRAIN_DIR        = os.path.join(DATA_DIR, 'train')
VAL_DIR          = os.path.join(DATA_DIR, 'val')

SEED             = 123
IMG_SIZE         = (224, 224) # Tamaño de imagen común para ViT pre-entrenados
BATCH_SIZE       = 32
EPOCHS           = 25 # Ajustar según sea necesario para transfer learning
LEARNING_RATE    = 1e-3 # Tasa de aprendizaje inicial para la cabeza

BUFFER_SIZE_PREFETCH = tf.data.AUTOTUNE
SHUFFLE_BUFFER_SIZE = 1000

# Nombre del archivo donde se guardarán los pesos del mejor modelo
BEST_WEIGHTS_PATH = "vit_transfer_best_weights.h5" # Nuevo nombre para los pesos ViT

# URL del modelo ViT pre-entrenado en TensorFlow Hub (Extractor de características)
# Usaremos ViT-Base parche 16x16 como extractor de características
VIT_MODEL_URL = "https://tfhub.dev/google/vision_transformer/vit_b16_fe/1"

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
    (Función idéntica a la anterior).
    """
    # ... (Código idéntico a la función prepare_datasets del script ResNet) ...
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
    # Usamos un dataset temporal para contar eficientemente
    count_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, seed=seed, image_size=img_size, batch_size=256, label_mode="categorical", shuffle=False
    )
    for _, labels_batch in count_ds:
        all_labels.append(labels_batch.numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    counts = all_labels.sum(axis=0)
    total_samples = counts.sum()
    num_classes_detected = len(class_names)
    # Calcular pesos de clase, evitando división por cero
    class_weight = {i: total_samples / (num_classes_detected * count)
                    for i, count in enumerate(counts) if count > 0}
    print("Pesos de clase calculados:", class_weight)

    # Optimizar datasets
    train_ds = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(BUFFER_SIZE_PREFETCH)
    val_ds = val_ds.cache().prefetch(BUFFER_SIZE_PREFETCH)

    return train_ds, val_ds, class_names, class_weight, num_classes


# ─────────────────── Modelo con Transfer Learning (ViT Pre-entrenado) ───────────────────

def create_vit_transfer_model(num_classes, input_shape=IMG_SIZE + (3,),
                              vit_model_url=VIT_MODEL_URL,
                              include_augmentation=True):
    """
    Construye el modelo usando un ViT pre-entrenado desde TensorFlow Hub.
    """
    # --- Capa de Entrada ---
    inputs = keras.Input(shape=input_shape)

    # --- Capa de Aumento de Datos (Opcional pero recomendado) ---
    x = inputs
    if include_augmentation:
        # Usar aumentos ligeros
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)

    # --- Capa de Reescalado ---
    # Normaliza los píxeles de [0, 255] a [0, 1], esperado por muchos modelos de TF Hub
    x = layers.Rescaling(1./255)(x)

    # --- Modelo Base Pre-entrenado (ViT desde TF Hub) ---
    # Cargar el ViT como una capa Keras, especificando que NO es entrenable inicialmente
    base_model_layer = hub.KerasLayer(vit_model_url, trainable=False, name='vit_base')

    # Pasar la entrada (posiblemente aumentada y reescalada) al modelo base
    # La salida suele ser un vector de características agrupado (pooled)
    x = base_model_layer(x)

    # --- Nueva Cabeza Clasificadora ---
    # Añadir capas personalizadas encima de la salida del ViT
    # x = layers.BatchNormalization()(x) # Opcional: puede ayudar a estabilizar
    x = layers.Dropout(0.3, name="top_dropout")(x) # Regularización Dropout
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x) # Capa final

    # --- Construir el Modelo Completo ---
    model = keras.Model(inputs=inputs, outputs=outputs, name="ViT_Transfer")

    return model, base_model_layer # Devolver la capa base para posible afinamiento

# ─────────────────────────── Entrenamiento y Evaluación ────────────────────────────

def compile_and_train_model(model, train_ds, val_ds, class_weight,
                            epochs=EPOCHS, lr=LEARNING_RATE,
                            checkpoint_path=BEST_WEIGHTS_PATH):
    """Compila y entrena el modelo (Fase 1: Entrenar la cabeza)."""
    # ... (Código idéntico a la función compile_and_train_model del script ResNet) ...
    class_weight = {int(k): float(v) for k, v in class_weight.items()}
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1
    )
    print(f"\nIniciando entrenamiento (Fase 1: Cabeza ViT) por hasta {epochs} épocas...")
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
def fine_tune_vit_model(model, base_model_layer, train_ds, val_ds, class_weight,
                        initial_hist, fine_tune_lr=1e-5, fine_tune_epochs=10,
                        checkpoint_path=BEST_WEIGHTS_PATH):
    """Descongela la capa base del ViT y entrena con LR baja."""
    # Descongelar la capa base del ViT
    base_model_layer.trainable = True
    print("Capa base ViT descongelada para afinamiento.")

    # Recompilar el modelo con una tasa de aprendizaje muy baja
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Modelo recompilado para afinamiento (fine-tuning).")
    model.summary() # Mostrar que la capa hub es entrenable

    # Continuar entrenamiento
    total_epochs = initial_hist.epoch[-1] + 1 + fine_tune_epochs
    print(f"\nIniciando afinamiento (Fase 2) por {fine_tune_epochs} épocas más...")

    # Usar los mismos callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1 # LR mínima más baja
    )

    hist_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_hist.epoch[-1] + 1,
        callbacks=[ckpt, early, reduce_lr],
        class_weight=class_weight
    )
    print("Afinamiento (Fase 2) completado.")
    return hist_fine, checkpoint_path


def evaluate_vit_model(weights_path, val_ds, num_classes, vit_model_url=VIT_MODEL_URL):
    """Carga los mejores pesos y evalúa el modelo ViT."""
    print(f"\nEvaluando el modelo ViT cargando los mejores pesos desde: {weights_path}")

    try:
        # Reconstruir la arquitectura (SIN aumento para evaluación)
        # Es crucial usar la misma URL del modelo base
        loaded_model, _ = create_vit_transfer_model(
            num_classes=num_classes,
            vit_model_url=vit_model_url,
            include_augmentation=False
        )

        # Cargar los mejores pesos
        loaded_model.load_weights(weights_path)

        # Compilar para evaluación
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) # LR no importa mucho aquí
        loaded_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Evaluar
        loss, acc = loaded_model.evaluate(val_ds)
        print(f"Resultados de la evaluación ViT: Loss: {loss:.4f} · Accuracy: {acc:.4f}")
        print(f"Los mejores pesos del modelo ViT entrenado se guardaron en: {weights_path}")

    except Exception as e:
        print(f"Error al cargar o evaluar el modelo ViT con pesos guardados: {e}")
        print(f"Asegúrate de que el archivo '{weights_path}' existe y es válido.")
        print(f"Verifica también que la URL del modelo ViT ({vit_model_url}) sea correcta y accesible.")


# ─────────────────────────── Ejecución Principal ────────────────────────────
if __name__ == "__main__":
    set_seeds()

    # 1. Preparar los datasets
    train_ds, val_ds, classes, class_weight, num_classes = prepare_datasets(TRAIN_DIR, VAL_DIR)
    print("Clases detectadas para el modelo ViT:", classes)
    print(f"Número de clases: {num_classes}")

    # 2. Construir el modelo con Transfer Learning (ViT CON aumento para entrenamiento)
    model, base_model_layer = create_vit_transfer_model(
        num_classes=num_classes,
        include_augmentation=True
    )
    print("Arquitectura del Modelo ViT Inicial (Cabeza Entrenable):")
    model.summary()

    # 3. Compilar y Entrenar (Fase 1: Cabeza)
    hist, best_weights_path_saved = compile_and_train_model(
        model, train_ds, val_ds, class_weight,
        epochs=EPOCHS, lr=LEARNING_RATE, checkpoint_path=BEST_WEIGHTS_PATH
    )

    # --- OPCIONAL: Fase de Afinamiento (Fine-tuning) ---
    # Descomenta las siguientes líneas si quieres intentar el afinamiento
    # print("\n--- Iniciando Fase de Afinamiento Opcional ViT ---")
    # hist_fine, best_weights_path_saved = fine_tune_vit_model(
    #     model, base_model_layer, train_ds, val_ds, class_weight,
    #     initial_hist=hist, fine_tune_lr=1e-5, fine_tune_epochs=10, # Ajusta LR y épocas
    #     checkpoint_path=BEST_WEIGHTS_PATH
    # )
    # ----------------------------------------------------

    # 4. Evaluar el modelo final cargando los mejores pesos guardados
    evaluate_vit_model(best_weights_path_saved, val_ds, num_classes=num_classes)

