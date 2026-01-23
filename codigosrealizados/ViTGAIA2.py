#!/usr/bin/env python3
"""
potato_vit.py – Clasificador de Tizón de la Papa con Vision Transformer (GPU-ready)

Implementa un modelo Vision Transformer para clasificar enfermedades de la papa.
Se ejecuta directamente con **Run ▶** en PyCharm.

• TensorFlow 2.10+ • CUDA 11.2+ • cuDNN 8.1+ • Python 3.10+
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────── Parámetros globales ────────────────────────────
# Asegúrate de que DATA_DIR apunta a la carpeta que contiene las carpetas train y val
# Por ejemplo, si tus carpetas train y val están en D:\data\potato_dataset,
# entonces DATA_DIR = r"D:\data\potato_dataset"
DATA_DIR         = r"D:\Python_proyectos_2025\Agricultura\archive\DATA"  # ← cambia aquí si mueves el dataset base
TRAIN_DIR        = os.path.join(DATA_DIR, 'train') # Ruta a la carpeta de entrenamiento
VAL_DIR          = os.path.join(DATA_DIR, 'val')   # Ruta a la carpeta de validación

SEED             = 123
IMG_SIZE         = (224, 224) # Tamaño de imagen para ViT
PATCH_SIZE       = 16        # Tamaño de los parches de imagen
# El número de parches se calcula a partir del tamaño de la imagen y el tamaño del parche
NUM_PATCHES      = (IMG_SIZE[0] // PATCH_SIZE) ** 2 # Número total de parches
PROJECTION_DIM   = 64        # Dimensión de la proyección de parches
NUM_HEADS        = 4         # Número de cabezas en el Multi-Head Attention
TRANSFORMER_LAYERS = 8       # Número de capas del Transformer Encoder
# Unidades en el MLP dentro de cada bloque Transformer
MLP_UNITS        = [PROJECTION_DIM * 2, PROJECTION_DIM] # Unidades en el MLP del Transformer
BATCH_SIZE       = 32
EPOCHS_INITIAL   = 50 # Épocas para la fase inicial (si aplicara, aunque ViT se entrena end-to-end)
EPOCHS_FINE_TUNE = 50 # Épocas para el entrenamiento completo del ViT
LEARNING_RATE    = 1e-4 # Tasa de aprendizaje inicial
# WEIGHT_DECAY ya no se usa directamente en el optimizador Adam
# WEIGHT_DECAY     = 1e-4 # Decaimiento de pesos para regularización
BUFFER_SIZE_PREFETCH = tf.data.AUTOTUNE # Tamaño del buffer para prefetch
SHUFFLE_BUFFER_SIZE = 1000 # Tamaño del buffer para shuffle (debe ser > 0)
# VAL_SPLIT ya no se usa para dividir, solo para referencia si es necesario
# VAL_SPLIT        = 0.2 # Porcentaje de datos para validación
# ────────────────────────────────────────────────────────────────────────────

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

    # --- INICIO: Corrección para calcular los conteos por clase ---
    # Iterar sobre el dataset de entrenamiento para recolectar todas las etiquetas
    all_labels = []
    # Usamos el dataset de entrenamiento normal (con batching)
    # Para el conteo, podemos usar un batch_size grande para ser más eficientes
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
    # --- FIN: Corrección para calcular los conteos por clase ---


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


# ─────────────────────────── Componentes del ViT ────────────────────────────

class Patches(layers.Layer):
    """Divide las imágenes en parches."""
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # Extraer parches usando extract_patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Reformar los parches para que cada uno sea un vector plano
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    """Codifica los parches y añade embeddings posicionales."""
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        # Capa para proyectar los parches a la dimensión deseada
        self.projection = layers.Dense(units=projection_dim)
        # Embedding posicional aprendible
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        # Crear posiciones para cada parche
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # Proyectar los parches y añadir el embedding posicional
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
            })
        return config

def mlp(x, hidden_units, dropout_rate):
    """Bloque MLP (Multi-Layer Perceptron)."""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# ─────────────────────────── Modelo ViT ────────────────────────────

def create_vit_classifier(num_classes, input_shape=IMG_SIZE + (3,),
                          patch_size=PATCH_SIZE, projection_dim=PROJECTION_DIM,
                          num_heads=NUM_HEADS, transformer_layers=TRANSFORMER_LAYERS,
                          mlp_units=MLP_UNITS):
    """Construye el modelo Vision Transformer."""
    inputs = keras.Input(shape=input_shape)
    # Capa de aumento de datos (Data Augmentation)
    augmented = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ], name="data_augmentation")(inputs)

    # Crear parches
    patches = Patches(patch_size)(augmented)
    # Codificar parches y añadir embedding posicional
    encoded_patches = PatchEncoder(NUM_PATCHES, projection_dim)(patches)

    # Crear múltiples capas del Transformer Encoder
    for _ in range(transformer_layers):
        # Normalización de la capa 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Añadir y Normalizar
        x2 = layers.Add()([attention_output, encoded_patches])
        # Normalización de la capa 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # Bloque MLP
        x3 = mlp(x3, hidden_units=mlp_units, dropout_rate=0.1)
        # Añadir y Normalizar
        encoded_patches = layers.Add()([x3, x2])

    # Crear una representación de alto nivel a partir de los parches codificados
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation) # Aplanar para la capa densa
    representation = layers.Dropout(0.5)(representation) # Dropout antes del clasificador

    # Capa clasificadora final
    outputs = layers.Dense(num_classes, activation="softmax")(representation)

    # Crear el modelo completo
    model = keras.Model(inputs=inputs, outputs=outputs, name="vit_classifier")
    return model


# ─────────────────────────── Entrenamiento y Evaluación ────────────────────────────

def compile_and_train_vit(model, train_ds, val_ds, class_weight,
                          epochs=EPOCHS_FINE_TUNE, lr=LEARNING_RATE,
                          # Eliminamos weight_decay de aquí ya que Adam no lo usa directamente
                          checkpoint="vit_checkpoint.keras"):
    """Compila y entrena el modelo ViT."""
    # Asegurar que class_weight sea serializable para Keras
    class_weight = {int(k): float(v) for k, v in class_weight.items()}

    # Usar el optimizador Adam
    # Eliminamos el argumento weight_decay
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compilar el modelo
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks para guardar el mejor modelo y detener el entrenamiento temprano
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        checkpoint,
        monitor="val_accuracy", # Monitorear la precisión en validación
        mode="max", # Guardar el modelo cuando la precisión sea máxima
        save_best_only=True, # Solo guardar el mejor modelo
        save_weights_only=True # Solo guardar los pesos
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, # Aumentar paciencia para ViT
                                             restore_best_weights=True) # Restaurar los mejores pesos al detener

    print(f"\nIniciando entrenamiento del modelo ViT por {epochs} épocas...")
    # Entrenar el modelo
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, early],
        class_weight=class_weight
    )
    print("Entrenamiento del ViT completado.")
    return hist, ckpt.filepath


def evaluate_and_save_vit(best_path, val_ds, num_classes, final_path="final_vit_model_tf"):
    """Evalúa el mejor modelo ViT guardado y lo guarda en formato TensorFlow SavedModel."""
    print(f"\nEvaluando el mejor modelo ViT guardado desde: {best_path}")

    # Reconstruir la arquitectura del modelo ViT.
    # Necesitamos pasar las clases personalizadas (Patches, PatchEncoder)
    # al cargar el modelo si se guarda la arquitectura completa.
    # Como solo guardamos pesos, reconstruimos la arquitectura manualmente.
    loaded_model = create_vit_classifier(num_classes=num_classes)

    # Cargar los pesos del mejor checkpoint
    loaded_model.load_weights(best_path)

    # *** COMPILAR EL MODELO CARGADO ANTES DE EVALUARLO ***
    # Usar los mismos parámetros de compilación que en compile_and_train_vit
    # Usar el optimizador Adam, sin weight_decay aquí tampoco
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    loaded_model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Evaluar el modelo en el dataset de validación
    loss, acc = loaded_model.evaluate(val_ds)
    print(f"Resultados de la evaluación: Loss: {loss:.4f} · Accuracy: {acc:.4f}")

    # Guardar el modelo completo en formato TensorFlow SavedModel
    try:
        loaded_model.save(final_path, save_format='tf')
        print(f"Modelo final ViT guardado en formato SavedModel en: {final_path}")
    except Exception as e:
        print(f"Error al guardar el modelo en formato SavedModel: {e}")
        print("Asegúrate de que no haya problemas de permisos.")

    return loaded_model


def convert_to_tflite(model, out_path="vit_model.tflite"):
    """Convierte el modelo Keras a formato TFLite."""
    print("\nConvirtiendo el modelo a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # *** HABILITAR TF SELECT PARA OPERACIONES NO NATIVAS ***
    # Permite que el conversor incluya operaciones de TensorFlow no nativas
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Incluir operaciones nativas de TFLite
        tf.lite.OpsSet.TF_SELECT # Incluir operaciones de TensorFlow (TF Select)
    ]

    # Aplicar optimizaciones por defecto
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
        # Guardar el modelo TFLite en un archivo
        with open(out_path, "wb") as f:
            f.write(tflite_model)
        print("Modelo TFLite guardado →", out_path)
    except Exception as e:
        print(f"Error al convertir a TFLite: {e}")
        print("Asegúrate de que el modelo sea compatible con TFLite y que TF Select esté habilitado si es necesario.")


def main():
    """Función principal para ejecutar el flujo de entrenamiento y conversión del ViT."""
    set_seeds()

    # Preparar los datasets
    # Pasamos las rutas a las carpetas train y val
    train_ds, val_ds, classes, class_weight, num_classes = prepare_datasets(TRAIN_DIR, VAL_DIR)
    print("Clases detectadas:", classes)
    print(f"Número de clases: {num_classes}")


    # Construir el modelo ViT
    model = create_vit_classifier(num_classes=num_classes)
    model.summary()

    # Entrenar el modelo ViT
    # No hay una fase de extracción de características separada para ViT,
    # se entrena de extremo a extremo.
    hist, best_ckpt = compile_and_train_vit(model, train_ds, val_ds, class_weight,
                                            epochs=EPOCHS_FINE_TUNE, lr=LEARNING_RATE,
                                            checkpoint="vit_checkpoint.keras")


    # Evaluar el mejor modelo y guardarlo en formato Keras
    # Pasamos num_classes a la función de evaluación
    final_model = evaluate_and_save_vit(best_ckpt, val_ds, num_classes=num_classes, final_path="final_vit_model_tf")

    # Convertir el modelo final a TFLite
    convert_to_tflite(final_model)


if __name__ == "__main__":
    main()
