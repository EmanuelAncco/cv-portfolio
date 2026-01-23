#!/usr/bin/env python3
"""
potato2.py  –  Clasificador de Tizón de la Papa (GPU‑ready)

Se ejecuta directamente con **Run ▶** en PyCharm (sin argumentos externos).

• TensorFlow 2.10  • CUDA 11.2  • cuDNN 8.1  • Python 3.10
"""

import os
import random
import numpy as np
import tensorflow as tf

# ─────────────────────────── Parámetros globales ────────────────────────────
DATA_DIR         = r"D:\Python_proyectos_2025\Agricultura\archive\DATA"  # ← cambia aquí si mueves el dataset
SEED             = 123
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32
EPOCHS_INITIAL   = 50
EPOCHS_FINE_TUNE = 20 # Este parámetro se usaría en la función fine_tune
RESCALE_RANGE    = 1.0 / 255.0
# ────────────────────────────────────────────────────────────────────────────

# → Activar GPU con memory‑growth si está disponible
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    for g in _gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.config.set_visible_devices(_gpus[0], "GPU")
    print("✔ GPU habilitada:", _gpus[0].name)
else:
    print("⚠️  No se detectó GPU; se usará CPU")


def set_seeds(seed: int = SEED):
    """Establece las semillas para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_datasets(data_dir: str,
                     img_size=IMG_SIZE,
                     batch_size=BATCH_SIZE,
                     val_split: float = 0.2,
                     seed: int = SEED):
    """Prepara los datasets de entrenamiento y validación."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Pase inicial (sin shuffle) para contar clases y pesos
    # Se usa subset="training" para obtener las etiquetas de todas las imágenes
    # en el conjunto de datos completo antes de la división real.
    print("Preparando dataset base para conteo de clases...")
    base_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=False)

    class_names = base_ds.class_names
    # Concatenar todas las etiquetas para calcular los pesos de clase
    labels      = np.concatenate([y for _, y in base_ds], axis=0)
    counts      = labels.sum(axis=0)
    # Calcular pesos de clase para manejar desbalance
    # Formula: total_samples / (num_classes * class_count)
    total_samples = counts.sum()
    num_classes = len(counts)
    class_weight = {i: total_samples / (num_classes * count)
                    for i, count in enumerate(counts)}
    print("Pesos de clase calculados:", class_weight)


    # Datasets reales con shuffle/cache/prefetch para entrenamiento y validación
    print(f"Preparando datasets de entrenamiento ({1.0-val_split:.0%}) y validación ({val_split:.0%})...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=True)

    val_ds   = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation",
        seed=seed, image_size=img_size, batch_size=batch_size,
        label_mode="categorical", shuffle=False)

    autotune = tf.data.AUTOTUNE
    # Aplicar cache, shuffle (solo a train), y prefetch para optimizar el pipeline de datos
    return (train_ds.cache().shuffle(1000).prefetch(autotune),
            val_ds.cache().prefetch(autotune),
            class_names, class_weight)


def build_model(num_classes: int, img_shape=IMG_SIZE + (3,)):
    """Construye el modelo de clasificación usando EfficientNetB0."""
    inputs = tf.keras.Input(shape=img_shape)
    # Capa de reescalado para normalizar los valores de píxeles
    x = tf.keras.layers.Rescaling(RESCALE_RANGE)(inputs)

    # Capas de aumento de datos (Data Augmentation)
    # Aplicadas como una capa Sequential para integrarlas en el modelo
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")(x) # Añadir nombre a la capa Sequential

    # Cargar el modelo base pre-entrenado (EfficientNetB0)
    # include_top=False elimina la capa clasificadora final
    # weights="imagenet" carga los pesos entrenados en ImageNet
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                input_shape=img_shape)
    # Congelar las capas del modelo base inicialmente
    base.trainable = False

    # Conectar el modelo base a la pipeline
    x = base(x, training=False) # training=False es importante cuando base.trainable es False

    # Añadir capas clasificadoras personalizadas
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # Pooling global para reducir dimensiones
    x = tf.keras.layers.Dropout(0.2)(x) # Dropout para regularización
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x) # Capa de salida

    # Crear el modelo completo
    return tf.keras.Model(inputs, outputs, name="potato_classifier"), base


def compile_and_train(model, train_ds, val_ds, class_weight,
                      epochs=EPOCHS_INITIAL, lr=1e-3,
                      checkpoint="feature_extraction.keras"):
    """Compila y entrena el modelo (fase de extracción de características)."""
    # Asegurar que class_weight sea serializable para Keras
    class_weight = {int(k): float(v) for k, v in class_weight.items()}

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks para guardar el mejor modelo y detener el entrenamiento temprano
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        checkpoint,
        monitor="val_accuracy", # Monitorear la precisión en validación
        mode="max", # Guardar el modelo cuando la precisión sea máxima
        save_best_only=True, # Solo guardar el mejor modelo
        save_weights_only=True # Solo guardar los pesos
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, # Monitorear la pérdida en validación
                                             restore_best_weights=True) # Restaurar los mejores pesos al detener

    print(f"\nIniciando entrenamiento (fase de extracción de características) por {epochs} épocas...")
    # Entrenar el modelo
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=epochs, # Usar el número de épocas especificado
                     initial_epoch=0, # Siempre empezar desde la época 0 en esta fase
                     callbacks=[ckpt, early],
                     class_weight=class_weight)
    print("Entrenamiento inicial completado.")
    return hist, ckpt.filepath

# TODO: Implementar la función fine_tune para la fase de ajuste fino
# def fine_tune(model, base, train_ds, val_ds, class_weight, start_epoch,
#               epochs=EPOCHS_FINE_TUNE, lr=1e-4,
#               checkpoint="fine_tune.keras"):
#     """Ajusta fino el modelo (descongelando capas del base)."""
#     # Descongelar algunas capas del modelo base
#     # base.trainable = True
#     # Ajustar la tasa de aprendizaje para el ajuste fino
#     # model.compile(...)
#     # Continuar el entrenamiento
#     # hist = model.fit(..., initial_epoch=start_epoch, epochs=start_epoch + epochs, ...)
#     # return hist, ckpt.filepath
#     pass # Placeholder

def evaluate_and_save(best_path, val_ds, final_path="final_model_tf"):
    """Evalúa el mejor modelo guardado y lo guarda en formato TensorFlow SavedModel."""
    print(f"\nEvaluando el mejor modelo guardado desde: {best_path}")
    # Reconstruir la arquitectura del modelo
    # Se necesita el número de clases, que se puede obtener del dataset de validación
    num_classes = val_ds.element_spec[1].shape[-1]
    loaded_model, _ = build_model(num_classes=num_classes)
    # Cargar los pesos del mejor checkpoint
    loaded_model.load_weights(best_path)

    # *** COMPILAR EL MODELO CARGADO ANTES DE EVALUARLO ***
    # Usar los mismos parámetros de compilación que en compile_and_train
    loaded_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), # Usar la misma tasa de aprendizaje inicial
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])

    # Evaluar el modelo en el dataset de validación
    loss, acc = loaded_model.evaluate(val_ds)
    print(f"Resultados de la evaluación: Loss: {loss:.4f} · Accuracy: {acc:.4f}")

    # Guardar el modelo completo en formato TensorFlow SavedModel
    # Cambiamos la extensión a .tf o simplemente usamos un nombre de carpeta
    # El formato 'tf' indica SavedModel
    try:
        loaded_model.save(final_path, save_format='tf')
        print(f"Modelo final guardado en formato SavedModel en: {final_path}")
    except Exception as e:
        print(f"Error al guardar el modelo en formato SavedModel: {e}")
        print("Asegúrate de que no haya problemas de permisos o de que el modelo sea compatible con SavedModel.")


    return loaded_model


def convert_to_tflite(model, out_path="model.tflite"):
    """Convierte el modelo Keras a formato TFLite."""
    print("\nConvirtiendo el modelo a TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
        print("Asegúrate de que el modelo esté completamente entrenado y guardado correctamente.")


def main():
    """Función principal para ejecutar el flujo de entrenamiento y conversión."""
    set_seeds()

    # Preparar los datasets
    train_ds, val_ds, classes, class_weight = prepare_datasets(DATA_DIR)
    print("Clases detectadas:", classes)
    print(f"Número de clases: {len(classes)}")


    # Construir el modelo
    model, base = build_model(num_classes=len(classes))
    model.summary()

    # Fase de extracción de características (entrenamiento inicial)
    hist, best_ckpt = compile_and_train(model, train_ds, val_ds, class_weight,
                                        epochs=EPOCHS_INITIAL, lr=1e-3,
                                        checkpoint="feature_extraction.keras")

    # Calcular la época inicial para la siguiente fase (si existiera fine-tuning)
    start_epoch = len(hist.history["loss"])
    print(f"\nFase inicial completada. Época de inicio para fine-tuning: {start_epoch}")


    # TODO: Llamar a la función fine_tune aquí si está implementada
    # Por ahora, solo usamos el mejor checkpoint de la fase inicial
    # hist_ft, best_ft = fine_tune(model, base, train_ds, val_ds,
    #                              class_weight, start_epoch, epochs=EPOCHS_FINE_TUNE, lr=1e-4)

    # Determinar la ruta del mejor modelo (actualmente solo el de extracción de características)
    # Si fine_tune estuviera implementado, se elegiría entre best_ft y best_ckpt
    best_model_path = best_ckpt
    print(f"Usando el mejor modelo de la fase inicial: {best_model_path}")

    # Evaluar el mejor modelo y guardarlo en formato Keras
    # Cambiamos el nombre del archivo final para reflejar el formato SavedModel
    final_model = evaluate_and_save(best_model_path, val_ds, final_path="final_model_tf")

    # Convertir el modelo final a TFLite
    # Asegúrate de que la conversión a TFLite sea compatible con el modelo guardado en SavedModel
    # converter = tf.lite.TFLiteConverter.from_saved_model(final_model_path) # Podrías necesitar cargar desde la ruta si from_keras_model falla con el modelo cargado
    convert_to_tflite(final_model)


if __name__ == "__main__":
    main()
