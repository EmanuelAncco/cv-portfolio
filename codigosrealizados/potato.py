#!/usr/bin/env python3
"""
train_potato_blight.py

Pipeline to train a classifier for Potato Early/Late blight and healthy leaves
using TensorFlow/Keras with transfer learning and fine‑tuning.

Usage:
    python train_potato_blight.py --data_dir /absolute/path/to/PlantVillage/Potato___Subset
"""

import os
import random
import argparse
import numpy as np
import tensorflow as tf

SEED = 123
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 50
EPOCHS_FINE_TUNE = 20
RESCALE_RANGE = 1.0 / 255.0


def set_seeds(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_datasets(
    data_dir: str,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split: float = 0.2,
    seed: int = SEED,
):
    """Load PlantVillage subset, return tf.data datasets and class weights."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # First pass (no shuffle) to compute class weights
    base_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = base_ds.class_names
    labels = np.concatenate([y for _, y in base_ds], axis=0)
    counts = labels.sum(axis=0)
    class_weight = {
        i: (1.0 / counts[i]) * (counts.sum() / len(counts))
        for i in range(len(counts))
    }

    # Reload with shuffling for real training
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names, class_weight


def build_model(num_classes: int, img_shape=IMG_SIZE + (3,)):
    """Construct EfficientNetB0‑based classifier."""
    rescale = tf.keras.layers.Rescaling(RESCALE_RANGE, name="rescale")

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_shape, name="input")
    x = rescale(inputs)
    x = data_augmentation(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="potato_blight_classifier")
    return model, base_model


def compile_and_train(
    model,
    train_ds,
    val_ds,
    class_weight,
    initial_epochs=EPOCHS_INITIAL,
    lr=1e-3,
    checkpoint_path="feature_extraction.keras",
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=[ckpt_cb, es_cb],
        class_weight=class_weight,
    )
    return history, ckpt_cb.filepath


def fine_tune(
    model,
    base_model,
    train_ds,
    val_ds,
    class_weight,
    start_epoch,
    fine_epochs=EPOCHS_FINE_TUNE,
    lr=1e-5,
    checkpoint_path="fine_tune.keras",
    unfreeze_from_layer_name="block6a_expand_conv",
):
    base_model.trainable = True

    found = False
    for layer in base_model.layers:
        if layer.name == unfreeze_from_layer_name:
            found = True
        if not found:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=start_epoch + fine_epochs,
        initial_epoch=start_epoch,
        callbacks=[ckpt_cb, es_cb],
        class_weight=class_weight,
    )
    return history, ckpt_cb.filepath


def evaluate_and_save(
    best_model_path, val_ds, final_model_path="final_model.keras"
):
    model = tf.keras.models.load_model(best_model_path)
    loss, acc = model.evaluate(val_ds)
    print(f"Validation Loss: {loss:.4f} — Accuracy: {acc:.4f}")
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    return model


def convert_to_tflite(model, output_path="model.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Potato Blight Classifier"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to PlantVillage/Potato___Subset directory",
    )
    parser.add_argument(
        "--skip_tflite", action="store_true", help="Do not export TFLite model"
    )
    args = parser.parse_args()

    set_seeds()

    train_ds, val_ds, class_names, class_weight = prepare_datasets(
        args.data_dir
    )
    print(f"Classes: {class_names}")
    print(f"Class weights: {class_weight}")

    model, base_model = build_model(num_classes=len(class_names))
    model.summary()

    history, best_feature_ckpt = compile_and_train(
        model, train_ds, val_ds, class_weight
    )
    start_epoch = len(history.history["loss"])

    history_ft, best_ft_ckpt = fine_tune(
        model,
        base_model,
        train_ds,
        val_ds,
        class_weight,
        start_epoch,
    )

    best_model_path = (
        best_ft_ckpt if os.path.exists(best_ft_ckpt) else best_feature_ckpt
    )
    final_model = evaluate_and_save(best_model_path, val_ds)

    if not args.skip_tflite:
        convert_to_tflite(final_model)


if __name__ == "__main__":
    main()
