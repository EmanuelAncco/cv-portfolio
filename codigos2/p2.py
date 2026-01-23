
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from sklearn.cluster import KMeans


#################################
# 1. Función de preprocesamiento (FIX)
#################################
def sobel_preprocessing(img):
    """
    Preprocesa cada imagen aplicando el filtro Sobel.
    """
    # Convertir a escala de grises
    gray = tf.image.rgb_to_grayscale(img)

    # Expande dimensión de batch para sobel_edges
    gray_expanded = tf.expand_dims(gray, axis=0)  # Agregar dimensión de batch: (1, h, w, 1)

    # Aplicar Sobel
    sobel = tf.image.sobel_edges(gray_expanded)  # (1, h, w, 1, 2)

    # Quitar dimensión de batch y separar gradientes
    sobel = tf.squeeze(sobel, axis=0)  # Ahora en forma (h, w, 1, 2)
    grad_x = sobel[..., 0]
    grad_y = sobel[..., 1]

    # Magnitud de bordes
    magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))  # (h, w, 1)

    # Normalizar a [0, 255]
    min_val = tf.reduce_min(magnitude)
    max_val = tf.reduce_max(magnitude)
    eps = 1e-5
    magnitude = (magnitude - min_val) / (max_val - min_val + eps) * 255.0

    # Replicar el canal para crear imagen RGB "sintética"
    magnitude_3ch = tf.tile(magnitude, [1, 1, 3])  # (h, w, 3)
    return magnitude_3ch


#################################
# El resto del código del modelo permanece igual
#################################

# Generadores de datos
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=sobel_preprocessing
)

train_generator = data_gen.flow_from_directory(
    r"D:\Python projectos 2025\CNN EMANUEL\archive\data\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    r"D:\Python projectos 2025\CNN EMANUEL\archive\data\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Modelo
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
model.save("cnn2025.h5")