import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from sklearn.cluster import KMeans


#################################
# 1. Función de preprocesamiento
#################################
def sobel_preprocessing(img):
    """
    Esta función será llamada automáticamente por el ImageDataGenerator
    sobre cada imagen cargada. 'img' es un tensor 3D en [0, 255] con 3 canales.
    Aquí convertimos a gris, aplicamos Sobel y replicamos a 3 canales.
    """
    # img vendrá como float32 en rango [0,255] si no usas rescale en el generator,
    # o [0,1] si sí usas rescale. Vamos a suponer que si vas a usar rescale=1/255,
    # entonces en este punto 'img' aún está en [0,255].

    # 1) Convertir a escala de grises
    #   tf.image.rgb_to_grayscale convierte (h, w, 3) -> (h, w, 1)
    gray = tf.image.rgb_to_grayscale(img)

    # 2) Aplicar Sobel (devuelve (h, w, 1, 2) => grad_x, grad_y)
    sobel = tf.image.sobel_edges(gray)
    grad_x = sobel[..., 0]
    grad_y = sobel[..., 1]

    # 3) Magnitud de bordes = sqrt(dx^2 + dy^2)
    magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))  # (h, w, 1)

    # 4) Normalizar a [0, 255] manualmente
    min_val = tf.reduce_min(magnitude)
    max_val = tf.reduce_max(magnitude)
    # Añadir un pequeño epsilon para evitar división por cero
    eps = 1e-5
    magnitude = (magnitude - min_val) / (max_val - min_val + eps) * 255.0

    # 5) Replicar el canal a 3 canales para que ResNet50 lo asuma como RGB
    magnitude_3ch = tf.tile(magnitude, [1, 1, 3])  # (h, w, 3)
    return magnitude_3ch


#################################
# 2. Generadores de datos
#################################
data_gen = ImageDataGenerator(
    # 1) reescala a [0,1] después de nuestro preprocessing_function
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # 80% train, 20% valid
    preprocessing_function=sobel_preprocessing  # Aplicamos Sobel a todo
)

train_generator = data_gen.flow_from_directory(    r"D:\Python projectos 2025\CNN EMANUEL\archive\data\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = data_gen.flow_from_directory(    r"D:\Python projectos 2025\CNN EMANUEL\archive\data\train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#################################
# 3. Definir y compilar el modelo
#################################
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelamos capas base (puedes descongelar después)

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#################################
# 4. Entrenamiento
#################################
history = model.fit(    train_generator,
    validation_data=val_generator,
    epochs=10
)


#################################
# 5. Análisis de grietas (inferencia)
#################################
def preprocess_image_for_inference(image_path):
    """
    Preprocesa manualmente una imagen en disco para predecir con el mismo pipeline
    usado en el entrenamiento: Sobel + normalización a 3 canales + escala [0,1].
    """
    # 1) Leer con OpenCV (BGR)
    image = cv2.imread(image_path)
    # 2) Cambiar tamaño a 224x224
    image = cv2.resize(image, (224, 224))
    # 3) Convertir a RGB para ser consistentes con tf.image.rgb_to_grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 4) Convertir a tensor
    img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # 5) Aplicar la misma lógica de `sobel_preprocessing`
    gray = tf.image.rgb_to_grayscale(img_tensor)
    sobel = tf.image.sobel_edges(gray)
    grad_x = sobel[..., 0]
    grad_y = sobel[..., 1]
    magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))

    min_val = tf.reduce_min(magnitude)
    max_val = tf.reduce_max(magnitude)
    eps = 1e-5
    magnitude = (magnitude - min_val) / (max_val - min_val + eps) * 255.0

    magnitude_3ch = tf.tile(magnitude, [1, 1, 3])  # (224,224,3)

    # 6) Convertir a [0,1] igual que en el training (rescale=1/255)
    magnitude_3ch = magnitude_3ch / 255.0

    # 7) Expande dimensión (batch=1)
    magnitude_3ch = tf.expand_dims(magnitude_3ch, axis=0)  # (1,224,224,3)

    return magnitude_3ch


def analyze_crack(image_path, model, n_clusters=2):
    """Usa la misma lógica de preprocesamiento para predecir y luego aplica K-Means."""
    preprocessed = preprocess_image_for_inference(image_path)  # (1,224,224,3)
    prediction = model.predict(preprocessed)[0][0]

    # Para el clustering, conviene usar el canal simple de bordes en rango [0..255]
    # Lo sacamos de 'preprocessed' (que está en [0..1]) y tomamos 1 canal:
    edges_single_channel = preprocessed[0, ..., 0] * 255.0  # (224,224), float32
    edges_single_channel = edges_single_channel.numpy().astype(np.uint8)

    if prediction > 0.5:
        print("Crack detected with confidence:", prediction)
        reshaped = edges_single_channel.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(reshaped)
        labeled_image = kmeans.labels_.reshape(edges_single_channel.shape)
        return labeled_image
    else:
        print("No crack detected.")
        return None


# Ejemplo de uso
test_image_path = r"D:\Python projectos 2025\CNN EMANUEL\archive\data\test\sample.jpg"
labeled_crack = analyze_crack(test_image_path, model)
if labeled_crack is not None:
    cv2.imshow('Labeled Crack', labeled_crack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
