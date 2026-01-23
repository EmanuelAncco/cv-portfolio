import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

#############################
# 1) Cargar el modelo
#############################
# Ajusta la ruta según tu carpeta
MODEL_PATH = r"C:\Users\Emanuel\PyCharmMiscProject\cnn2025.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")


#############################
# 2) Preprocesamiento (Sobel)
#############################
def preprocess_image_for_inference(image_path):
    """
    Reproduce la misma lógica de Sobel + normalización que usaste en entrenamiento.
    Deja la imagen resultante en (1,224,224,3) y [0..1].
    """
    # 1) Leer con OpenCV (BGR)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

    # 2) Cambiar tamaño a 224x224
    image = cv2.resize(image, (224, 224))

    # 3) Convertir a RGB (para que coincida con tf.image.rgb_to_grayscale)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # ---- Aplicar Sobel ----
    # a) Escala de grises
    gray = tf.image.rgb_to_grayscale(image_rgb)  # (224,224,1)
    gray_expanded = tf.expand_dims(gray, axis=0)  # (1,224,224,1)

    # b) Sobel
    sobel = tf.image.sobel_edges(gray_expanded)  # (1,224,224,1,2)
    sobel = tf.squeeze(sobel, axis=0)  # (224,224,1,2)
    grad_x = sobel[..., 0]
    grad_y = sobel[..., 1]

    magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))  # (224,224,1)

    # c) Normalizar [0..255]
    min_val = tf.reduce_min(magnitude)
    max_val = tf.reduce_max(magnitude)
    eps = 1e-5
    magnitude = (magnitude - min_val) / (max_val - min_val + eps) * 255.0

    # d) Replicar a 3 canales y reescalar a [0..1]
    magnitude_3ch = tf.tile(magnitude, [1, 1, 3]) / 255.0  # (224,224,3)

    # e) Expandir dimensión batch => (1,224,224,3)
    magnitude_3ch = tf.expand_dims(magnitude_3ch, axis=0)

    return magnitude_3ch.numpy()


#############################
# 3) Función para analizar grieta
#############################
def analyze_crack(image_path, model, threshold=0.5, n_clusters=2):
    """
    - Preprocesa la imagen con Sobel.
    - Predice si hay grieta (pred > threshold).
    - Si hay grieta, segmenta usando K-Means para resaltar la zona de grieta.
    - Retorna:
        1) labeled_image: matriz con los labels de cada píxel (cluster 0..n_clusters-1)
        2) original_image: la imagen original (224x224 BGR) con el contorno/área resaltada
        3) info: diccionario con mediciones (área, bounding box, etc.)
    """
    # Cargar imagen original también para sobreponer el resultado (BGR)
    original_bgr = cv2.imread(image_path)
    original_bgr = cv2.resize(original_bgr, (224, 224))

    # Preprocesar para predecir
    preprocessed = preprocess_image_for_inference(image_path)
    prediction = model.predict(preprocessed)[0][0]

    if prediction > threshold:
        print(f"Grieta detectada con confianza: {prediction:.4f}")

        # Para k-means, necesitamos el canal de bordes en [0..255], 1 canal
        edges_single_channel = (preprocessed[0, ..., 0] * 255.0).astype(np.uint8)  # (224,224)
        reshaped = edges_single_channel.reshape(-1, 1)  # (224*224,1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reshaped)
        labels = kmeans.labels_.reshape(edges_single_channel.shape)  # (224,224)

        # Suponiendo que la "grieta" corresponde al cluster con intensidades más altas
        # Calculamos la intensidad media en cada cluster
        cluster_means = []
        for cl in range(n_clusters):
            cluster_means.append(np.mean(edges_single_channel[labels == cl]))

        # Indice del cluster "crack" => aquel con mayor intensidad
        crack_cluster_idx = np.argmax(cluster_means)

        # Creamos una máscara booleana => True donde hay grieta
        crack_mask = (labels == crack_cluster_idx).astype(np.uint8)

        # OPCIÓN A: medir área = número de píxeles en crack_mask
        crack_area_pixels = np.sum(crack_mask)

        # OPCIÓN B: bounding box de la grieta
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Tomar el contorno más grande (asumiendo 1 grieta principal)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
        else:
            x, y, w, h = 0, 0, 0, 0

        # Sobrescribir en la imagen original con un color semitransparente, p.ej. color rojo
        overlay = original_bgr.copy()
        overlay[crack_mask == 1] = (0, 0, 255)  # BGR: rojo
        # Fusionar un 50% overlay
        final_result = cv2.addWeighted(overlay, 0.5, original_bgr, 0.5, 0)

        # info dict con mediciones
        info = {
            "prediction_confidence": float(prediction),
            "crack_area_pixels": int(crack_area_pixels),
            "bounding_box": (int(x), int(y), int(w), int(h))
        }

        return labels, final_result, info
    else:
        print(f"No se detectó grieta. Valor predicho: {prediction:.4f}")
        return None, original_bgr, {"prediction_confidence": float(prediction)}


#############################
# 4) Ejemplo de uso
#############################
def main():
    # Ruta de la imagen que queremos analizar
    test_image_path = r"C:\Users\Emanuel\PyCharmMiscProject\imagenes_test\2.jpg"

    labeled_image, result_image, info = analyze_crack(test_image_path, model)

    if labeled_image is not None:
        print("=== Métricas de la grieta ===")
        print(f" - Confianza de predicción: {info['prediction_confidence']:.4f}")
        print(f" - Área en píxeles: {info['crack_area_pixels']}")
        x, y, w, h = info["bounding_box"]
        print(f" - Bounding box (x={x}, y={y}, w={w}, h={h})")
    else:
        print("Sin grieta. Métricas no disponibles.")

    # Mostrar la imagen final con grieta sombreada (si se encontró)
    cv2.imshow("Resultado", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
