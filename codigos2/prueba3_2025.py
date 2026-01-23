import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from PIL import Image, ImageTk

#############################
# 1) Parámetros de entrenamiento
#############################
# Ajusta la resolución al tamaño que usaste en tu dataset
TARGET_SIZE = (224, 224)

# Cargar tu modelo .h5 entrenado
MODEL_PATH = r"C:\Users\Emanuel\PyCharmMiscProject\cnn2025.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")


#############################
# 2) Preprocesamiento Sobel adaptado a (227,227)
#############################
def preprocess_image_for_inference(image_bgr):
    """
    Recibe una imagen BGR de OpenCV (de cualquier tamaño).
    La redimensiona a TARGET_SIZE (227x227),
    aplica Sobel y la deja en un tensor (1,227,227,3) normalizado en [0..1].
    """
    # 1) Redimensionar a 227x227
    image_bgr = cv2.resize(image_bgr, TARGET_SIZE)

    # 2) Convertir a RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 3) Escala de grises
    gray = tf.image.rgb_to_grayscale(image_rgb)  # (227,227,1)
    gray_expanded = tf.expand_dims(gray, axis=0)  # (1,227,227,1)

    # 4) Sobel
    sobel = tf.image.sobel_edges(gray_expanded)  # (1,227,227,1,2)
    sobel = tf.squeeze(sobel, axis=0)  # (227,227,1,2)
    grad_x = sobel[..., 0]
    grad_y = sobel[..., 1]

    # Magnitud de Sobel
    magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))  # (227,227,1)

    # 5) Normalizar a [0..255]
    min_val = tf.reduce_min(magnitude)
    max_val = tf.reduce_max(magnitude)
    eps = 1e-5
    magnitude = (magnitude - min_val) / (max_val - min_val + eps) * 255.0

    # 6) Replicar a 3 canales + escalar a [0..1] (simulando rescale=1/255)
    magnitude_3ch = tf.tile(magnitude, [1, 1, 3]) / 255.0  # (227,227,3)

    # 7) Expandir dimensión para batch => (1,227,227,3)
    magnitude_3ch = tf.expand_dims(magnitude_3ch, axis=0)

    return magnitude_3ch.numpy()


#############################
# 3) Análisis de grieta con K-Means
#############################
def analyze_crack(image_bgr, threshold=0.5, n_clusters=2):
    """
    - Aplica el preprocesamiento Sobel a la imagen BGR.
    - Predice si hay grieta con `model`.
    - Si hay grieta, segmenta con K-Means y retorna la imagen sombreada.
    - Retorna: (result_image_bgr, info_dict)
    """
    # Guardar copia en 227x227 para el overlay
    original_227 = cv2.resize(image_bgr, TARGET_SIZE)

    # Preprocesar
    preprocessed = preprocess_image_for_inference(image_bgr)
    prediction = model.predict(preprocessed)[0][0]

    info = {"confidence": float(prediction), "crack_detected": False}

    if prediction > threshold:
        info["crack_detected"] = True

        # El canal 0 de preprocessed es la magnitud de bordes en [0..1]; lo pasamos a [0..255]
        edges_single_channel = (preprocessed[0, ..., 0] * 255.0).astype(np.uint8)  # (227,227)
        reshaped = edges_single_channel.reshape(-1, 1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reshaped)
        labels = kmeans.labels_.reshape(edges_single_channel.shape)

        # cluster con mayor intensidad => grieta
        cluster_means = []
        for cl in range(n_clusters):
            cluster_means.append(np.mean(edges_single_channel[labels == cl]))
        crack_cluster_idx = np.argmax(cluster_means)

        # Máscara binaria
        crack_mask = (labels == crack_cluster_idx).astype(np.uint8)

        # Sombrear la grieta en color rojo (BGR)
        overlay = original_227.copy()
        overlay[crack_mask == 1] = (0, 0, 255)  # rojo BGR
        # Mezclar a 50%
        final_result = cv2.addWeighted(overlay, 0.5, original_227, 0.5, 0)

        return final_result, info
    else:
        # No grieta => devolver la imagen 227x227 sin cambios
        return original_227, info


#############################
# 4) Creación de la GUI con Tkinter (ajustada a 227x227)
#############################
class CrackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Grietas")

        self.original_bgr = None

        # Botón para cargar imagen
        self.btn_cargar = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        self.btn_cargar.pack(pady=5)

        # Botón para analizar
        self.btn_analizar = tk.Button(root, text="Analizar Grieta", command=self.analizar_imagen)
        self.btn_analizar.pack(pady=5)

        # Label para mostrar imagen original (reducida a 227x227)
        self.lbl_imagen_original = tk.Label(root, text="Vista previa original/redimensionada")
        self.lbl_imagen_original.pack()

        # Label para mostrar resultado
        self.lbl_imagen_resultado = tk.Label(root, text="Resultado análisis")
        self.lbl_imagen_resultado.pack()

    def cargar_imagen(self):
        """Seleccionar imagen con file dialog y mostrar vista previa (224x224)."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        bgr_img = cv2.imread(file_path)
        if bgr_img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return

        self.original_bgr = bgr_img

        # Redimensionar a 227x227 solo para la vista previa
        preview = cv2.resize(bgr_img, TARGET_SIZE)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(preview_rgb)
        img_tk = ImageTk.PhotoImage(pil_image)

        self.lbl_imagen_original.configure(image=img_tk)
        self.lbl_imagen_original.image = img_tk

    def analizar_imagen(self):
        """Analiza la imagen con el modelo y muestra la imagen sombreada."""
        if self.original_bgr is None:
            messagebox.showwarning("Atención", "Primero carga una imagen.")
            return

        result_bgr, info = analyze_crack(self.original_bgr)

        if info["crack_detected"]:
            msg = f"Grieta detectada!\nConfianza: {info['confidence']:.4f}"
        else:
            msg = f"No hay grieta.\nConfianza: {info['confidence']:.4f}"
        messagebox.showinfo("Resultado", msg)

        # Mostrar imagen resultante (227x227) en la GUI
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(result_rgb)
        img_tk_result = ImageTk.PhotoImage(pil_result)

        self.lbl_imagen_resultado.configure(image=img_tk_result)
        self.lbl_imagen_resultado.image = img_tk_result


def main():
    root = tk.Tk()
    app = CrackGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
