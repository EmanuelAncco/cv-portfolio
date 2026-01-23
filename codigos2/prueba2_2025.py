import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from PIL import Image, ImageTk

#############################
# 1) Cargar el modelo
#############################
MODEL_PATH = r"C:\Users\Emanuel\PyCharmMiscProject\cnn2025.h5"
model = tf.keras.models.load_model(MODEL_PATH)


#############################
# 2) Preprocesamiento (Sobel)
#############################
def preprocess_image_for_inference(image_bgr):
    """
    Recibe una imagen BGR de OpenCV (de cualquier tamaño),
    la redimensiona a (224,224), aplica Sobel y
    devuelve un tensor (1,224,224,3) en [0..1].
    """
    # 1) Redimensionar
    image_bgr = cv2.resize(image_bgr, (224, 224))

    # 2) Convertir a RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 3) Escala de grises
    gray = tf.image.rgb_to_grayscale(image_rgb)  # (224,224,1)
    gray_expanded = tf.expand_dims(gray, axis=0)  # (1,224,224,1)

    # 4) Sobel
    sobel = tf.image.sobel_edges(gray_expanded)  # (1,224,224,1,2)
    sobel = tf.squeeze(sobel, axis=0)  # (224,224,1,2)
    grad_x = sobel[..., 0]
    grad_y = sobel[..., 1]

    magnitude = tf.sqrt(tf.square(grad_x) + tf.square(grad_y))  # (224,224,1)

    # 5) Normalizar a [0..255]
    min_val = tf.reduce_min(magnitude)
    max_val = tf.reduce_max(magnitude)
    eps = 1e-5
    magnitude = (magnitude - min_val) / (max_val - min_val + eps) * 255.0

    # 6) Replicar a 3 canales y escalar a [0..1]
    magnitude_3ch = tf.tile(magnitude, [1, 1, 3]) / 255.0  # (224,224,3)

    # 7) Expandir dimensión batch => (1,224,224,3)
    magnitude_3ch = tf.expand_dims(magnitude_3ch, axis=0)

    return magnitude_3ch.numpy()


#############################
# 3) Función para analizar grieta
#############################
def analyze_crack(image_bgr, threshold=0.5, n_clusters=2):
    """
    - Aplica el preprocesamiento Sobel a la imagen BGR.
    - Predice si hay grieta con el modelo global `model`.
    - Si hay grieta, segmenta con K-Means y retorna la imagen sombreada.
    - Devuelve: (result_image_bgr, info_dict)
    """
    # Copia en 224x224 para el overlay
    original_224 = cv2.resize(image_bgr, (224, 224))

    # Preprocesar
    preprocessed = preprocess_image_for_inference(image_bgr)
    prediction = model.predict(preprocessed)[0][0]

    info = {"confidence": float(prediction), "crack_detected": False}

    if prediction > threshold:
        info["crack_detected"] = True

        # Para KMeans: canal 0 de preprocessed => bordes en [0..1], *255 => [0..255]
        edges_single_channel = (preprocessed[0, ..., 0] * 255.0).astype(np.uint8)
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

        # Sombrear en color rojo (BGR)
        overlay = original_224.copy()
        overlay[crack_mask == 1] = (0, 0, 255)  # rojo BGR
        # Mezclar
        final_result = cv2.addWeighted(overlay, 0.5, original_224, 0.5, 0)

        return final_result, info
    else:
        # No grieta => devolvemos la imagen 224x224 tal cual
        return original_224, info


#############################
# 4) Creación de la GUI con Tkinter
#############################
class CrackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Grietas")

        # Variable para guardar la imagen BGR original
        self.original_bgr = None

        # Botón para cargar imagen
        self.btn_cargar = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        self.btn_cargar.pack(pady=5)

        # Botón para analizar
        self.btn_analizar = tk.Button(root, text="Analizar Grieta", command=self.analizar_imagen)
        self.btn_analizar.pack(pady=5)

        # Label para mostrar la imagen original/redimensionada
        self.lbl_imagen_original = tk.Label(root, text="Vista previa original/redimensionada")
        self.lbl_imagen_original.pack()

        # Label para mostrar la imagen con grieta sombreada
        self.lbl_imagen_resultado = tk.Label(root, text="Resultado análisis")
        self.lbl_imagen_resultado.pack()

    def cargar_imagen(self):
        """Abre un cuadro de diálogo para seleccionar la imagen y la muestra."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        # Cargar con OpenCV en BGR
        bgr_img = cv2.imread(file_path)
        if bgr_img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen.")
            return

        self.original_bgr = bgr_img

        # Redimensionar a 224x224 solo para la vista previa (NO hacemos Sobel aquí)
        preview = cv2.resize(bgr_img, (224, 224))
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        # Convertir a ImageTk
        pil_image = Image.fromarray(preview_rgb)
        img_tk = ImageTk.PhotoImage(pil_image)

        self.lbl_imagen_original.configure(image=img_tk)
        self.lbl_imagen_original.image = img_tk  # evitar recolección de basura

    def analizar_imagen(self):
        """Analiza la imagen con el modelo y muestra la imagen sombreada."""
        if self.original_bgr is None:
            messagebox.showwarning("Atención", "Primero carga una imagen.")
            return

        # Analizar
        result_bgr, info = analyze_crack(self.original_bgr)

        # Mostrar info
        if info["crack_detected"]:
            msg = (f"Grieta detectada!\n"
                   f"Confianza: {info['confidence']:.4f}")
        else:
            msg = (f"No hay grieta.\n"
                   f"Confianza: {info['confidence']:.4f}")
        messagebox.showinfo("Resultado", msg)

        # Convertir la imagen resultante (224x224 BGR) a RGB para mostrar
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
