# -*- coding: utf-8 -*-
"""
Aplicación de escritorio con GUI para la inferencia del modelo detector de aves.

Esta aplicación:
1. Carga el modelo DETR entrenado previamente.
2. Proporciona una interfaz para que el usuario cargue una imagen.
3. Realiza la inferencia sobre la imagen.
4. Muestra la imagen con los cuadros delimitadores (bounding boxes) de las aves detectadas.

Dependencias: pip install customtkinter torch transformers Pillow timm
"""
import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Asegúrate de que esta ruta apunte a la carpeta donde guardaste tu modelo entrenado.
MODEL_PATH = r"D:\Python_proyectos_2025\AVES\HamaBurung.v84i.yolov11\detr-bird-detector-final"
CONFIDENCE_THRESHOLD = 0.6  # Umbral de confianza para mostrar una detección (0.0 a 1.0)


# --- CLASE PRINCIPAL DE LA APLICACIÓN ---

class BirdDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuración de la Ventana Principal ---
        self.title("Detector de Aves - Inferencia")
        self.geometry("900x700")
        self.minsize(600, 500)
        ctk.set_appearance_mode("System")  # Puede ser "Dark", "Light"
        ctk.set_default_color_theme("blue")

        # --- Atributos de la clase ---
        self.model = None
        self.image_processor = None
        self.original_image = None
        self.ctk_image = None

        # --- Diseño de la Interfaz (Layout) ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Frame superior para los controles
        self.control_frame = ctk.CTkFrame(self, height=80)
        self.control_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.control_frame.grid_columnconfigure((0, 1), weight=1)

        # Frame central para la imagen
        self.image_frame = ctk.CTkFrame(self, fg_color="gray20")
        self.image_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.image_frame.grid_propagate(False)  # Evita que el frame cambie de tamaño
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        # Frame inferior para el estado
        self.status_frame = ctk.CTkFrame(self, height=40)
        self.status_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="ew")

        # --- Widgets (Controles) ---
        # Botones
        self.btn_load_image = ctk.CTkButton(self.control_frame, text="Cargar Imagen", command=self.load_image)
        self.btn_load_image.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.btn_analyze = ctk.CTkButton(self.control_frame, text="Analizar Imagen", command=self.analyze_image,
                                         state="disabled")
        self.btn_analyze.grid(row=0, column=1, padx=20, pady=20, sticky="e")

        # Etiqueta para mostrar la imagen
        self.image_label = ctk.CTkLabel(self.image_frame, text="Carga una imagen para comenzar", text_color="gray70",
                                        font=("Arial", 16))
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Etiqueta de estado
        self.status_label = ctk.CTkLabel(self.status_frame, text="Cargando modelo, por favor espera...")
        self.status_label.pack(side="left", padx=10, pady=5)

        # --- Cargar el modelo al iniciar ---
        self.after(100, self.load_model)

    def load_model(self):
        """Carga el modelo y el procesador de imágenes en un hilo separado."""
        self.status_label.configure(text="Cargando modelo...")
        self.update_idletasks()  # Actualiza la GUI

        try:
            if not os.path.exists(MODEL_PATH):
                self.status_label.configure(text=f"Error: No se encontró el directorio del modelo en {MODEL_PATH}",
                                            text_color="red")
                return

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(self.device)
            self.image_processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
            self.model.eval()  # Poner el modelo en modo de evaluación
            self.status_label.configure(text=f"Modelo cargado en {self.device}. Listo para analizar.",
                                        text_color="green")

        except Exception as e:
            self.status_label.configure(text=f"Error al cargar el modelo: {e}", text_color="red")
            print(f"Error detallado: {e}")

    def load_image(self):
        """Abre un diálogo para seleccionar un archivo de imagen y lo muestra."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        self.original_image = Image.open(file_path).convert("RGB")
        self.display_image(self.original_image)
        self.btn_analyze.configure(state="normal")
        self.status_label.configure(text=f"Imagen cargada: {os.path.basename(file_path)}", text_color="gray70")

    def display_image(self, pil_image):
        """Muestra una imagen PIL en la etiqueta de la GUI, ajustándola al tamaño."""
        frame_w, frame_h = self.image_frame.winfo_width(), self.image_frame.winfo_height()

        # Redimensionar imagen para que quepa en el frame manteniendo la relación de aspecto
        img_w, img_h = pil_image.size
        ratio = min(frame_w / img_w, frame_h / img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))

        # Crear CTkImage y mostrarla
        self.ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=new_size)
        self.image_label.configure(image=self.ctk_image, text="")  # Quitar el texto de placeholder

    def analyze_image(self):
        """Realiza la inferencia sobre la imagen cargada."""
        if not self.original_image or not self.model:
            return

        self.status_label.configure(text="Analizando...", text_color="orange")
        self.btn_analyze.configure(state="disabled")
        self.update_idletasks()

        try:
            # Procesar la imagen
            inputs = self.image_processor(images=self.original_image, return_tensors="pt").to(self.device)

            # Realizar inferencia
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-procesar resultados
            target_sizes = torch.tensor([self.original_image.size[::-1]], device=self.device)
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes
            )[0]

            # Dibujar los resultados
            self.draw_results(results)
            num_detections = len(results["scores"])
            self.status_label.configure(text=f"Análisis completo. Se encontraron {num_detections} aves.",
                                        text_color="green")

        except Exception as e:
            self.status_label.configure(text=f"Error durante el análisis: {e}", text_color="red")
            print(f"Error detallado: {e}")
        finally:
            self.btn_analyze.configure(state="normal")

    def draw_results(self, results):
        """Dibuja los cuadros delimitadores sobre la imagen original."""
        image_with_boxes = self.original_image.copy()
        draw = ImageDraw.Draw(image_with_boxes)

        try:
            # Usar una fuente TrueType si está disponible para mejor apariencia
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            # Dibujar el cuadro
            draw.rectangle(box, outline="#059669", width=3)

            # Dibujar etiqueta con confianza
            label_text = f"Ave: {score.item():.2f}"
            text_position = (box[0], box[1] - 25)

            # Dibujar fondo para el texto
            text_bbox = draw.textbbox(text_position, label_text, font=font)
            draw.rectangle(text_bbox, fill="#059669")
            draw.text(text_position, label_text, fill="white", font=font)

        self.display_image(image_with_boxes)


# --- PUNTO DE ENTRADA DEL PROGRAMA ---
if __name__ == "__main__":
    app = BirdDetectorApp()
    app.mainloop()
