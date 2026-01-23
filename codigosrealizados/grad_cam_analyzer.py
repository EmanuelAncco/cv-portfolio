import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO
import logging
import os
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

# --- CONFIGURACIÓN DEL ANÁLISIS ---
MODEL_PATH = 'runs/detect/emarc_vision_v3.1_finetune_manual_lr/weights/best.pt'
YAML_PATH = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced\data.yaml'
# --- Actualiza esta ruta a la imagen que quieres analizar ---
IMAGE_PATH = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced\train\images\image80_jpg.rf.f027b9164a49896597c3a0715ccbd87f.jpg'
OUTPUT_DIR = 'final_cam_analysis'
# --- Analiza múltiples clases a la vez ---
CLASSES_TO_EXPLAIN = ['Human', 'Gloves', 'Boots']


class YOLOV8Target:
    """
    Clase objetivo para los métodos CAM en YOLOv8.
    Extrae la confianza máxima para una clase específica de la salida del modelo.
    """

    def __init__(self, class_id):
        self.class_id = class_id

    def __call__(self, model_output):
        output = model_output[0]
        if output.dim() == 2:
            output = output.unsqueeze(0)
        output = output.permute(0, 2, 1)

        objectness = output[:, :, 4]
        class_confidence = output[:, :, 5 + self.class_id]
        final_confidence = objectness * class_confidence
        return final_confidence.max(axis=-1)[0]


class DetectionCAMAnalyzer:
    """
    Genera visualizaciones de Eigen-CAM para entender la localización de la atención de YOLOv8.
    """

    def __init__(self, model_path, yaml_path):
        logging.info(f"Cargando modelo YOLOv8 desde: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(model_path)
        # Obtenemos el modelo de PyTorch subyacente para el análisis CAM
        self.model = self.yolo_model.model.to(self.device)
        self.model.eval()

        logging.info(f"Cargando información de clases desde: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.class_names = data['names']
        logging.info(f"Clases cargadas: {self.class_names}")

    def get_class_index(self, class_name):
        try:
            return self.class_names.index(class_name)
        except ValueError:
            logging.error(f"La clase '{class_name}' no se encuentra en el archivo YAML.")
            return None

    def analyze_and_create_visuals(self, image_path, classes_to_explain):
        """
        Genera una imagen comparativa con la detección original y el mapa de calor de Eigen-CAM.
        """
        logging.info(f"Iniciando análisis de localización con Eigen-CAM para la imagen: {image_path}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logging.error(f"No se pudo cargar la imagen en: {image_path}")
            return

        # 1. Obtener la detección original del "Laboratorio de Inferencia"
        logging.info("Generando predicción original...")
        results = self.yolo_model.predict(source=img_bgr, conf=0.25, verbose=False)
        annotated_img = results[0].plot()
        annotated_img_resized = cv2.resize(annotated_img, (640, 640))

        # Preparar la imagen para el análisis CAM
        img_resized = cv2.resize(img_bgr, (640, 640))
        input_tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).unsqueeze(0).float().to(
            self.device) / 255.0
        rgb_img_float = np.float32(img_resized) / 255

        # La capa objetivo sigue siendo la 9, la última del 'backbone'
        target_layer = [self.model.model[9]]

        cam_results = {}

        # 2. Generar mapa de calor Eigen-CAM para cada clase
        for class_name in classes_to_explain:
            class_id = self.get_class_index(class_name)
            if class_id is None: continue

            logging.info(f"Analizando clase '{class_name}' con Eigen-CAM...")
            targets = [YOLOV8Target(class_id)]

            # --- CORRECCIÓN CLAVE (v1.3): Se elimina el argumento 'use_cuda' ---
            # Las versiones más recientes de la librería pytorch-grad-cam gestionan
            # automáticamente el dispositivo (CPU/GPU) a través del modelo, por lo que
            # el parámetro 'use_cuda' ya no es necesario y causaba el error.
            with EigenCAM(model=self.model, target_layers=target_layer) as cam:
                grayscale_eigencam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                eigencam_viz = show_cam_on_image(rgb_img_float, grayscale_eigencam, use_rgb=True)

            cam_results[class_name] = eigencam_viz

        # 3. Crear la imagen comparativa final
        self.create_comparison_image(image_path, annotated_img_resized, cam_results)

    def create_comparison_image(self, image_path, annotated_img, cam_results):
        """
        Ensambla las imágenes generadas en una sola visualización comparativa.
        """
        num_classes = len(cam_results)
        if num_classes == 0:
            logging.warning("No se generó ningún mapa de calor.")
            return

        annotated_img_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

        # Crear un canvas para mostrar la imagen original, la detección y un mapa de calor por clase
        canvas_width = 640 * (2 + num_classes)
        canvas_height = 640 + 80  # Espacio para títulos
        comparison_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # Añadir la imagen original
        cv2.putText(comparison_canvas, "Original", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        comparison_canvas[80:, 0:640] = cv2.resize(cv2.imread(image_path), (640, 640))

        # Añadir la imagen con las detecciones
        cv2.putText(comparison_canvas, "Deteccion (Laboratorio)", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),
                    3)
        comparison_canvas[80:, 640:1280] = annotated_img_bgr

        # Añadir los resultados de Eigen-CAM
        offset = 1280
        for class_name, eigencam_viz in cam_results.items():
            eigencam_bgr = cv2.cvtColor(eigencam_viz, cv2.COLOR_RGB2BGR)

            cv2.putText(comparison_canvas, f"Eigen-CAM: '{class_name}'", (offset + 10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 0), 3)
            comparison_canvas[80:, offset:offset + 640] = eigencam_bgr
            offset += 640

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_eigencam_comparison.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, comparison_canvas)
        logging.info(f"¡Análisis comparativo completado! Imagen guardada en: {output_path}")


if __name__ == '__main__':
    # Nota: He renombrado el script a 'detection_cam_analyzer.py' para reflejar mejor su función.
    # Puedes renombrar tu archivo localmente si lo deseas.
    if not os.path.exists(MODEL_PATH) or not os.path.exists(YAML_PATH):
        logging.error("Error: La ruta del modelo o del archivo YAML no son válidas.")
    elif not os.path.exists(IMAGE_PATH):
        logging.error(f"Error: La ruta de la imagen '{IMAGE_PATH}' no es válida.")
    else:
        analyzer = DetectionCAMAnalyzer(MODEL_PATH, YAML_PATH)
        analyzer.analyze_and_create_visuals(IMAGE_PATH, CLASSES_TO_EXPLAIN)
