import cv2
import torch
import numpy as np
import shap
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import logging
import os

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

# --- CONFIGURACIÓN DEL ANÁLISIS ---
MODEL_PATH = 'runs/detect/emarc_vision_v3.1_finetune_manual_lr/weights/best.pt'
YAML_PATH = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced\data.yaml'
# --- CAMBIO: Actualiza esta ruta a la imagen que quieras analizar, recortada o no ---
IMAGE_PATH = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced\train\images\1ed34c16-aee2-4896-afa2-d3083a4231c4_jpg.rf.4c078dd581e8100b216f859873dfb202.jpg'
OUTPUT_DIR = 'xai_analysis_results'
CLASS_TO_EXPLAIN = 'No-Gloves'
MAX_EVALS = 5000


class YoloV8Explainer:
    """
    Clase que encapsula la lógica para explicar las predicciones de un modelo YOLOv8 usando SHAP.
    """

    def __init__(self, model_path, yaml_path):
        """
        Inicializa el explicador cargando el modelo y las clases.
        """
        logging.info(f"Cargando modelo YOLOv8 desde: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(model_path)
        self.model = self.yolo_model.model.to(self.device)
        self.model.eval()
        logging.info("Modelo puesto en modo de evaluación correctamente.")

        logging.info(f"Cargando información de clases desde: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        self.class_names = data['names']
        logging.info(f"Clases cargadas: {self.class_names}")

    def get_class_index(self, class_name):
        """Obtiene el índice numérico de una clase a partir de su nombre."""
        try:
            return self.class_names.index(class_name)
        except ValueError:
            logging.error(f"La clase '{class_name}' no se encuentra en el archivo YAML.")
            return None

    def predict_wrapper(self, images):
        """
        Función envoltorio (wrapper) que SHAP utilizará para obtener las predicciones.
        """
        images_tensor = torch.from_numpy(images.copy()).to(self.device)
        images_tensor = images_tensor.permute(0, 3, 1, 2).float() / 255.0

        with torch.no_grad():
            results = self.model(images_tensor)

        preds = results[0]
        output_confidences = []
        target_class_idx = self.get_class_index(CLASS_TO_EXPLAIN)
        if target_class_idx is None:
            return np.zeros(images.shape[0])

        preds_transposed = preds.permute(0, 2, 1)

        for i in range(preds_transposed.shape[0]):
            img_preds = preds_transposed[i]
            max_confidence = 0.0
            class_confidences = img_preds[:, 5 + target_class_idx]
            objectness_scores = img_preds[:, 4]
            final_confidences = class_confidences * objectness_scores

            if len(final_confidences) > 0:
                max_confidence = torch.max(final_confidences).item()

            output_confidences.append(max_confidence)

        return np.array(output_confidences)

    def explain(self, image_path, class_to_explain):
        """
        Genera y guarda la explicación SHAP para una imagen y una clase dadas.
        """
        logging.info(f"Iniciando análisis SHAP para la imagen: {image_path}")
        logging.info(f"Explicando la clase: '{class_to_explain}'")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            logging.error(f"No se pudo cargar la imagen en: {image_path}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- CORRECCIÓN CLAVE (v1.3): Redimensionar la imagen a un tamaño compatible con YOLO ---
        # Los modelos YOLO son sensibles a las dimensiones de entrada. Al redimensionar la imagen
        # al tamaño estándar de entrenamiento (640x640) ANTES de que SHAP la procese, nos
        # aseguramos de que todas las particiones y máscaras generadas internamente tengan
        # dimensiones "limpias", evitando errores de desajuste de tensores en las capas profundas
        # de la red neuronal.
        logging.info(f"Redimensionando imagen de {img_rgb.shape[:2]} a (640, 640) para el análisis SHAP.")
        img_rgb_resized = cv2.resize(img_rgb, (640, 640))

        # El explicador SHAP ahora trabajará sobre la imagen redimensionada y estandarizada.
        masker = shap.maskers.Image("inpaint_telea", img_rgb_resized.shape)
        explainer = shap.Explainer(self.predict_wrapper, masker)

        logging.info(f"Calculando valores SHAP... Esto puede tardar varios minutos (max_evals={MAX_EVALS}).")
        shap_values = explainer(
            np.expand_dims(img_rgb_resized.copy(), axis=0),
            max_evals=MAX_EVALS,
            batch_size=50
        )
        logging.info("Cálculo de SHAP completado.")

        plt.figure()
        shap.image_plot(
            shap_values=shap_values.values,
            pixel_values=shap_values.data,
            show=False
        )

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_shap_{class_to_explain}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"¡Análisis completado! Gráfico guardado en: {output_path}")


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(YAML_PATH):
        logging.error(
            "Error: La ruta del modelo o del archivo YAML no son válidas. Revisa la sección de CONFIGURACIÓN.")
    elif not os.path.exists(IMAGE_PATH):
        logging.error(
            f"Error: La ruta de la imagen '{IMAGE_PATH}' no es válida. Por favor, especifica una imagen para analizar.")
    else:
        yolo_explainer = YoloV8Explainer(MODEL_PATH, YAML_PATH)
        yolo_explainer.explain(IMAGE_PATH, CLASS_TO_EXPLAIN)
