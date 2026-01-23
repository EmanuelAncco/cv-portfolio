import cv2
import os
import logging
import time
import numpy as np
import yaml
import pickle

# --- Dependencias Clave ---
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
from deepface import DeepFace

# --- CONFIGURACIÓN GENERAL ---
TFLITE_MODEL_PATH = "emairc.tflite"
EMBEDDINGS_DB_PATH = "face_database.pkl"
YAML_PATH = "data.yaml"

# --- CONFIGURACIÓN DE LÓGICA DE NEGOCIO ---
MANDATORY_EPP = {"Helmet"}
FACE_DISTANCE_THRESHOLD = 0.6
EPP_CONF_THRESHOLD = 0.05

# --- CONFIGURACIÓN DEL LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class PrototypeController:
    """
    Controla toda la lógica de la aplicación, incluyendo la carga de modelos,
    la verificación facial y la auditoría de EPP.
    """

    def __init__(self):
        self.MANDATORY_EPP = MANDATORY_EPP
        logging.info(f"Cargando modelo TFLite desde: {TFLITE_MODEL_PATH}")
        self.interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.img_height = self.input_details[0]['shape'][1]
        self.img_width = self.input_details[0]['shape'][2]

        self.class_names, self.class_colors = self.load_class_info_from_yaml()
        if not self.class_names: raise RuntimeError("No se pudieron cargar las clases.")

        logging.info(f"Cargando base de datos de embeddings desde: {EMBEDDINGS_DB_PATH}")
        try:
            with open(EMBEDDINGS_DB_PATH, 'rb') as f:
                self.face_database = pickle.load(f)
            logging.info(f"¡Base de datos con {len(self.face_database)} trabajadores cargada!")
        except FileNotFoundError:
            logging.error(f"Error crítico: No se encontró el archivo '{EMBEDDINGS_DB_PATH}'.")
            raise

    def load_class_info_from_yaml(self):
        if not os.path.exists(YAML_PATH):
            logging.error(f"No se encontró el archivo YAML en: {YAML_PATH}")
            return None, None
        with open(YAML_PATH, 'r') as f: data = yaml.safe_load(f)
        class_names = data['names']
        custom_palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                          (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
        colors = {name: custom_palette[i % len(custom_palette)] for i, name in enumerate(class_names)}
        return class_names, colors

    def verify_person(self, frame):
        """
        Orquesta el proceso completo de verificación bajo demanda (tecla 'v').
        """
        temp_image_path = "temp_capture.jpg"
        cv2.imwrite(temp_image_path, frame)
        raw_detections = []

        logging.info("Iniciando verificación facial optimizada...")
        try:
            embedding_objs = DeepFace.represent(img_path=temp_image_path, model_name='ArcFace', enforce_detection=False)
            if not embedding_objs or embedding_objs[0]["face_confidence"] < 0.9:
                logging.warning("No se detectó un rostro claro en la imagen.")
                return "ERROR: No se detectó rostro", None, frame, []

            target_embedding = embedding_objs[0]["embedding"]
            best_match_name, lowest_distance = self.find_best_match(target_embedding)

            if lowest_distance > FACE_DISTANCE_THRESHOLD:
                logging.warning(
                    f"Coincidencia débil. Más cercano: {best_match_name} (Distancia: {lowest_distance:.2f})")
                return "ACCESO DENEGADO (Rostro no reconocido)", None, frame, []

            worker_name = best_match_name
            logging.info(f"Identidad confirmada: {worker_name} (Distancia: {lowest_distance:.2f})")

            logging.info("Iniciando auditoría de EPP...")
            annotated_frame, detected_epps, raw_detections = self.detect_epp(frame)
            logging.info(f"EPPs detectados (post-filtro): {detected_epps}")

            missing_epp = self.MANDATORY_EPP - detected_epps
            if not missing_epp:
                logging.info("Auditoría de EPP exitosa.")
                return "ACCESO PERMITIDO", worker_name, annotated_frame, raw_detections
            else:
                missing_str = ", ".join(missing_epp)
                logging.warning(f"Resultado: Falta EPP - {missing_str}")
                return f"ACCESO DENEGADO (Falta {missing_str})", worker_name, annotated_frame, raw_detections

        except Exception as e:
            logging.error(f"Error inesperado durante la verificación: {e}", exc_info=True)
            return "ERROR DE SISTEMA", None, frame, []
        finally:
            if os.path.exists(temp_image_path): os.remove(temp_image_path)

    def find_best_match(self, target_embedding):
        """Encuentra el rostro más cercano en la base de datos."""
        best_match_name = None
        lowest_distance = float('inf')
        for worker_name, db_embedding in self.face_database:
            distance = 1 - np.dot(target_embedding, db_embedding) / (
                        np.linalg.norm(target_embedding) * np.linalg.norm(db_embedding))
            if distance < lowest_distance:
                lowest_distance = distance
                best_match_name = worker_name
        return best_match_name, lowest_distance

    def detect_epp(self, frame):
        """
        Ejecuta la inferencia del modelo TFLite y devuelve las detecciones.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.img_width, self.img_height))
        input_data = np.expand_dims(image_resized, axis=0)

        if self.input_details[0]['dtype'] != np.uint8:
            input_data = (input_data / 255.0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0].T

        raw_detections = self.process_raw_detections(output_data, frame.shape)

        annotated_frame, detected_classes = self.draw_final_detections(frame, raw_detections)

        return annotated_frame, detected_classes, raw_detections

    def process_raw_detections(self, output_data, original_shape):
        """Procesa la salida cruda del modelo a una lista de diccionarios."""
        raw_detections = []
        original_h, original_w, _ = original_shape
        scale_w, scale_h = original_w / self.img_width, original_h / self.img_height

        for detection in output_data:
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            max_score = class_scores[class_id]
            cx, cy, w, h = detection[:4]
            x = int((cx - w / 2) * scale_w)
            y = int((cy - h / 2) * scale_h)
            w = int(w * scale_w)
            h = int(h * scale_h)
            raw_detections.append({"class_name": self.class_names[class_id], "score": max_score, "box": [x, y, w, h]})
        return raw_detections

    def draw_final_detections(self, frame, raw_detections):
        """Dibuja las detecciones que superan el umbral y NMS."""
        boxes, scores, class_ids = [], [], []
        for det in raw_detections:
            if det['score'] > EPP_CONF_THRESHOLD:
                scores.append(det['score'])
                class_ids.append(self.class_names.index(det['class_name']))
                boxes.append(det['box'])

        indices = cv2.dnn.NMSBoxes(boxes, scores, EPP_CONF_THRESHOLD, 0.5)
        detected_classes = set()
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.class_names[class_ids[i]]
                score = scores[i]
                detected_classes.add(class_name)
                color = self.class_colors.get(class_name, (255, 0, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{class_name}: {score:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame, detected_classes


def main():
    """
    Función principal que ejecuta el bucle de la cámara y la lógica de la interfaz.
    """
    logging.info("Inicializando sistema...")
    try:
        controller = PrototypeController()
    except Exception as e:
        logging.critical(f"Error Crítico al inicializar: {e}", exc_info=True)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("Error: No se puede abrir la cámara.")
        return

    live_analysis_mode = False

    print("\n" + "=" * 50)
    print("--- Prototipo de Control de Acceso EMARC v2.5 (Visibilidad Mejorada) ---")
    print("--- Presiona 'v' para una verificación de acceso COMPLETA ---")
    print("--- Presiona 'd' para activar/desactivar el ANÁLISIS EN VIVO ---")
    print("--- Presiona 'q' para salir ---")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- INICIO: CAMBIO v2.5 - Lógica del Modo de Análisis en Vivo con Visibilidad Mejorada ---
        if live_analysis_mode:
            # Solo ejecuta la detección de EPP en tiempo real.
            _, _, raw_detections = controller.detect_epp(frame.copy())

            # Dibuja todas las detecciones crudas directamente en el frame principal.
            for det in sorted(raw_detections, key=lambda x: x['score'], reverse=True)[:10]:
                x, y, w, h = det['box']
                score = det['score']
                name = det['class_name']

                # Dibuja la caja de detección en amarillo para indicar que es de depuración.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

                # Crea una etiqueta con fondo para el texto para mejorar la legibilidad.
                debug_text = f"{name} ({score:.2f})"
                (text_width, text_height), baseline = cv2.getTextSize(debug_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Dibuja el rectángulo de fondo.
                cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y - 5), (0, 255, 255), -1)

                # Dibuja el texto en negro para máximo contraste.
                cv2.putText(frame, debug_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # --- FIN: CAMBIO v2.5 ---

        info_text = "Analisis en Vivo: ACTIVADO" if live_analysis_mode else "Presiona 'v' para verificar"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Presiona 'q' para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow('EMARC VISIÓN - Prototipo v2.5', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('d'):
            live_analysis_mode = not live_analysis_mode
            status = "ACTIVADO" if live_analysis_mode else "DESACTIVADO"
            print(f"** ANÁLISIS EN VIVO {status} **")
        elif key == ord('v'):
            print("\n" + "-" * 20)
            logging.info("Iniciando verificación de acceso completa...")
            start_time = time.time()
            result_text, worker_name, result_frame, _ = controller.verify_person(frame.copy())
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Verificación completada en {processing_time:.2f} segundos.")
            print(f"--> Resultado Final: {result_text}")

            color = (0, 255, 0) if "PERMITIDO" in result_text else (0, 0, 255)
            cv2.putText(result_frame, result_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(result_frame, result_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            if worker_name:
                cv2.putText(result_frame, worker_name, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(result_frame, worker_name, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.imshow('Resultado de Verificación', result_frame)

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Recursos liberados. Programa finalizado.")


if __name__ == "__main__":
    main()
