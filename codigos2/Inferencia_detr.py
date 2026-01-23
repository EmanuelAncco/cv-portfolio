import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/modelo_detr_ppe.pth"
model_name = "facebook/detr-resnet-50"

# Tus clases personalizadas (de acuerdo a tu dataset)
CLASSES = [
    "Casco", "Sin Casco", "Chaleco", "Sin Chaleco", "Guantes",
    "Sin Guantes", "Botas", "Sin Botas", "Protección Facial",
    "Sin Protección Facial", "Lentes", "Sin Lentes", "Arnés",
    "Sin Arnés", "Protección Auditiva", "Sin Protección Auditiva",
    "Ropa de Trabajo", "Sin Ropa de Trabajo"
]

# Carga del modelo
print("✅ Cargando modelo...")
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name, num_labels=len(CLASSES))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print("✅ Modelo cargado correctamente.")

# Función para dibujar resultados
def draw_boxes(image, outputs, threshold=0.5):
    keep = outputs["scores"] > threshold
    boxes = outputs["boxes"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        color = (0, 255, 0)
        x_min, y_min, x_max, y_max = box.astype(int)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        text = f"{CLASSES[label]}: {score:.2f}"
        cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Función de inferencia para imágenes
def predict_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    processed_outputs = processor.post_process_object_detection(outputs, target_sizes=[image.size[::-1]], threshold=0.5)[0]

    # Convertir PIL a OpenCV para dibujar
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_with_boxes = draw_boxes(image_cv, processed_outputs)

    cv2.imwrite(output_path, image_with_boxes)
    print(f"✅ Imagen procesada guardada en: {output_path}")

# Función de inferencia para videos
def predict_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Procesando video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocesamiento
            inputs = processor(images=frame, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            processed_outputs = processor.post_process_object_detection(outputs, target_sizes=[frame.shape[:2][::-1]], threshold=0.5)[0]

            # Dibujar cajas
            frame = draw_boxes(frame, processed_outputs)

            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"✅ Video procesado guardado en: {output_path}")

# -------------------------------------------
# 🔥 Pruebas
# Ruta de entrada y salida para pruebas
image_input = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/prueba.jpg"
image_output = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/resultado_prueba.jpg"

video_input = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/prueba_video.mp4"
video_output = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/resultado_video.mp4"

# Ejecutar pruebas
predict_image(image_input, image_output)
predict_video(video_input, video_output)
