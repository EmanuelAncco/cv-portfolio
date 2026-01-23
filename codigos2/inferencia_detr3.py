import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import numpy as np
import os
import csv
import json
from pathlib import Path
from tqdm import tqdm

# ============================
# Configuración del proyecto
# ============================

# Rutas
MODEL_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/modelo_detr_ppe.pth'
ANNOTATIONS_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco/train/_annotations.coco.limpiotrain.json'
IMAGES_FOLDER = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco/test/images/'
OUTPUT_DIR = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/resultados/'
VIDEO_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/test_video.mp4'  # Cambia si deseas

os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_EXPORT = True
CSV_PATH = os.path.join(OUTPUT_DIR, 'detecciones.csv')

SCORE_THRESHOLD = 0.5

# ============================
# Carga de clases desde JSON
# ============================
with open(ANNOTATIONS_PATH, 'r') as f:
    annotations = json.load(f)

categories = annotations['categories']
CLASSES = [cat['name'] for cat in categories]
id2label = {cat['id']: cat['name'] for cat in categories}
label2id = {v: k for k, v in id2label.items()}

print(f"✅ Clases cargadas desde JSON: {CLASSES}")

# ============================
# Configuración de dispositivo
# ============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Usando dispositivo: {device}")

# ============================
# Carga del modelo personalizado
# ============================

print("🔄 Cargando modelo DETR personalizado...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(CLASSES),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ Modelo cargado correctamente.")

# ============================
# Funciones auxiliares
# ============================

def draw_boxes(image, outputs, scores_csv, image_name):
    h, w, _ = image.shape
    for score, label, box in zip(outputs['scores'], outputs['labels'], outputs['boxes']):
        if score >= SCORE_THRESHOLD:
            box = box.cpu().detach().numpy()
            xmin, ymin, xmax, ymax = map(int, box)
            class_name = id2label[label.item()]
            confidence = score.item()

            color = (0, 255, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            text = f'{class_name}: {confidence:.2f}'
            cv2.putText(image, text, (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if CSV_EXPORT:
                scores_csv.append([image_name, class_name, confidence, xmin, ymin, xmax, ymax])

    return image

def run_inference(image_path, save_path, image_name, scores_csv, show=False):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoding = processor(images=image_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=SCORE_THRESHOLD)[0]

    image = draw_boxes(image, results, scores_csv, image_name)
    cv2.imwrite(save_path, image)

    if show:
        cv2.imshow('Detecciones', image)
        cv2.waitKey(1)

def export_csv(scores_csv):
    if CSV_EXPORT and scores_csv:
        with open(CSV_PATH, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Imagen', 'Clase', 'Confianza', 'xmin', 'ymin', 'xmax', 'ymax'])
            writer.writerows(scores_csv)
        print(f'✅ CSV exportado: {CSV_PATH}')

# ============================
# Inferencia para imagen única
# ============================

def infer_single_image():
    image_path = os.path.join(IMAGES_FOLDER, 'image_196_jpg.rf.bd368085d2771b68c6fdab9ddbb1e6dc.jpg')  # Cambia tu imagen aquí
    image_name = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, f'det_{image_name}')
    scores_csv = []

    run_inference(image_path, save_path, image_name, scores_csv, show=True)
    export_csv(scores_csv)

    cv2.destroyAllWindows()
    print("✅ Inferencia completada para imagen única.")

# ============================
# Inferencia para carpeta de imágenes
# ============================

def infer_folder():
    scores_csv = []
    image_paths = list(Path(IMAGES_FOLDER).glob('*.*'))

    for image_path in tqdm(image_paths, desc="Procesando imágenes"):
        image_name = os.path.basename(image_path)
        save_path = os.path.join(OUTPUT_DIR, f'det_{image_name}')
        run_inference(str(image_path), save_path, image_name, scores_csv, show=False)

    export_csv(scores_csv)
    print("✅ Inferencia completada para carpeta de imágenes.")

# ============================
# Inferencia para video
# ============================

def infer_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("🚫 Error al abrir el video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'det_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    scores_csv = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoding = processor(images=image_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoding)

        target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=SCORE_THRESHOLD)[0]

        frame = draw_boxes(frame, results, scores_csv, 'frame')

        out.write(frame)
        cv2.imshow('Detecciones', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    export_csv(scores_csv)
    print("✅ Inferencia completada para video.")

# ============================
# Ejecución principal
# ============================

if __name__ == '__main__':
    # Elige qué función deseas ejecutar:

    # Imagen única
    infer_single_image()

    # Carpeta completa
    # infer_folder()

    # Video
    # infer_video()
