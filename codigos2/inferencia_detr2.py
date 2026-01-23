import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import os
import time
import pandas as pd
from tqdm import tqdm

# ============================
# CONFIGURACIÓN
# ============================

# Rutas importantes
MODEL_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/modelo_detr_ppe.pth'
INPUT_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco/test/images/'  # Cambia por imagen, carpeta o video
OUTPUT_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/resultados/'
CSV_EXPORT = True
VISUALIZE = True
CONFIDENCE_THRESHOLD = 0.5

# Clases personalizadas
CLASSES = [
    'Sin casco', 'Casco', 'Chaleco', 'Guantes', 'Protección auditiva',
    'Lentes', 'Botas', 'Mascarilla', 'Arnés', 'Señalización',
    'Camisa reflectante', 'Pantalones reflectantes', 'Protección facial',
    'Rodilleras', 'Cinta de seguridad', 'Monja', 'Petos', 'Otros'
]

# Preparar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Usando dispositivo: {device}')

# Crear carpeta de resultados
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================
# CARGAR MODELO PERSONALIZADO
# ============================
print('🔄 Cargando modelo DETR personalizado...')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained(
    'facebook/detr-resnet-50',
    num_labels=len(CLASSES),
    id2label={i: c for i, c in enumerate(CLASSES)},
    label2id={c: i for i, c in enumerate(CLASSES)},
    ignore_mismatched_sizes=True
)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device).eval()
print('✅ Modelo cargado correctamente.')

# ============================
# FUNCIONES DE INFERENCIA
# ============================

def detect_image(image_path, save_path, writer=None):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    encoding = processor(images=image_rgb, return_tensors='pt').to(device)
    start = time.time()
    outputs = model(**encoding)
    end = time.time()

    target_sizes = torch.tensor([image_rgb.shape[:2]])
    results = processor.post_process_object_detection(outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes)[0]

    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        score = score.item()
        label_text = CLASSES[label.item()]
        box = [round(i, 2) for i in box.tolist()]

        # Dibujar la caja
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, f'{label_text}: {score:.2f}', (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Guardar en CSV si se requiere
        if writer is not None:
            writer.writerow([os.path.basename(image_path), label_text, score, *box, end - start])

    # Guardar resultado
    cv2.imwrite(save_path, image)

    # Visualizar
    if VISUALIZE:
        cv2.imshow('Detección', image)
        cv2.waitKey(1)

    print(f'🖼️ Imagen procesada: {os.path.basename(image_path)} - Tiempo: {end - start:.2f} s')


def detect_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoding = processor(images=image_rgb, return_tensors='pt').to(device)
        start = time.time()
        outputs = model(**encoding)
        end = time.time()

        target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=CONFIDENCE_THRESHOLD, target_sizes=target_sizes)[0]

        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            score = score.item()
            label_text = CLASSES[label.item()]
            box = [round(i, 2) for i in box.tolist()]

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{label_text}: {score:.2f}', (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

        if VISUALIZE:
            cv2.imshow('Video detección', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'🎥 Video procesado: {os.path.basename(video_path)} - Total de frames: {frame_count}')

# ============================
# EJECUCIÓN PRINCIPAL
# ============================

if __name__ == '__main__':
    # Exportar a CSV
    csv_file = None
    if CSV_EXPORT:
        csv_file = open(os.path.join(OUTPUT_PATH, 'detecciones.csv'), mode='w', newline='')
        import csv
        writer = csv.writer(csv_file)
        writer.writerow(['Archivo', 'Clase', 'Confianza', 'X_min', 'Y_min', 'X_max', 'Y_max', 'Tiempo'])
    else:
        writer = None

    # Detectar imagen individual
    if os.path.isfile(INPUT_PATH):
        save_path = os.path.join(OUTPUT_PATH, os.path.basename(INPUT_PATH))
        detect_image(INPUT_PATH, save_path, writer)

    # Detectar en carpeta de imágenes
    elif os.path.isdir(INPUT_PATH):
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in os.listdir(INPUT_PATH) if os.path.splitext(f)[-1].lower() in image_extensions]

        for img_file in tqdm(image_files, desc='Procesando imágenes'):
            img_path = os.path.join(INPUT_PATH, img_file)
            save_path = os.path.join(OUTPUT_PATH, img_file)
            detect_image(img_path, save_path, writer)

    # Detectar en video
    elif INPUT_PATH.lower().endswith(('.mp4', '.avi', '.mov')):
        save_path = os.path.join(OUTPUT_PATH, os.path.basename(INPUT_PATH))
        detect_video(INPUT_PATH, save_path)

    else:
        print('❌ Ruta de entrada no válida. Verifica la ruta de INPUT_PATH.')

    if CSV_EXPORT and csv_file:
        csv_file.close()
        print(f'📄 Detecciones guardadas en: {os.path.join(OUTPUT_PATH, "detecciones.csv")}')

    print('✅ Inferencia terminada.')
