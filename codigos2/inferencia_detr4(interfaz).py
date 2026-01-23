import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import numpy as np
import os
import json
from tkinter import Tk, filedialog, Label, Button
from PIL import Image, ImageTk

# ============================
# Configuración inicial
# ============================
MODEL_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/modelo_detr_ppe_epoch_20.pth'
ANNOTATIONS_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco/train/_annotations.coco.limpiotrain.json'
OUTPUT_DIR = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/resultados/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colores para cada clase (definimos algunos para distinguir visualmente)
CLASS_COLORS = {}

# ============================
# Cargar clases desde el JSON
# ============================
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

CLASSES = [category['name'] for category in coco_data['categories']]
print(f"✅ Clases cargadas desde JSON: {CLASSES}")

# Asignamos colores únicos a cada clase
np.random.seed(42)
for class_name in CLASSES:
    CLASS_COLORS[class_name] = tuple(np.random.randint(0, 255, size=3).tolist())

# ============================
# Carga del modelo
# ============================
print("🔄 Cargando modelo DETR personalizado...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
id2label = {i: name for i, name in enumerate(CLASSES)}
label2id = {name: i for i, name in id2label.items()}

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(CLASSES),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("✅ Modelo cargado correctamente.")


# ============================
# Función de inferencia
# ============================

def infer_image(image_path):
    original_image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    encoding = processor(images=image_rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = torch.tensor([image_rgb.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.35)[0]

    # Dibujar resultados sobre la imagen
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        color = CLASS_COLORS.get(id2label[label.item()], (0, 255, 0))
        box = box.cpu().numpy().astype(int)
        class_name = id2label[label.item()]
        confidence = score.item()

        # Rectángulo
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Texto con fondo para mejor visibilidad
        text = f'{class_name}: {confidence:.2f}'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(original_image, (box[0], box[1] - text_height - baseline),
                      (box[0] + text_width, box[1]), color, -1)
        cv2.putText(original_image, text, (box[0], box[1] - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    output_path = os.path.join(OUTPUT_DIR, f'det_{os.path.basename(image_path)}')
    cv2.imwrite(output_path, original_image)
    print(f"✅ Resultado guardado en: {output_path}")

    return image_path, output_path


# ============================
# Interfaz Gráfica Tkinter
# ============================

def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path:
        original, detected = infer_image(file_path)
        show_images(original, detected)


def show_images(original_path, detected_path):
    original_image = Image.open(original_path)
    detected_image = Image.open(detected_path)

    original_image = original_image.resize((400, 400))
    detected_image = detected_image.resize((400, 400))

    original_photo = ImageTk.PhotoImage(original_image)
    detected_photo = ImageTk.PhotoImage(detected_image)

    original_label.config(image=original_photo)
    original_label.image = original_photo
    detected_label.config(image=detected_photo)
    detected_label.image = detected_photo


# Ventana principal
root = Tk()
root.title("Detección de Implementos de Seguridad - DETR")

# Botón para cargar imagen
Button(root, text="Seleccionar Imagen", command=open_file, font=('Arial', 12)).pack()

# Etiquetas de imagen
original_label = Label(root)
original_label.pack(side='left', padx=10, pady=10)

detected_label = Label(root)
detected_label.pack(side='right', padx=10, pady=10)

# Iniciar la interfaz
root.mainloop()
