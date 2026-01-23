import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import numpy as np

# Configuraciones
model_load_path = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/modelo_vit_ppe.pth"
image_path = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco/test/images/2_jpg.rf.fb92643b92bddf88796b89d6403cf5f7.jpg"  # Cambia por tu imagen de prueba
num_classes = 17
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Cargar modelo
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes
)
model.load_state_dict(torch.load(model_load_path, map_location=device))
model.to(device)
model.eval()

# 2. Preprocesamiento de la imagen
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast=True)

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 3. Inferencia
with torch.no_grad():
    outputs = model(**inputs).logits
    probs = torch.sigmoid(outputs).cpu().numpy()[0]

# 4. Interpretación
threshold = 0.5  # Puedes ajustar esto según tu dataset
predicted_labels = np.where(probs > threshold)[0]
print("✅ Predicciones terminadas.")
print(f"Probabilidades: {probs}")
print(f"Clases predichas (IDs): {predicted_labels}")

# Opcional: si tienes el mapeo de clases, lo mostramos
class_names = [
    "Sin Casco", "Casco", "Chaleco", "Sin Chaleco", "Guantes", "Sin Guantes",
    # Continúa con tus clases...
]

for idx in predicted_labels:
    print(f"➡️  {class_names[idx] if idx < len(class_names) else f'Clase {idx}'} (confianza: {probs[idx]:.2f})")
