import os
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import matplotlib.pyplot as plt

# --- Rutas ---
model_dir = "segformer_cracks_v1"
image_path = r"D:\python_proyectos_2025\PyCharmMiscProject\imagenes_test\2.jpg"

# --- Cargar modelo ---
processor = SegformerImageProcessor.from_pretrained(model_dir)
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval()

# --- Leer imagen ---
image_pil = Image.open(image_path).convert("RGB")
image_np = np.array(image_pil)
inputs = processor(images=image_pil, return_tensors="pt")

# --- Inferencia ---
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False)
    predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

# --- Postprocesamiento ---
binary_mask = (predicted_mask == 1).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# --- Métricas ---
resolucion_cm = 2.54 / 96  # 96 ppi
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

area_px = np.sum(cleaned_mask == 255)
area_cm2 = (area_px * (resolucion_cm ** 2))

ancho_promedio = 0
mayor_largo = 0

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    (w_px, h_px) = rect[1]
    if w_px * h_px == 0:
        continue
    largo_px = max(w_px, h_px)
    ancho_px = min(w_px, h_px)

    largo_cm = largo_px * resolucion_cm
    ancho_cm = ancho_px * resolucion_cm

    if largo_cm > mayor_largo:
        mayor_largo = largo_cm
    ancho_promedio += ancho_cm

# --- Promedios y daño estimado ---
ancho_promedio /= max(len(contours), 1)
daño_porcentual = (area_px / (image_np.shape[0] * image_np.shape[1])) * 100

# --- Dibujar resultado ---
overlay = image_np.copy()
overlay[cleaned_mask == 255] = [0, 0, 255]  # azul

cv2.putText(overlay, f"Area: {area_cm2:.2f} cm2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(overlay, f"Largo max: {mayor_largo:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(overlay, f"Ancho prom: {ancho_promedio:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(overlay, f"Dano: {daño_porcentual:.1f} %", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# --- Mostrar resultado ---
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Resultado segmentado")
plt.axis('off')
plt.show()
