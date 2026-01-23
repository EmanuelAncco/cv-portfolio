import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === RUTAS ===
mask_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\generated_masks"
image_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\train\Positive"
output_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\postprocessed_masks"

os.makedirs(output_dir, exist_ok=True)

# === PARÁMETROS DE LIMPIEZA MORFOLÓGICA ===
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# === PROCESAMIENTO POR LOTES ===
for filename in os.listdir(mask_dir):
    if filename.endswith(".png"):
        mask_path = os.path.join(mask_dir, filename)
        name, _ = os.path.splitext(filename)

        # Ruta de imagen original asociada
        image_path = os.path.join(image_dir, name + ".jpg")
        if not os.path.exists(image_path):
            print(f"[ADVERTENCIA] Imagen no encontrada para: {filename}")
            continue

        # Cargar imagen y máscara
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarizar máscara
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Limpieza morfológica
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Contornos y métricas
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = np.sum(cleaned_mask == 255)
        print(f"[INFO] {filename} - Área total segmentada: {total_area} px")

        for i, cnt in enumerate(contours):
            length = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"    Grieta {i+1}: Largo = {length:.2f} px, Ancho estimado = {w} px")

        # Superposición
        overlay = cv2.addWeighted(image, 1.0, cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2RGB), 0.5, 0)

        # Guardar resultados
        cv2.imwrite(os.path.join(output_dir, f"{name}_cleaned_mask.png"), cleaned_mask)
        cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("\n✅ Postprocesamiento completado.")
