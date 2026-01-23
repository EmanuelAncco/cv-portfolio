import os
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# === Parámetros de conversión (96 ppp => ~37.8 px/cm) ===
PIXELS_POR_CM = 37.8

# === Rutas ===
post_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\postprocessed_masks"  # máscaras limpias
originals_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\train\Positive"   # imágenes originales
save_path = r"D:\Python proyectos 2025\CNN EMANUEL\archive\resultados_grilla\grietas_shaded_fixed.png"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Escoger 3 máscaras "_cleaned_mask" al azar
cleaned_masks = [f for f in os.listdir(post_dir) if f.endswith("_cleaned_mask.png")]
sampled = random.sample(cleaned_masks, 3)

plt.figure(figsize=(12, 12))

for i, mask_file in enumerate(sampled):
    # Nombre base
    name = mask_file.replace("_cleaned_mask.png", "")

    # Rutas
    mask_path = os.path.join(post_dir, mask_file)
    img_path = os.path.join(originals_dir, name + ".jpg")

    # Cargar imagen y máscara
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Binarizar
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Encuentra contornos
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = image.copy()
    area_px_total = 0

    for cnt in contours:
        # 1) Pintar la región con color rojo semitransparente
        region_mask = np.zeros_like(overlay, dtype=np.uint8)
        cv2.drawContours(region_mask, [cnt], -1, (0, 0, 255), -1)
        alpha = 0.3
        overlay = cv2.addWeighted(overlay, 1.0, region_mask, alpha, 0)

        # 2) Bounding box rotado para longitud diagonal
        rot_rect = cv2.minAreaRect(cnt)
        box_points = cv2.boxPoints(rot_rect)  # Devuelve float
        box_points = box_points.astype(int)   # Convertir a int

        # Dibujar rectángulo rotado
        cv2.drawContours(overlay, [box_points], 0, (0, 255, 0), 2)

        # Ancho y alto en pixeles
        w_px, h_px = rot_rect[1]
        major_dim_px = max(w_px, h_px)
        major_dim_cm = major_dim_px / PIXELS_POR_CM

        # Área local
        area_px = cv2.contourArea(cnt)
        area_px_total += area_px

        # Texto en uno de los vértices
        ref_x, ref_y = box_points[0][0], box_points[0][1]
        label_text = f"{major_dim_cm:.1f}cm"
        cv2.putText(overlay, label_text, (ref_x, ref_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convertir área total a cm²
    area_cm2 = area_px_total / (PIXELS_POR_CM ** 2)

    # Texto de área total en la parte inferior
    area_text = f"Área total ~ {area_px_total:.1f} px² => {area_cm2:.2f} cm²"
    cv2.putText(overlay, area_text,
                (10, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar en grilla
    plt.subplot(3, 2, 2*i + 1)
    plt.title(f"{name} - Máscara")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 2, 2*i + 2)
    plt.title(f"{name} - Sombreado + Rect. Rotado")
    plt.imshow(overlay)
    plt.axis('off')

plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\n✅ Grilla guardada en: {save_path}")
plt.show()
