import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# === Rutas ===
post_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\postprocessed_masks"
originals_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\train\Positive"
save_path = r"D:\Python proyectos 2025\CNN EMANUEL\archive\resultados_grilla\grilla_con_metricas.png"

# Crear carpeta si no existe
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# === Selección aleatoria de nombres base ===
names = sorted(set(f.split('_')[0] for f in os.listdir(post_dir) if f.endswith('_overlay.png')))
sampled = random.sample(names, 5)

# === Visualización ===
plt.figure(figsize=(12, 10))

for i, name in enumerate(sampled):
    try:
        # Cargar imágenes
        img_path = os.path.join(originals_dir, f"{name}.jpg")
        mask_path = os.path.join(post_dir, f"{name}_cleaned_mask.png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarizar máscara
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Crear overlay rojo
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = binary_mask  # Solo canal rojo
        overlay = cv2.addWeighted(image, 1.0, red_mask, 0.5, 0)

        # === MÉTRICAS ===
        area = np.sum(binary_mask == 255)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"\n🔎 [Imagen: {name}]")
        print(f"   Área total segmentada: {area} px")

        for j, cnt in enumerate(contours):
            length = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"   Grieta {j + 1}: Largo ≈ {length:.2f} px, Ancho estimado ≈ {w} px")

        # === Mostrar en grilla ===
        plt.subplot(len(sampled), 3, 3 * i + 1)
        plt.imshow(image)
        plt.title(f"{name} - Original")
        plt.axis('off')

        plt.subplot(len(sampled), 3, 3 * i + 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Máscara limpia")
        plt.axis('off')

        plt.subplot(len(sampled), 3, 3 * i + 3)
        plt.imshow(overlay)
        plt.title("Superposición (Rojo)")
        plt.axis('off')

    except Exception as e:
        print(f"[ERROR] Falló en {name}: {e}")

plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"\n✅ Grilla con métricas guardada en: {save_path}")
plt.show()
