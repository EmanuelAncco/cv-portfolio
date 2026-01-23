import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Ruta a carpeta de imágenes
image_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\images"
output_mask_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\masks_auto"
os.makedirs(output_mask_dir, exist_ok=True)

# Cargar modelo ViT entrenado
from ViTCRACKS1seg import create_fn  # asegúrate que esto esté bien con tu archivo

model = create_fn()
model.load_state_dict(torch.load("best_vit_model.pth"))
model.eval().cuda()

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def is_cracked(img_path):
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred == 1  # 1 = agrietado

def generate_mask(img_path, output_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1. Mejorar contraste sin exagerar
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # 2. Suavizado más balanceado
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 3. Bordes con umbrales más altos (menos ruido)
    edges = cv2.Canny(smoothed, 70, 150)

    # 4. Morfología más conservadora
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Eliminar manchas pequeñas (ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = 80  # elimina objetos pequeños (ajustable)

    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):  # saltamos fondo (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    cv2.imwrite(output_path, cleaned)



# Procesar imágenes
image_paths = list(Path(image_dir).glob("*.jpg"))
for img_path in image_paths:
    if is_cracked(str(img_path)):
        output_path = os.path.join(output_mask_dir, img_path.stem + ".png")
        generate_mask(str(img_path), output_path)
        print(f"✅ Máscara generada: {output_path}")
    else:
        print(f"❌ Imagen no agrietada: {img_path.name}")
