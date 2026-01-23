import os
from PIL import Image
import torch
from torchvision.transforms import functional as TF
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# === Rutas ===
model_dir = "segformer_cracks_v1"
input_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\train\imagesformasks"
output_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive\generated_masks"

# Crear carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# === Cargar modelo y procesador ===
processor = SegformerImageProcessor.from_pretrained(model_dir)
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval()

# === Inferencia por lote ===
# === Inferencia por lote ===
with torch.no_grad():
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            mask_img = (predicted_mask.astype(np.uint8)) * 255
            output_path = os.path.join(output_dir, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))
            Image.fromarray(mask_img).save(output_path)

            print(f"[✓] Guardada: {output_path}")
