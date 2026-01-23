import os
import json
import numpy as np
from PIL import Image
from labelme import utils

# Directorios
json_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\images"
output_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\masks_labelme"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(json_dir):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, filename)
    with open(json_path, "r") as f:
        data = json.load(f)

    # Crear mapeo: etiqueta "grieta" -> 1
    label_name_to_value = {"_background_": 0}
    for shape in data["shapes"]:
        label_name = shape["label"]
        if label_name not in label_name_to_value:
            label_name_to_value[label_name] = 1  # todas serán clase 1

    # Generar máscara
    lbl, _ = utils.shapes_to_label(
        (data["imageHeight"], data["imageWidth"]),
        data["shapes"],
        label_name_to_value
    )

    out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
    Image.fromarray(lbl.astype(np.uint8) * 255).save(out_path)  # Multiplica para visibilidad
    print(f"✅ Máscara generada: {out_path}")
