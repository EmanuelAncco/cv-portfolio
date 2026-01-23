import json
import os

json_path = r"D:\Python proyectos 2025\SEGURIDAD_CONSTRUCCION\PPE-0.3.v1i.coco\train\_annotations.coco.json"

with open(json_path, 'r') as f:
    data = json.load(f)

category_ids = [ann['category_id'] for ann in data['annotations']]
print(f"Máximo category_id: {max(category_ids)}")
print(f"Número total de categorías únicas: {len(set(category_ids))}")
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
