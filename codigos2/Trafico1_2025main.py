import os

ruta_yaml = "D:/Python projectos 2025/TRAFICO IA EMANUEL/dataset_vehiculos/data.yaml"

comando = f"python yolov5/train.py --img 416 --batch 8 --epochs 50 --data \"{ruta_yaml}\" --weights yolov5s.pt --name prueba_trafico"

os.system(comando)
