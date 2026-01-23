import os

# Usa la ruta ABSOLUTA a tu archivo data.yaml
ruta_yaml = "C:/Users/Emanuel/PyCharmMiscProject/Deteccion Vehiculos.v1-prueba-1-22-05-23.yolov5pytorch/data.yaml"

# Comando de entrenamiento
comando = f"python yolov5/train.py --img 416 --batch 8 --epochs 50 --data \"{ruta_yaml}\" --weights yolov5s.pt --name prueba_trafico"

# Ejecutar el entrenamiento
os.system(comando)
