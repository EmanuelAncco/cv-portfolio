# export_model.py (v2 - Con calibración de datos)
from ultralytics import YOLO
import os

# --- CONFIGURACIÓN ---

# 1. Ruta a tu mejor modelo de EPP entrenado
model_path = 'runs/detect/emarc_vision_v3.1_finetune_manual_lr/weights/best.pt'

# 2. Ruta al archivo de configuración de tu dataset
#    Este es el archivo que se usará para la calibración INT8.
dataset_yaml_path = r'D:\Python_proyectos_2025\SEGURIDAD2.0\dataset_v3_rebalanced\data.yaml'

# --- VERIFICACIÓN ---
if not os.path.exists(model_path):
    print(f"ERROR: No se encontró el modelo en: {model_path}")
    exit()
if not os.path.exists(dataset_yaml_path):
    print(f"ERROR: No se encontró el archivo data.yaml en: {dataset_yaml_path}")
    exit()

# --- PROCESO DE EXPORTACIÓN ---
try:
    # Cargar tu modelo
    model = YOLO(model_path)

    # Exportar a formato TensorFlow Lite con cuantización INT8
    # y calibración usando tu dataset específico.
    model.export(
        format='onnx',
        # int8=True,              # <-- Comenta esta línea
        imgsz=320,
        # data=dataset_yaml_path  # <-- Y también esta
    )

    print("\n" + "="*50)
    print("¡Éxito! Modelo exportado a formato .tflite con cuantización INT8.")
    print("El archivo se encuentra en la misma carpeta que tu modelo original:")
    print(os.path.dirname(model_path))
    print("="*50)

except Exception as e:
    print(f"\nOcurrió un error durante la exportación: {e}")
    print("Asegúrate de haber instalado 'tensorflow' y de haber corregido las versiones de las librerías (numpy, protobuf).")