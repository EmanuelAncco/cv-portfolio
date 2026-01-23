from ultralytics import YOLO
import pprint

# --- ¡IMPORTANTE! ---
# Asegúrate de que esta ruta apunte a tu archivo best.pt del Modelo v3
MODEL_PATH = r"D:\Python_proyectos_2025\GAIATECH\runs\detect\emarc_vision_v3_run13\weights\best.pt"

print(f"Inspeccionando el modelo en: {MODEL_PATH}")

try:
    # Cargar el modelo
    model = YOLO(MODEL_PATH)

    # La propiedad 'names' es un diccionario que contiene el ID de la clase y su nombre.
    # Esta es la configuración exacta con la que se entrenó el modelo.
    print("\n--- Configuración de Clases Interna del Modelo ---")
    pprint.pprint(model.names)
    print(f"\nEl modelo fue entrenado con {len(model.names)} clases.")
    print("-------------------------------------------------")

except Exception as e:
    print(f"Error al cargar o inspeccionar el modelo: {e}")