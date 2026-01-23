import os
import json

# Lista de carpetas a limpiar
carpetas = [
    r"D:\Python proyectos 2025\SEGURIDAD_CONSTRUCCION\PPE-0.3.v1i.coco\train",
    r"D:\Python proyectos 2025\SEGURIDAD_CONSTRUCCION\PPE-0.3.v1i.coco\valid",
    r"D:\Python proyectos 2025\SEGURIDAD_CONSTRUCCION\PPE-0.3.v1i.coco\test"
]

def limpiar_json(carpeta_imagenes):
    print(f"\n🚀 Procesando carpeta: {carpeta_imagenes}")

    json_path = os.path.join(carpeta_imagenes, "_annotations.coco.json")
    json_output = os.path.join(carpeta_imagenes, "_annotations.coco.limpio.json")

    if not os.path.exists(json_path):
        print(f"⚠️ No se encontró el JSON en: {json_path}")
        return

    # Cargar JSON original
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Obtener imágenes existentes en la carpeta
    imagenes_existentes = set(os.listdir(carpeta_imagenes))
    print(f"Imágenes encontradas: {len(imagenes_existentes)}")

    # Filtrar imágenes válidas
    imagenes_validas = []
    ids_validos = set()

    for imagen in data['images']:
        nombre_imagen = imagen['file_name']
        if nombre_imagen in imagenes_existentes:
            imagenes_validas.append(imagen)
            ids_validos.add(imagen['id'])
        else:
            print(f"⚠️ Imagen faltante: {nombre_imagen}")

    print(f"Imágenes válidas: {len(imagenes_validas)}")

    # Filtrar anotaciones que correspondan a las imágenes válidas
    anotaciones_validas = [ann for ann in data['annotations'] if ann['image_id'] in ids_validos]
    print(f"Anotaciones válidas: {len(anotaciones_validas)}")

    # Crear nuevo JSON limpio
    data_limpio = {
        "images": imagenes_validas,
        "annotations": anotaciones_validas,
        "categories": data['categories']
    }

    # Guardar nuevo JSON limpio
    with open(json_output, 'w') as f:
        json.dump(data_limpio, f, indent=4)

    print(f"✅ Dataset limpio guardado en: {json_output}")

# Ejecutar limpieza para todas las carpetas
for carpeta in carpetas:
    limpiar_json(carpeta)

print("\n🎉 Limpieza completada para todas las carpetas.")
