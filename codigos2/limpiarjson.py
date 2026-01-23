import json
import os

# Carpetas de subsets
subsets = ['train', 'valid', 'test']
base_dir = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco'

for subset in subsets:
    json_path = os.path.join(base_dir, subset, '_annotations.coco.json')
    images_dir = os.path.join(base_dir, subset, 'images')

    print(f"\n✅ Procesando subset: {subset}")

    # Cargar JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Limpiar imágenes existentes
    valid_images = []
    valid_image_ids = set()

    for img in data['images']:
        # Eliminar prefijo 'images/' si existe
        img['file_name'] = img['file_name'].replace('images/', '').replace('\\', '/')

        img_path = os.path.join(images_dir, img['file_name'])
        if os.path.isfile(img_path):
            valid_images.append(img)
            valid_image_ids.add(img['id'])
        else:
            print(f"⚠️ Imagen faltante: {img_path}")

    # Limpiar anotaciones que no tengan imagen válida
    valid_annotations = [ann for ann in data['annotations'] if ann['image_id'] in valid_image_ids]

    # Reescribir JSON limpio
    clean_json_path = os.path.join(base_dir, subset, '_annotations.coco.limpio.json')
    cleaned_data = {
        'images': valid_images,
        'annotations': valid_annotations,
        'categories': data['categories']
    }

    with open(clean_json_path, 'w') as f:
        json.dump(cleaned_data, f)

    print(f"=== [{subset}] ===")
    print(f"Imágenes válidas: {len(valid_images)}")
    print(f"Anotaciones válidas: {len(valid_annotations)}")
    print(f"Archivo limpio guardado en: {clean_json_path}")

print("\n🎉 Proceso completado. Tu dataset está limpio y sin rutas duplicadas.")
