import os
import shutil

# Rutas
images_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\images"
masks_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\masks"
output_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\images_matched"

os.makedirs(output_dir, exist_ok=True)

# Obtener los nombres base de las máscaras
mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]
mask_basenames = set(os.path.splitext(f)[0] for f in mask_files)

# Filtrar imágenes que coincidan
matched_count = 0
for image_file in os.listdir(images_dir):
    if image_file.endswith(".jpg"):
        name = os.path.splitext(image_file)[0]
        if name in mask_basenames:
            src_path = os.path.join(images_dir, image_file)
            dst_path = os.path.join(output_dir, image_file)
            shutil.copyfile(src_path, dst_path)
            matched_count += 1

print(f"✅ Imágenes copiadas: {matched_count}")
