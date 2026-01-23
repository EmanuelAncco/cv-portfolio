import os
import shutil
import random

# Ruta base del dataset original
base_dir = r"D:\Python projectos 2025\CNN EMANUEL\archive"
original_dirs = ['Negative', 'Positive']
val_split = 0.2

# Crear carpetas de train y val
for split in ['train', 'val']:
    for category in original_dirs:
        split_dir = os.path.join(base_dir, split, category)
        os.makedirs(split_dir, exist_ok=True)

# Inicializar listas para los archivos txt
train_list = []
val_list = []

# Separar y mover imágenes
for category in original_dirs:
    src_folder = os.path.join(base_dir, category)
    images = os.listdir(src_folder)
    random.shuffle(images)

    split_index = int(len(images) * (1 - val_split))
    train_images = images[:split_index]
    val_images = images[split_index:]

    for img in train_images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(base_dir, 'train', category, img)
        shutil.move(src, dst)
        train_list.append(f"train/{category}/{img}")

    for img in val_images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(base_dir, 'val', category, img)
        shutil.move(src, dst)
        val_list.append(f"val/{category}/{img}")

# Eliminar carpetas originales vacías
for category in original_dirs:
    os.rmdir(os.path.join(base_dir, category))

# Guardar archivos de texto con rutas
with open(os.path.join(base_dir, 'train_images.txt'), 'w') as f:
    for item in train_list:
        f.write(f"{item}\n")

with open(os.path.join(base_dir, 'val_images.txt'), 'w') as f:
    for item in val_list:
        f.write(f"{item}\n")

print("✅ Dataset dividido y archivos .txt generados correctamente.")
