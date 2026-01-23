import os
import shutil
import random

# CONFIGURA TU RUTA BASE
base_dir = "D:/Python proyectos 2025/CNN EMANUEL"
train_pos_dir = os.path.join(base_dir, "archive", "train", "Positive")
train_neg_dir = os.path.join(base_dir, "archive", "train", "Negative")
val_pos_dir = os.path.join(base_dir, "archive", "val", "Positive")
val_neg_dir = os.path.join(base_dir, "archive", "val", "Negative")

# CARPETAS DESTINO
images_dir = os.path.join(base_dir, "data", "images")
masks_dir = os.path.join(base_dir, "data", "masks")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

# RECOGER TODAS LAS IMAGENES
train_imgs = [("train", "Positive", f) for f in os.listdir(train_pos_dir) if f.lower().endswith(('.jpg', '.png'))]
train_imgs += [("train", "Negative", f) for f in os.listdir(train_neg_dir) if f.lower().endswith(('.jpg', '.png'))]
val_imgs = [("val", "Positive", f) for f in os.listdir(val_pos_dir) if f.lower().endswith(('.jpg', '.png'))]
val_imgs += [("val", "Negative", f) for f in os.listdir(val_neg_dir) if f.lower().endswith(('.jpg', '.png'))]

# TOMAR UN SUBCONJUNTO PARA SEGMENTACION
random.seed(42)
selected_train = random.sample(train_imgs, 200)
selected_val = random.sample(val_imgs, 50)

def copy_images(image_list):
    for grupo, tipo, nombre in image_list:
        folder = train_pos_dir if (grupo == "train" and tipo == "Positive") else \
                 train_neg_dir if (grupo == "train" and tipo == "Negative") else \
                 val_pos_dir if (grupo == "val" and tipo == "Positive") else \
                 val_neg_dir
        src = os.path.join(folder, nombre)
        dst = os.path.join(images_dir, nombre)
        shutil.copy(src, dst)

copy_images(selected_train)
copy_images(selected_val)

# CREAR ARCHIVOS TXT DE ENTRENAMIENTO Y VALIDACION
def write_txt(file_path, items):
    with open(file_path, 'w') as f:
        for _, _, name in items:
            f.write(f"images/{name} masks/{name}\n")

data_dir = os.path.join(base_dir, "data")
write_txt(os.path.join(data_dir, "train.txt"), selected_train)
write_txt(os.path.join(data_dir, "val.txt"), selected_val)
open(os.path.join(data_dir, "test.txt"), 'w').close()

print("Carpetas organizadas y archivos .txt generados correctamente.")
