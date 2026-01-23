# -*- coding: utf-8 -*-
"""
Script para entrenar un modelo de detección de objetos (DETR) y generar gráficos para investigación.

Este script realiza las siguientes acciones:
1.  Divide el dataset original en conjuntos de entrenamiento y validación.
2.  Define un Dataset de PyTorch personalizado para cargar imágenes y etiquetas en formato YOLO.
3.  Utiliza un modelo DETR (DEtection TRansformer) pre-entrenado de la biblioteca Hugging Face.
4.  Entrena y evalúa el modelo durante un número específico de épocas.
5.  Genera y guarda gráficos de la curva de pérdida y ejemplos de predicciones.

Dependencias: pip install torch torchvision torchaudio transformers scikit-learn pillow matplotlib seaborn timm
"""
import os
import shutil
import random
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Cambia esta ruta a la carpeta raíz de tu dataset.
# Debería contener la carpeta 'train' con 'images' y 'labels'.
DATASET_BASE_PATH = r"D:\Python_proyectos_2025\AVES\HamaBurung.v84i.yolov11"
TRAIN_DIR = os.path.join(DATASET_BASE_PATH, "train")
VALID_DIR = os.path.join(DATASET_BASE_PATH, "valid")
TEST_SPLIT_SIZE = 0.2
NUM_EPOCHS = 12
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_PREDICTION_EXAMPLES = 5  # Número de imágenes de ejemplo para visualizar

# Ignorar advertencias
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --- FUNCIONES DEL DATASET ---

def create_validation_set(base_path, train_dir, valid_dir, test_size=0.2):
    """Crea un conjunto de validación si no existe."""
    if os.path.exists(valid_dir):
        print(f"El directorio de validación '{valid_dir}' ya existe. Saltando la creación.")
        return
    print("Creando el conjunto de datos de validación...")
    os.makedirs(os.path.join(valid_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, "labels"), exist_ok=True)
    train_images_path = os.path.join(train_dir, "images")
    image_files = [f for f in os.listdir(train_images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError(f"No se encontraron imágenes en {train_images_path}")
    train_files, valid_files = train_test_split(image_files, test_size=test_size, random_state=42)
    for filename in valid_files:
        shutil.move(os.path.join(train_dir, "images", filename), os.path.join(valid_dir, "images", filename))
        label_filename = os.path.splitext(filename)[0] + ".txt"
        shutil.move(os.path.join(train_dir, "labels", label_filename),
                    os.path.join(valid_dir, "labels", label_filename))
    print(f"Dataset listo: {len(train_files)} para entrenamiento, {len(valid_files)} para validación.")


def yolo_to_coco(bbox, width, height):
    """Convierte bbox de YOLO a COCO."""
    x_center, y_center, w, h = bbox
    x_min = (x_center - w / 2) * width
    y_min = (y_center - h / 2) * height
    x_max = (x_center + w / 2) * width
    y_max = (y_center + h / 2) * height
    return [x_min, y_min, x_max, y_max]


class BirdDetectionDataset(Dataset):
    """Dataset de PyTorch para las imágenes y etiquetas."""

    def __init__(self, image_dir, label_dir, image_processor):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_processor = image_processor
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    bbox_yolo = list(map(float, parts[1:]))
                    bbox_coco = yolo_to_coco(bbox_yolo, width, height)
                    annotations.append({'category_id': class_id, 'bbox': bbox_coco,
                                        'area': (bbox_coco[2] - bbox_coco[0]) * (bbox_coco[3] - bbox_coco[1]),
                                        'iscrowd': 0})
        target = {'image_id': idx, 'annotations': annotations}
        encoding = self.image_processor(images=image, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    """Función 'collate' para el padding."""
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    return {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": labels}


# --- FUNCIONES DE VISUALIZACIÓN ---

def plot_training_history(train_losses, valid_losses, output_dir):
    """
    Genera y guarda un gráfico de las pérdidas de entrenamiento y validación.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Pérdida de Entrenamiento', marker='o')
    plt.plot(valid_losses, label='Pérdida de Validación', marker='o')
    plt.title('Historial de Entrenamiento: Pérdida vs. Épocas')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (Loss)')
    plt.legend()
    plt.xticks(range(len(train_losses)))
    plt.grid(True)
    save_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de historial de entrenamiento guardado en: {save_path}")


def plot_predictions(model, dataset, image_processor, output_dir, num_images=5):
    """
    Genera y guarda imágenes con las predicciones del modelo y las etiquetas reales.
    """
    print(f"Generando {num_images} ejemplos de predicción...")
    model.eval()
    device = model.device

    # Asegurarse que el directorio de salida para las predicciones exista
    predictions_path = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_path, exist_ok=True)

    # Seleccionar imágenes aleatorias del dataset
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))

    for i, idx in enumerate(indices):
        # Obtener una imagen y sus datos
        item = dataset[idx]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)

        # Cargar la imagen original para dibujar sobre ella
        original_image_path = os.path.join(dataset.image_dir, dataset.image_files[idx])
        image = Image.open(original_image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        # Dibujar Ground Truth (etiquetas reales)
        for ann in dataset[idx]['labels']:
            box = ann['boxes'].tolist()
            draw.rectangle(box, outline="green", width=3)

        # Realizar predicción
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)

        # Procesar resultados
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

        # Dibujar predicciones
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
            # Podrías añadir texto si tienes un mapa de etiquetas
            # label_text = f"Pred: {label.item()} ({score:.2f})"
            # draw.text((box[0], box[1]), label_text, fill="red")

        # Guardar imagen
        save_path = os.path.join(predictions_path, f"prediction_example_{i + 1}.png")
        image.save(save_path)

    print(f"Ejemplos de predicciones guardados en: {predictions_path}")


# --- SCRIPT PRINCIPAL ---
if __name__ == '__main__':
    # Paso 1: Preparar el dataset
    create_validation_set(DATASET_BASE_PATH, TRAIN_DIR, VALID_DIR, test_size=TEST_SPLIT_SIZE)

    # Paso 2: Inicializar el procesador de imágenes
    model_checkpoint = "facebook/detr-resnet-50"
    image_processor = DetrImageProcessor.from_pretrained(model_checkpoint)

    # Paso 3: Crear Datasets y DataLoaders
    train_dataset = BirdDetectionDataset(image_dir=os.path.join(TRAIN_DIR, "images"),
                                         label_dir=os.path.join(TRAIN_DIR, "labels"), image_processor=image_processor)
    valid_dataset = BirdDetectionDataset(image_dir=os.path.join(VALID_DIR, "images"),
                                         label_dir=os.path.join(VALID_DIR, "labels"), image_processor=image_processor)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)
    print(
        f"Dataset de entrenamiento: {len(train_dataset)} muestras. | Dataset de validación: {len(valid_dataset)} muestras.")

    # Paso 4: Cargar el modelo
    # Asumiendo 1 clase "ave" (id 0) + 1 clase "no-objeto" (id 1). Ajusta num_labels si tienes más clases.
    model = DetrForObjectDetection.from_pretrained(model_checkpoint, num_labels=2, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Modelo cargado en el dispositivo: {device}")

    # Paso 5: Bucle de entrenamiento
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = {'train_loss': [], 'valid_loss': []}

    print("\n--- INICIANDO ENTRENAMIENTO ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_total = 0
        for i, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            if (i + 1) % 20 == 0:
                print(
                    f"  Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss_total / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        valid_loss_total = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                valid_loss_total += outputs.loss.item()

        avg_valid_loss = valid_loss_total / len(valid_dataloader)
        history['valid_loss'].append(avg_valid_loss)

        print(f"\n--- Fin de Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        print(f"Pérdida de Entrenamiento (Avg): {avg_train_loss:.4f}")
        print(f"Pérdida de Validación (Avg): {avg_valid_loss:.4f}\n")

    # Paso 6: Guardar modelo y generar gráficos
    output_dir = os.path.join(DATASET_BASE_PATH, "detr-bird-detector-final")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    image_processor.save_pretrained(output_dir)
    print(f"\nEntrenamiento completado. Modelo guardado en: {output_dir}")

    # Generar y guardar los gráficos
    plot_training_history(history['train_loss'], history['valid_loss'], output_dir)
    plot_predictions(model, valid_dataset, image_processor, output_dir, num_images=NUM_PREDICTION_EXAMPLES)
    print("\n¡Proceso finalizado con éxito!")
