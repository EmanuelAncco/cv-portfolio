import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import torch.nn as nn

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")

# Rutas de tus datos
data_dir = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco"
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')

# Carga de clases desde tu dataset
CLASSES = [
    'head', 'helmet', 'person', 'vest', 'gloves', 'boots',
    'glasses', 'mask', 'long_sleeves', 'long_pants',
    'safety_harness', 'ear_protection', 'face_shield',
    'respirator', 'shield', 'safety_vest', 'safety_gloves', 'protective_suit'
]
num_classes = len(CLASSES)

# Preprocesador de imágenes
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# Dataset personalizado
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.coco = COCO(ann_file)
        self.img_folder = os.path.join(img_folder, 'images')
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms if transforms else transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.img_folder, path)

        image = Image.open(img_path).convert("RGB")

        # Procesar las anotaciones
        boxes = []
        class_labels = []

        for ann in anns:
            bbox = ann['bbox']
            # COCO format: [x, y, width, height]
            # DETR espera: [x_min, y_min, x_max, y_max]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(ann['category_id'])

        encoding = processor(images=image, annotations={"bbox": boxes, "category_id": class_labels}, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

    def __len__(self):
        return len(self.ids)

# DataLoaders
train_dataset = CocoDataset(
    img_folder=train_dir,
    ann_file=os.path.join(train_dir, "_annotations.coco.limpiotrain.json"),
)

valid_dataset = CocoDataset(
    img_folder=valid_dir,
    ann_file=os.path.join(valid_dir, "_annotations.coco.limpiovalid.json"),
)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Modelo
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

# Reemplaza la cabeza de clasificación por una nueva para tus clases personalizadas
model.class_labels_classifier = nn.Linear(model.config.hidden_size, num_classes).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Función de entrenamiento
def train_one_epoch(epoch, model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        pixel_values, target = batch
        pixel_values = torch.stack(pixel_values).to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in target]

        outputs = model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} finalizado. Pérdida promedio: {avg_loss:.4f}")

    return avg_loss

# Entrenamiento
epochs = 10
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(epoch, model, train_loader, optimizer)

# Guardar modelo
model_save_path = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/detr_modelo_ppe.pth"
torch.save(model.state_dict(), model_save_path)
print(f"✅ Modelo guardado en: {model_save_path}")
