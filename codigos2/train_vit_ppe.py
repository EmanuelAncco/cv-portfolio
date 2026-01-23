import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

# === Configuración de directorios y dispositivo ===
data_dir = "D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/PPE-0.3.v1i.coco"

train_images_dir = os.path.join(data_dir, 'train', 'images')
val_images_dir = os.path.join(data_dir, 'valid', 'images')

train_ann_file = os.path.join(data_dir, 'train', '_annotations.coco.limpio.json')
val_ann_file = os.path.join(data_dir, 'valid', '_annotations.coco.limpio.json')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {device}")

# === Hiperparámetros ===
num_classes = 17  # Número de clases en tu dataset
batch_size = 32
learning_rate = 2e-5
epochs = 5

# === Preprocesamiento de imágenes ===
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast=True)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

# === Dataset personalizado ===
class CustomCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transform=None):
        super().__init__(img_folder, ann_file)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)

        labels = torch.zeros(num_classes)
        for annotation in target:
            labels[annotation['category_id']] = 1

        return img, labels

train_dataset = CustomCocoDataset(
    img_folder=train_images_dir,
    ann_file=train_ann_file,
    transform=transform
)
val_dataset = CustomCocoDataset(
    img_folder=val_images_dir,
    ann_file=val_ann_file,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# === Modelo ===
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=num_classes
)
model.to(device)

# === Optimizer y Loss ===
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# === TensorBoard ===
writer = SummaryWriter()

# === GradScaler solo si hay GPU ===
grad_scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# === Entrenamiento ===
def train():
    model.train()
    for epoch in range(epochs):
        print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
                outputs = model(imgs).logits
                loss = criterion(outputs, labels)

            if grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
            all_preds.extend(preds.astype(int))
            all_labels.extend(labels.detach().cpu().numpy().astype(int))

        avg_loss = running_loss / len(train_loader)
        f1 = f1_score(all_labels, all_preds, average='macro')

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('F1 Score/train', f1, epoch)

        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, F1 Score: {f1:.4f}')

    writer.close()
    print("✅ Entrenamiento completado!")

if __name__ == "__main__":
    train()
