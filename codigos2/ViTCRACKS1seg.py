import os
import torch
import timm
import random
import numpy as np

from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# =========================================================
# FUNCIÓN DE CREACIÓN DEL MODELO
# =========================================================
def create_fn():
    model_name = "vit_base_patch16_224_in21k"
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    return model

# =========================================================
# ENTRENAMIENTO (sólo si se ejecuta directamente este script)
# =========================================================
if __name__ == "__main__":
    # 1) CONFIGURACIÓN INICIAL
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    data_dir = r"D:\Python proyectos 2025\CNN EMANUEL\archive"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES = 2
    image_size = 224

    # 2) DATA AUGMENTATION
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 3) DATASETS & DATALOADERS
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Total imágenes de entrenamiento: {len(train_dataset)}")
    print(f"Total imágenes de validación:    {len(val_dataset)}")

    # 4) CREACIÓN DEL MODELO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Dispositivo:", device)

    model = create_fn().to(device)

    # 5) OPTIMIZADOR, CRITERIO Y SCHEDULER
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)

    # 6) LOOP DE ENTRENAMIENTO Y VALIDACIÓN
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100. * running_corrects.double() / len(train_dataset)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = 100. * val_running_corrects.double() / len(val_dataset)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]"
              f" | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%"
              f" | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), "best_vit_model.pth")
            print("🔽 Nuevo mejor modelo guardado (best_vit_model.pth)")

    print("Entrenamiento finalizado.")
    print(f"Mejor accuracy de validación: {best_val_acc:.2f}%")
