import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.datasets import CocoDetection
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ============================
# Configuración
# ============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {DEVICE}")

# Rutas
TRAIN_DIR = 'PPE-0.3.v1i.coco/train/images'
VAL_DIR = 'PPE-0.3.v1i.coco/valid/images'
TRAIN_ANN = 'PPE-0.3.v1i.coco/train/_annotations.coco.limpiotrain.json'
VAL_ANN = 'PPE-0.3.v1i.coco/valid/_annotations.coco.limpiovalid.json'
CHECKPOINT_PATH = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/checkpoints/modelo_detr_ppe_epoch_11.pth'
OUTPUT_MODEL = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/checkpoints/modelo_detr_ppe_epoch_20.pth'
OUTPUT_PLOT = 'D:/Python proyectos 2025/SEGURIDAD_CONSTRUCCION/checkpoints/loss_plot.png'

# Parámetros
START_EPOCH = 11
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-5

# ============================
# Dataset personalizado COCO
# ============================

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class CustomCocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        annotations = {
            'image_id': self.ids[idx],
            'annotations': targets
        }
        encoding = processor(images=img, annotations=annotations, return_tensors="pt")
        encoding = {k: (v.squeeze() if isinstance(v, torch.Tensor) else v) for k, v in encoding.items()}
        return encoding

def collate_fn(batch):
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# ============================
# Preparación de datos
# ============================

train_dataset = CustomCocoDataset(TRAIN_DIR, TRAIN_ANN)
val_dataset = CustomCocoDataset(VAL_DIR, VAL_ANN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

categorias = train_dataset.coco.cats
id2label = {k: v['name'] for k, v in categorias.items()}
label2id = {v: k for k, v in id2label.items()}

# ============================
# Carga del modelo base
# ============================

print("\n🔄 Cargando modelo base DETR...")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
).to(DEVICE)

print("🔄 Cargando pesos anteriores entrenados...")
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
print("✅ Modelo cargado desde checkpoint previo!")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ============================
# Entrenamiento
# ============================

train_losses = []
val_losses = []

for epoch in range(START_EPOCH, EPOCHS + 1):
    print(f"\n🚀 Época {epoch}/{EPOCHS}")
    model.train()
    total_train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoca {epoch}/{EPOCHS}')

    for batch in progress_bar:
        pixel_values = batch['pixel_values'].to(DEVICE)
        labels = [{k: v.to(DEVICE) for k, v in label.items()} for label_list in batch['labels'] for label in label_list]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validación
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in label.items()} for label_list in batch['labels'] for label in label_list]

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_val_loss += outputs.loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'✅ Epoch {epoch} completada - Loss Entrenamiento: {avg_train_loss:.4f}, Loss Validación: {avg_val_loss:.4f}')

# ============================
# Guardar modelo final
# ============================

torch.save(model.state_dict(), OUTPUT_MODEL)
print(f'✅ Modelo final guardado en: {OUTPUT_MODEL}')

# ============================
# Graficar pérdidas
# ============================

plt.figure(figsize=(10, 5))
plt.plot(range(START_EPOCH, EPOCHS + 1), train_losses, label='Entrenamiento')
plt.plot(range(START_EPOCH, EPOCHS + 1), val_losses, label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig(OUTPUT_PLOT)
print(f'✅ Gráfico de pérdidas guardado en: {OUTPUT_PLOT}')
print("🎉 Fine-tuning completado!")
