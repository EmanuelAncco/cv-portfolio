import os
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

# CONFIGURACION
BASE_DIR = "D:/Python proyectos 2025/CNN EMANUEL"
IMG_DIR = os.path.join(BASE_DIR, "data", "images")
MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
TXT_PATH = os.path.join(BASE_DIR, "data", "train.txt")
MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PREPROCESADOR
feature_extractor = SegformerFeatureExtractor.from_pretrained(MODEL_NAME)

# DATASET
class CrackDataset(Dataset):
    def __init__(self, txt_file):
        with open(txt_file, 'r') as f:
            self.lines = [line.strip().split() for line in f.readlines()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path = os.path.join(BASE_DIR, 'data', self.lines[idx][0])
        mask_path = os.path.join(BASE_DIR, 'data', self.lines[idx][1])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()
        mask = T.ToTensor()(mask).long().squeeze()
        return pixel_values, mask

# CARGAR DATOS
train_ds = CrackDataset(TXT_PATH)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

# MODELO
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(DEVICE)

# OPTIMIZADOR
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# ENTRENAMIENTO BÁSICO
for epoch in range(1, 6):
    model.train()
    epoch_loss = 0
    for pixel_values, masks in train_loader:
        pixel_values = pixel_values.to(DEVICE)
        masks = masks.to(DEVICE)
        outputs = model(pixel_values=pixel_values, labels=masks)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"\nEpoch {epoch} - Loss: {epoch_loss/len(train_loader):.4f}")

# GUARDAR MODELO
torch.save(model.state_dict(), os.path.join(BASE_DIR, "outputs", "segformer_cracks.pth"))
