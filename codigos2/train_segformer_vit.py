import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, Trainer, TrainingArguments
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ======================================================
# 1. CONFIGURACIÓN
# ======================================================
train_images_dir = r"D:\Python proyectos 2025\CNN EMANUEL\data\train\images"
train_masks_dir  = r"D:\Python proyectos 2025\CNN EMANUEL\data\train\masks"
val_images_dir   = r"D:\Python proyectos 2025\CNN EMANUEL\data\val\images"
val_masks_dir    = r"D:\Python proyectos 2025\CNN EMANUEL\data\val\masks"

NUM_CLASSES = 2
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 25
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# ======================================================
# 2. PREPROCESAMIENTO
# ======================================================
feature_extractor = SegformerFeatureExtractor(do_reduce_labels=True, size=IMAGE_SIZE)

class CrackSegmentationDataset(Dataset):
    def __init__(self, images_folder, masks_folder, feature_extractor):
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.feature_extractor = feature_extractor
        self.image_filenames = [f for f in os.listdir(images_folder) if f.endswith(".jpg") or f.endswith(".png")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.images_folder, img_name)
        mask_path = os.path.join(self.masks_folder, img_name.replace(".jpg", ".png"))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # una sola clase: grieta o fondo

        encoded_inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoded_inputs["pixel_values"].squeeze()

        mask = mask.resize(IMAGE_SIZE)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)

        labels = torch.tensor(mask, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# ======================================================
# 3. CARGA DE DATOS
# ======================================================
train_dataset = CrackSegmentationDataset(train_images_dir, train_masks_dir, feature_extractor)
val_dataset = CrackSegmentationDataset(val_images_dir, val_masks_dir, feature_extractor)

# ======================================================
# 4. MODELO
# ======================================================
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    id2label={0: "fondo", 1: "grieta"},
    label2id={"fondo": 0, "grieta": 1}
)

# ======================================================
# 5. ENTRENAMIENTO
# ======================================================
args = TrainingArguments(
    output_dir="./segformer_crack_detection",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

def compute_metrics(eval_pred):
    return {}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("segformer_cracks_v1")
