import torch
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchvision.datasets import CocoDetection
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'✅ Usando dispositivo: {device}')

# Procesador
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
    batch = {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
    return batch

def main():
    # Rutas
    train_dir = 'PPE-0.3.v1i.coco/train/images'
    val_dir = 'PPE-0.3.v1i.coco/valid/images'
    train_ann = 'PPE-0.3.v1i.coco/train/_annotations.coco.limpiotrain.json'
    val_ann = 'PPE-0.3.v1i.coco/valid/_annotations.coco.limpiovalid.json'

    # Dataset
    train_dataset = CustomCocoDataset(train_dir, train_ann)
    val_dataset = CustomCocoDataset(val_dir, val_ann)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Clases
    categorias = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in categorias.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Modelo
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Entrenamiento
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Época {epoch+1}/{epochs}')

        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label_list in batch['labels'] for label in
                      label_list]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f'✅ Época {epoch+1}/{epochs} terminada - Loss promedio: {avg_loss:.4f}')

    # Guardar modelo
    torch.save(model.state_dict(), 'modelo_detr_ppe.pth')
    print('✅ Modelo guardado: modelo_detr_ppe.pth')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
