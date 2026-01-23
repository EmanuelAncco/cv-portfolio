import os
import torch
import timm
import random
import numpy as np
import time # Importar time
import matplotlib.pyplot as plt # Importar para gráficos
from tqdm.auto import tqdm # Importar tqdm para barras de progreso

from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
# --- NUEVO: Para Precisión Mixta (AMP) ---
from torch.cuda.amp import GradScaler, autocast
# -----------------------------------------
# Necesario para multiprocessing en Windows si se congela la app
from multiprocessing import freeze_support

# Función principal para encapsular la lógica
def main():
    # =========================================================
    # 1) CONFIGURACIÓN INICIAL
    # =========================================================
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Ruta base donde se dividieron las imágenes en: train/ & val/
    data_dir = r"D:\Python_proyectos_2025\CNN EMANUEL\archive"
    output_dir = "." # Directorio donde guardar modelo y gráficos (directorio actual)
    os.makedirs(output_dir, exist_ok=True)

    # Hiperparámetros
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES = 2
    NUM_WORKERS = 2
    # PRINT_FREQ ya no es necesario con tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo: {device}")
    # --- NUEVO: Habilitar AMP si se usa CUDA ---
    use_amp = device.type == 'cuda'
    print(f"⚡ Usando Precisión Mixta Automática (AMP): {use_amp}")
    # -------------------------------------------

    # =========================================================
    # 2) DATA AUGMENTATION
    # =========================================================
    image_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # =========================================================
    # 3) DATASSETS & DATALOADERS
    # =========================================================
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"ERROR: No se encuentran los directorios 'train' o 'val' dentro de: {data_dir}")
        return

    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

        print("--- Verificación de Clases ---")
        print(f"Clases detectadas: {train_dataset.classes}")
        print(f"Mapeo clase -> índice: {train_dataset.class_to_idx}")
        if len(train_dataset.classes) != NUM_CLASSES:
             print(f"¡ALERTA! Se detectaron {len(train_dataset.classes)} clases, pero NUM_CLASSES es {NUM_CLASSES}.")
        print("-----------------------------")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True if device.type == "cuda" else False,
                                  persistent_workers=True if NUM_WORKERS > 0 else False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True if device.type == "cuda" else False,
                                persistent_workers=True if NUM_WORKERS > 0 else False)

        print(f"Total imágenes de entrenamiento: {len(train_dataset)}")
        print(f"Total imágenes de validación:    {len(val_dataset)}")
        print(f"Lotes por época de entrenamiento: {len(train_loader)}")
        print(f"Lotes por época de validación:   {len(val_loader)}")

    except Exception as e:
        print(f"ERROR al crear Datasets/DataLoaders: {e}")
        return

    # =========================================================
    # 4) CREACIÓN DEL MODELO (ViT DE TIMM)
    # =========================================================
    model_name = "vit_base_patch16_224.augreg_in21k"
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
        model = model.to(device)
        print(f"Modelo '{model_name}' cargado correctamente.")
    except Exception as e:
        print(f"ERROR al crear el modelo '{model_name}': {e}")
        return

    # =========================================================
    # 5) OPTIMIZADOR, CRITERIO Y SCHEDULER
    # =========================================================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    total_steps = NUM_EPOCHS * len(train_loader)
    print(f"Total steps para OneCycleLR: {total_steps}")
    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)
    # --- NUEVO: GradScaler para AMP ---
    scaler = GradScaler(enabled=use_amp)
    # ---------------------------------

    # =========================================================
    # 6) LOOP DE ENTRENAMIENTO Y VALIDACIÓN + HISTORIAL + TQDM + AMP
    # =========================================================
    best_val_acc = 0.0
    start_time_train = time.time()
    print("\n--- Iniciando Entrenamiento ---")

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # --- ENTRENAMIENTO ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0

        # --- NUEVO: Envolver train_loader con tqdm ---
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        # -------------------------------------------

        for i, (images, labels) in train_pbar: # Iterar sobre la barra de progreso
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # --- NUEVO: Contexto autocast para AMP ---
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            # ----------------------------------------

            optimizer.zero_grad(set_to_none=True)
            # --- NUEVO: Escalar la pérdida con scaler ---
            scaler.scale(loss).backward()
            # --- NUEVO: scaler.step() actualiza el optimizador ---
            scaler.step(optimizer)
            # --- NUEVO: Actualizar el scaler para la próxima iteración ---
            scaler.update()
            # -------------------------------------------------------
            scheduler.step() # Actualizar scheduler en cada paso

            # Calcular métricas para mostrar en tqdm
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            processed_samples += batch_size

            # --- NUEVO: Actualizar descripción de tqdm ---
            # Calcular métricas promedio actuales
            current_avg_loss = running_loss / processed_samples
            current_avg_acc = 100. * running_corrects.double() / processed_samples
            train_pbar.set_postfix(loss=f"{current_avg_loss:.4f}", acc=f"{current_avg_acc:.2f}%", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            # -------------------------------------------

        epoch_loss = running_loss / processed_samples
        epoch_acc = 100. * running_corrects.double() / processed_samples

        # --- VALIDACIÓN ---
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_processed_samples = 0
        val_start_time = time.time()

        # --- NUEVO: Envolver val_loader con tqdm ---
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        # -----------------------------------------

        with torch.no_grad():
            for j, (images, labels) in val_pbar: # Iterar sobre la barra de progreso
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # --- NUEVO: Contexto autocast también para validación (consistencia) ---
                with autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                # ----------------------------------------------------------------------

                batch_size_val = images.size(0)
                val_running_loss += loss.item() * batch_size_val
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                val_processed_samples += batch_size_val

                # --- NUEVO: Actualizar descripción de tqdm para validación ---
                current_val_avg_loss = val_running_loss / val_processed_samples
                current_val_avg_acc = 100. * val_running_corrects.double() / val_processed_samples
                val_pbar.set_postfix(loss=f"{current_val_avg_loss:.4f}", acc=f"{current_val_avg_acc:.2f}%")
                # ----------------------------------------------------------

        val_epoch_loss = val_running_loss / val_processed_samples
        val_epoch_acc = 100. * val_running_corrects.double() / val_processed_samples

        epoch_end_time = time.time()
        epoch_duration_seconds = epoch_end_time - epoch_start_time
        epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration_seconds))

        # Guardar métricas de la época en el historial
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc.cpu().numpy())
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc.cpu().numpy())

        # Imprimir resumen final de la época (tqdm ya mostró progreso)
        print("-" * 80)
        print(f"FIN ÉPOCA [{epoch+1}/{NUM_EPOCHS}] | Duración: {epoch_duration_str}"
              f" | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%"
              f" | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        print("-" * 80)

        # Guardar el mejor modelo
        if val_epoch_acc > best_val_acc:
            print(f"🔼 Mejora en Validación Acc ({best_val_acc:.2f}% -> {val_epoch_acc:.2f}%). Guardando modelo...")
            best_val_acc = val_epoch_acc
            try:
                checkpoint_path = os.path.join(output_dir, "best_vit_model_checkpoint.pth")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_loss_history': train_loss_history,
                    'train_acc_history': train_acc_history,
                    'val_loss_history': val_loss_history,
                    'val_acc_history': val_acc_history,
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"✅ Modelo guardado como '{checkpoint_path}'")
            except Exception as e:
                print(f"❌ ERROR al guardar el modelo: {e}")
        else:
             print(f"🔻 No hubo mejora respecto al mejor Val Acc ({best_val_acc:.2f}%)")


    end_time_train = time.time()
    total_train_duration_seconds = end_time_train - start_time_train
    total_train_duration_str = time.strftime("%H:%M:%S", time.gmtime(total_train_duration_seconds))

    print("\n--- Entrenamiento Finalizado ---")
    print(f"Duración Total del Entrenamiento: {total_train_duration_str}")
    print(f"Mejor accuracy de validación obtenido: {best_val_acc:.2f}%")

    # =========================================================
    # 7) GRAFICAR HISTORIAL DE ENTRENAMIENTO
    # =========================================================
    print("\n--- Generando Gráficos del Historial de Entrenamiento ---")
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Gráfico de Precisión (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_history, label='Precisión Entrenamiento', marker='o')
    plt.plot(epochs_range, val_acc_history, label='Precisión Validación', marker='o')
    plt.title('Historial de Precisión (Accuracy)')
    plt.xlabel('Época')
    plt.ylabel('Precisión (%)')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Gráfico de Pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label='Pérdida Entrenamiento', marker='o')
    plt.plot(epochs_range, val_loss_history, label='Pérdida Validación', marker='o')
    plt.title('Historial de Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    history_plot_path = os.path.join(output_dir, "training_history.png")
    try:
        plt.savefig(history_plot_path)
        print(f"📈 Gráficos del historial guardados como '{history_plot_path}'")
    except Exception as e:
        print(f"❌ ERROR al guardar los gráficos del historial: {e}")
    plt.close()

# Punto de entrada principal del script
if __name__ == '__main__':
    # freeze_support() # Descomentar SOLO si vas a 'congelar' el script en un ejecutable .exe
    # --- NUEVO: Instalar tqdm si no lo tienes: pip install tqdm ---
    main()
