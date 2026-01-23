import os
import torch
import timm
import random
import numpy as np
import time # Importar time para medir el tiempo

from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
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
    # Configura el comportamiento determinista, puede afectar ligeramente la velocidad
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Ruta base donde se dividieron las imágenes en: train/ & val/
    # Estructura:
    # archive/
    #   └── train/
    #       ├── Negative/
    #       └── Positive/
    #   └── val/
    #       ├── Negative/
    #       └── Positive/
    # Asegúrate que la ruta sea correcta para tu sistema
    data_dir = r"D:\Python projectos 2025\CNN EMANUEL\archive" # MODIFICA ESTA RUTA SI ES NECESARIO

    # Hiperparámetros
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES = 2 # Asegúrate que coincida con las carpetas (Negative, Positive)
    NUM_WORKERS = 2 # Número de procesos para cargar datos (0 si da problemas)
    # --- AJUSTE AQUÍ LA FRECUENCIA DE IMPRESIÓN ---
    PRINT_FREQ = 20 # Frecuencia (en lotes) para imprimir progreso (antes era 100)
    # ---------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Dispositivo:", device)

    # =========================================================
    # 2) DATA AUGMENTATION
    # =========================================================
    image_size = 224

    # Usar la media/std de ImageNet es más común para modelos preentrenados en ImageNet
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
        transforms.Resize((image_size, image_size)), # Redimensionar manteniendo aspecto y luego centrar/cortar podría ser mejor
        # transforms.Resize(256), # Ejemplo: Redimensionar lado corto a 256
        # transforms.CenterCrop(image_size), # Luego cortar centro 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # =========================================================
    # 3) DATASSETS & DATALOADERS
    # =========================================================
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Verificar si las rutas existen
    if not os.path.isdir(train_dir):
        print(f"ERROR: El directorio de entrenamiento no existe: {train_dir}")
        return # Salir si no existe
    if not os.path.isdir(val_dir):
        print(f"ERROR: El directorio de validación no existe: {val_dir}")
        return # Salir si no existe

    try:
        train_dataset = datasets.ImageFolder(root=train_dir,
                                             transform=train_transforms)
        val_dataset = datasets.ImageFolder(root=val_dir,
                                           transform=val_transforms)

        # Verificación rápida de clases (opcional, pero útil)
        print("--- Verificación de Clases ---")
        print(f"Clases detectadas: {train_dataset.classes}")
        print(f"Mapeo clase -> índice: {train_dataset.class_to_idx}")
        if len(train_dataset.classes) != NUM_CLASSES:
             print(f"¡ALERTA! Se detectaron {len(train_dataset.classes)} clases, pero NUM_CLASSES es {NUM_CLASSES}.")
        print("-----------------------------")


        # Crear DataLoaders
        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True if device.type == "cuda" else False, # pin_memory solo útil con GPU
                                  persistent_workers=True if NUM_WORKERS > 0 else False, # Puede acelerar inicio de épocas
                                  drop_last=False) # No descartar el último lote si es incompleto

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True if device.type == "cuda" else False,
                                persistent_workers=True if NUM_WORKERS > 0 else False,
                                drop_last=False)

        print(f"Total imágenes de entrenamiento: {len(train_dataset)}")
        print(f"Total imágenes de validación:    {len(val_dataset)}")
        print(f"Lotes por época de entrenamiento: {len(train_loader)}")
        print(f"Lotes por época de validación:   {len(val_loader)}")


    except FileNotFoundError:
        print(f"ERROR: No se pudieron encontrar las carpetas de clases dentro de {train_dir} o {val_dir}.")
        print("Asegúrate que la estructura sea data_dir/train/ClaseA/, data_dir/train/ClaseB/, etc.")
        return
    except Exception as e:
        print(f"ERROR al crear Datasets/DataLoaders: {e}")
        return

    # =========================================================
    # 4) CREACIÓN DEL MODELO (ViT DE TIMM)
    # =========================================================
    model_name = "vit_base_patch16_224.augreg_in21k" # Usar el nombre actual recomendado por timm
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

    # Calcular total_steps correctamente
    # Si drop_last=False (default), len(train_loader) ya considera el último lote parcial
    total_steps = NUM_EPOCHS * len(train_loader)
    print(f"Total steps para OneCycleLR: {total_steps}")

    scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)

    # =========================================================
    # 6) LOOP DE ENTRENAMIENTO Y VALIDACIÓN
    # =========================================================
    best_val_acc = 0.0
    start_time_train = time.time() # Tiempo total de entrenamiento
    print("\n--- Iniciando Entrenamiento ---")

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time() # Tiempo por época

        # --------------------------
        #   ENTRENAMIENTO
        # --------------------------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0 # Contador para manejar último lote incompleto
        batch_start_time = time.time() # Para medir tiempo entre impresiones
        accumulated_batches_since_print = 0 # Contador para tiempo por lote

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True) # non_blocking con pin_memory

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Acumular métricas
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            processed_samples += batch_size
            accumulated_batches_since_print += 1

            # *** IMPRESIÓN DE PROGRESO CADA PRINT_FREQ LOTES ***
            if (i + 1) % PRINT_FREQ == 0 or (i + 1) == len(train_loader):
                batches_processed = i + 1
                batches_total = len(train_loader)
                # Calcular métricas actuales promedio
                current_loss = running_loss / processed_samples
                current_acc = 100. * running_corrects.double() / processed_samples
                # Calcular tiempo transcurrido
                time_elapsed_since_last_print = time.time() - batch_start_time
                # Estimar tiempo restante para la época
                time_per_batch = time_elapsed_since_last_print / accumulated_batches_since_print if accumulated_batches_since_print > 0 else 0
                remaining_batches = batches_total - batches_processed
                eta_epoch_seconds = remaining_batches * time_per_batch
                eta_epoch_str = time.strftime("%H:%M:%S", time.gmtime(eta_epoch_seconds)) if time_per_batch > 0 else "N/A"


                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batches_processed}/{batches_total} "
                      f"| Train Loss (avg): {current_loss:.4f} | Train Acc (avg): {current_acc:.2f}% "
                      f"| LR: {optimizer.param_groups[0]['lr']:.6f} " # Obtener LR actual del optimizador
                      f"| ETA Época: {eta_epoch_str}")
                batch_start_time = time.time() # Reiniciar timer para la próxima impresión
                accumulated_batches_since_print = 0 # Reiniciar contador de lotes
            # *** FIN DE IMPRESIÓN DE PROGRESO ***


        # Calcular métricas finales de la época de entrenamiento
        epoch_loss = running_loss / processed_samples
        epoch_acc = 100. * running_corrects.double() / processed_samples

        # --------------------------
        #   VALIDACIÓN
        # --------------------------
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_processed_samples = 0
        val_start_time = time.time() # Tiempo para validación

        with torch.no_grad():
            for j, (images, labels) in enumerate(val_loader): # Usar 'j' para índice de lote de validación
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size_val = images.size(0)
                val_running_loss += loss.item() * batch_size_val
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
                val_processed_samples += batch_size_val

                # Opcional: Imprimir progreso de validación también
                # if (j + 1) % PRINT_FREQ == 0 or (j + 1) == len(val_loader):
                #    print(f"    Validating... Batch {j+1}/{len(val_loader)}")


        # Calcular métricas finales de la época de validación
        val_epoch_loss = val_running_loss / val_processed_samples
        val_epoch_acc = 100. * val_running_corrects.double() / val_processed_samples

        epoch_end_time = time.time()
        epoch_duration_seconds = epoch_end_time - epoch_start_time
        epoch_duration_str = time.strftime("%H:%M:%S", time.gmtime(epoch_duration_seconds))
        val_duration_seconds = epoch_end_time - val_start_time
        val_duration_str = time.strftime("%M:%S", time.gmtime(val_duration_seconds)) # Duración de validación

        # Imprimir resumen de la época
        print("-" * 80) # Separador más largo
        print(f"FIN ÉPOCA [{epoch+1}/{NUM_EPOCHS}] | Duración Total: {epoch_duration_str} (Val: {val_duration_str})"
              f" | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%"
              f" | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        print("-" * 80) # Separador

        # Guardar el mejor modelo
        if val_epoch_acc > best_val_acc:
            print(f"🔼 Mejora en Validación Acc ({best_val_acc:.2f}% -> {val_epoch_acc:.2f}%). Guardando modelo...")
            best_val_acc = val_epoch_acc
            try:
                # Guardar también la época y el estado del optimizador/scheduler puede ser útil
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }
                torch.save(checkpoint, "best_vit_model_checkpoint.pth")
                # torch.save(model.state_dict(), "best_vit_model_state_dict.pth") # Guardar solo pesos si prefieres
                print(f"✅ Modelo guardado como 'best_vit_model_checkpoint.pth'")
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

# Punto de entrada principal del script
if __name__ == '__main__':
    # freeze_support() # Descomentar SOLO si vas a 'congelar' el script en un ejecutable
    # Añadir manejo de CUDA_LAUNCH_BLOCKING para depuración más fácil si vuelve a fallar
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Descomentar si reaparece el error CUDA assert
    main() # Llamar a la función principal que contiene toda la lógica

