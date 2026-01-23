import os
import torch
import timm
import numpy as np
import time # Importar time
# Importaciones adicionales para curvas ROC y PR
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns # Para graficar la matriz de confusión
import matplotlib.pyplot as plt # Para graficar

from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from multiprocessing import freeze_support # Para Windows con num_workers > 0

# Función principal para encapsular la lógica de evaluación
def evaluate():
    # =========================================================
    # 1) CONFIGURACIÓN (Ajustar según tu entrenamiento)
    # =========================================================
    # --- Parámetros del Modelo ---
    model_name = "vit_base_patch16_224.augreg_in21k" # Debe ser el mismo que usaste para entrenar
    NUM_CLASSES = 2 # Número de clases (Negative, Positive)
    image_size = 224

    # --- Rutas ---
    # Ruta EXACTA al archivo .pth con los pesos guardados
    model_path = r"D:\Python_proyectos_2025\PyCharmMiscProject\best_vit_model.pth" # <-- RUTA DEL MODELO

    # Ruta base donde están las carpetas 'train/' y 'val/' del DATASET
    # --- ACTUALIZADO: Verifica que esta sea la ruta correcta a la carpeta 'archive' ---
    data_dir = r"D:\Python_proyectos_2025\CNN EMANUEL\archive"
    # ---------------------------------------------------------------------------------
    val_dir = os.path.join(data_dir, "val") # Usaremos el conjunto de validación para evaluar

    # --- Otros Parámetros ---
    BATCH_SIZE = 32 # Puede ser mayor que en el entrenamiento si tienes suficiente VRAM
    NUM_WORKERS = 2 # 0 si da problemas
    # --- Frecuencia de impresión para evaluación ---
    EVAL_PRINT_FREQ = 20
    # -------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Dispositivo: {device}")

    # =========================================================
    # 2) TRANSFORMACIONES DE VALIDACIÓN (Deben ser las mismas)
    # =========================================================
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # =========================================================
    # 3) DATASET Y DATALOADER DE VALIDACIÓN
    # =========================================================
    if not os.path.isdir(val_dir):
        print(f"ERROR: El directorio de validación del dataset no existe: {val_dir}")
        print("Por favor, verifica la variable 'data_dir'.")
        return # Salir de la función evaluate si no existe

    try:
        val_dataset = datasets.ImageFolder(root=val_dir,
                                           transform=val_transforms)

        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False, # No barajar para evaluación
                                num_workers=NUM_WORKERS,
                                pin_memory=True if device.type == "cuda" else False,
                                persistent_workers=True if NUM_WORKERS > 0 else False)

        print(f"Total imágenes de validación para evaluar: {len(val_dataset)}")
        print(f"Clases encontradas: {val_dataset.classes}")
        class_names = val_dataset.classes # Guardar nombres de clases para reporte
        # Asumiendo que 'Positive' es la clase 1 (importante para ROC/PR)
        positive_class_index = val_dataset.class_to_idx.get('Positive', 1)
        print(f"Índice de la clase positiva ('Positive'): {positive_class_index}")


    except Exception as e:
        print(f"ERROR al cargar el dataset de validación desde '{val_dir}': {e}")
        return # Salir de la función evaluate

    # =========================================================
    # 4) CARGAR MODELO Y PESOS
    # =========================================================
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el archivo de pesos en la ruta especificada:")
        print(f" - {model_path}")
        return # Salir de la función evaluate

    try:
        # Crear la arquitectura del modelo (sin pesos preentrenados de timm esta vez)
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        print(f"Arquitectura del modelo '{model_name}' creada.")

        # Cargar los pesos desde el archivo .pth especificado
        print(f"Intentando cargar state_dict desde: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')

        # Comprobar si el state_dict está anidado
        if 'model_state_dict' in state_dict:
            print("Detectado state_dict anidado (probablemente de un checkpoint). Extrayendo...")
            model.load_state_dict(state_dict['model_state_dict'])
            loaded_epoch = state_dict.get('epoch', 'N/A')
            loaded_acc = state_dict.get('best_val_acc', 'N/A')
            print(f"   Info del checkpoint: Época: {loaded_epoch}, Val Acc guardada: {loaded_acc}%")
        else:
            model.load_state_dict(state_dict)

        print(f"State_dict cargado exitosamente desde {model_path}")

        model = model.to(device) # Mover modelo al dispositivo

    except Exception as e:
        print(f"ERROR al cargar el modelo o los pesos desde '{model_path}': {e}")
        return # Salir de la función evaluate

    # =========================================================
    # 5) EVALUACIÓN DEL MODELO (MODIFICADO PARA GUARDAR SCORES)
    # =========================================================
    model.eval() # ¡MUY IMPORTANTE! Poner el modelo en modo evaluación

    all_preds = []      # Para guardar las predicciones finales (0 o 1)
    all_labels = []     # Para guardar las etiquetas verdaderas
    all_probs = []      # NUEVO: Para guardar las probabilidades de la clase positiva
    running_corrects = 0
    total_samples = 0

    print("\n--- Iniciando Evaluación ---")
    eval_start_time = time.time()

    with torch.no_grad(): # No calcular gradientes durante la evaluación
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            # Aplicar Softmax para obtener probabilidades
            probs = torch.softmax(outputs, dim=1)
            # Obtener la clase predicha (la que tiene mayor probabilidad)
            _, preds = torch.max(outputs, 1)

            # Guardar etiquetas verdaderas, predicciones finales y probabilidades de la clase positiva
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, positive_class_index].cpu().numpy()) # Guardar prob de clase positiva

            # Calcular correctas para accuracy simple
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            # *** IMPRESIÓN DE PROGRESO CADA EVAL_PRINT_FREQ LOTES ***
            if (i + 1) % EVAL_PRINT_FREQ == 0 or (i + 1) == len(val_loader):
                 current_eval_acc = 100. * running_corrects.double() / total_samples
                 print(f"  Evaluando... Lote {i+1}/{len(val_loader)} | Acc parcial: {current_eval_acc:.2f}%")
            # *** FIN DE IMPRESIÓN DE PROGRESO ***

    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time
    print(f"--- Evaluación Finalizada (Duración: {eval_duration:.2f}s) ---")

    # Convertir listas a arrays numpy para sklearn
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # =========================================================
    # 6) CÁLCULO Y VISUALIZACIÓN DE MÉTRICAS
    # =========================================================
    if total_samples == 0:
        print("ERROR: No se procesaron muestras. Verifica el dataloader de validación.")
        return # Salir de la función evaluate

    # --- Precisión (Accuracy) ---
    accuracy = 100. * running_corrects.double() / total_samples
    print(f"\n📊 Precisión Global (Accuracy): {accuracy:.2f}% ({running_corrects.item()}/{total_samples})")

    # --- Reporte de Clasificación ---
    print("\n📊 Reporte de Clasificación Detallado:")
    try:
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
        print(report)
    except ValueError as e:
        print(f"Advertencia al generar reporte de clasificación: {e}")
        print(f"Etiquetas únicas verdaderas: {np.unique(all_labels)}")
        print(f"Etiquetas únicas predichas: {np.unique(all_preds)}")

    # --- Matriz de Confusión ---
    print("\n📊 Matriz de Confusión:")
    try:
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicción (Predicted)')
        plt.ylabel('Verdadero (Actual)')
        plt.title('Matriz de Confusión')
        plt.tight_layout() # Ajustar layout
        plt.savefig("confusion_matrix.png")
        print("\n📈 Matriz de confusión guardada como 'confusion_matrix.png'")
        # plt.show() # Descomentar para mostrar interactivamente
        plt.close() # Cerrar la figura actual

    except ValueError as e:
         print(f"Advertencia al generar matriz de confusión: {e}")

    # --- NUEVO: Curva ROC y AUC ---
    print("\n📊 Calculando y graficando Curva ROC...")
    try:
        # Calcular fpr, tpr para varios umbrales
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs, pos_label=positive_class_index)
        # Calcular el Área Bajo la Curva (AUC)
        roc_auc = auc(fpr, tpr)
        print(f"   Área Bajo la Curva ROC (AUC): {roc_auc:.4f}")

        # Graficar Curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio') # Línea de referencia
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (False Positive Rate)')
        plt.ylabel('Tasa de Verdaderos Positivos (True Positive Rate / Recall)')
        plt.title('Curva ROC (Receiver Operating Characteristic)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        print("📈 Curva ROC guardada como 'roc_curve.png'")
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"Error al generar la curva ROC: {e}")

    # --- NUEVO: Curva Precision-Recall y AP ---
    print("\n📊 Calculando y graficando Curva Precision-Recall...")
    try:
        # Calcular precision, recall para varios umbrales
        precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs, pos_label=positive_class_index)
        # Calcular el Average Precision (AP), que resume la curva PR
        average_precision = average_precision_score(all_labels, all_probs, pos_label=positive_class_index)
        print(f"   Precisión Promedio (Average Precision - AP): {average_precision:.4f}")

        # Graficar Curva Precision-Recall
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'Curva PR (AP = {average_precision:.4f})')
        # Línea de referencia: rendimiento de un clasificador aleatorio (proporción de positivos)
        no_skill = len(all_labels[all_labels==positive_class_index]) / len(all_labels)
        plt.plot([0, 1], [no_skill, no_skill], color='grey', lw=2, linestyle='--', label=f'Clasificador Aleatorio (AP ≈ {no_skill:.4f})')
        plt.xlabel('Recall (Exhaustividad / True Positive Rate)')
        plt.ylabel('Precision (Precisión)')
        plt.title('Curva Precision-Recall')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="best") # Usar 'best' para mejor ubicación
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("precision_recall_curve.png")
        print("📈 Curva Precision-Recall guardada como 'precision_recall_curve.png'")
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"Error al generar la curva Precision-Recall: {e}")


    print("\nEvaluación completada.")


# Punto de entrada principal del script
if __name__ == '__main__':
    # freeze_support() # Descomentar SOLO si vas a 'congelar' el script en un ejecutable .exe
    evaluate() # Llamar a la función principal que contiene toda la lógica de evaluación
