# -*- coding: utf-8 -*-
"""
Script de Entrenamiento v3.7 (Final para Publicación) para EMARC VISIÓN

Objetivo:
Entrenar modelos especializados con un pipeline robusto, generando todos los
artefactos necesarios para un artículo científico, incluyendo la parada
temprana y el guardado completo de artefactos.

Mejoras sobre la v3.6:
- Se implementa Early Stopping para detener el entrenamiento si no hay mejora,
  evitando el sobreajuste y ahorrando recursos.
- Se guarda el tokenizer junto con el modelo, crucial para la inferencia y
  la reproducibilidad.
- Se guarda un archivo `hyp.yaml` con todos los hiperparámetros de la
  ejecución para una trazabilidad científica completa.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import logging
import os
import pickle
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yaml # Para guardar los hiperparámetros

# --- 1. CONFIGURACIÓN DE ARGUMENTOS ---
def setup_parser():
    """Configura el parser de argumentos para la línea de comandos."""
    BASE_PROJECT_DIR = r"D:\Python_proyectos_2025\SEGURIDAD2.0\ModeloV3.2"
    parser = argparse.ArgumentParser(description="Script de entrenamiento de modelos especializados de riesgo.")
    parser.add_argument('target_column', type=str, choices=['NatureTitle', 'Part_of_Body_Title', 'EventTitle'], help="La columna objetivo para la cual entrenar el modelo.")
    parser.add_argument('--data_file', type=str, default=os.path.join(BASE_PROJECT_DIR, 'data/fatalities_augmented_FINAL.csv'), help="Ruta al archivo CSV de datos.")
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help="Nombre del modelo Transformer a usar.")
    parser.add_argument('--output_dir', type=str, default=os.path.join(BASE_PROJECT_DIR, 'output'), help="Directorio base para guardar los resultados.")
    parser.add_argument('--log_dir', type=str, default=os.path.join(BASE_PROJECT_DIR, 'logs'), help="Directorio para guardar los archivos de log.")
    parser.add_argument('--max_len', type=int, default=256, help="Longitud máxima de la secuencia.")
    parser.add_argument('--batch_size', type=int, default=16, help="Tamaño del lote.")
    parser.add_argument('--epochs', type=int, default=15, help="Número máximo de épocas de entrenamiento.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Tasa de aprendizaje.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proporción para validación.")
    parser.add_argument('--random_state', type=int, default=42, help="Semilla aleatoria.")
    # --- NUEVO: Parámetro para Early Stopping ---
    parser.add_argument('--patience', type=int, default=3, help="Épocas a esperar sin mejora antes de detener el entrenamiento.")
    return parser

# --- 2. CONFIGURACIÓN DEL LOGGING ---
def setup_logging(log_dir, run_timestamp, target_column):
    log_filename = os.path.join(log_dir, f"training_{target_column}_{run_timestamp}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s', handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()])

# --- 3. PREPROCESAMIENTO DE DATOS ---
def load_and_preprocess_data(filepath, target_column, encoder_path):
    logging.info(f"Cargando datos desde {filepath}...")
    if not os.path.exists(filepath):
        logging.error(f"¡Archivo de datos no encontrado en '{filepath}'!")
        raise FileNotFoundError(f"Asegúrate de que el archivo exista en la ruta especificada.")
    df = pd.read_csv(filepath, encoding='utf-8-sig', delimiter=';')
    logging.info(f"Archivo CSV cargado. Preprocesando para la columna objetivo: '{target_column}'")
    df = df[['FinalNarrative', target_column]].copy()
    df.dropna(inplace=True)
    logging.info(f"Registros antes del filtrado de clases raras: {len(df)}")
    df = df.groupby(target_column).filter(lambda x: len(x) > 1)
    logging.info(f"Registros después del filtrado: {len(df)}. Se eliminaron clases con un solo miembro.")
    df.reset_index(drop=True, inplace=True)
    le = LabelEncoder()
    labels = le.fit_transform(df[target_column])
    logging.info(f"Etiquetas codificadas. Número de clases únicas restantes: {len(le.classes_)}")
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    logging.info(f"LabelEncoder guardado en {encoder_path}")
    return df['FinalNarrative'], labels, le

# --- 4. DATASET Y MODELO DE PYTORCH ---
class SingleLabelDataset(Dataset):
    def __init__(self, narratives, labels, tokenizer, max_len):
        self.narratives = narratives
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.narratives)
    def __getitem__(self, item):
        narrative = str(self.narratives[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(narrative, add_special_tokens=True, max_length=self.max_len, return_token_type_ids=False, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label, dtype=torch.long)}

class RiskClassifier(torch.nn.Module):
    def __init__(self, n_classes, model_name):
        super(RiskClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# --- 5. LÓGICA DE ENTRENAMIENTO Y EVALUACIÓN ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device, class_names, label_ids):
    model.eval()
    total_loss = 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, labels=label_ids, zero_division=0, output_dict=False)
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    return avg_loss, f1, accuracy, report, cm

# --- FUNCIÓN PARA VISUALIZACIÓN ---
def save_plots(history, plot_path):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Curvas de Aprendizaje', fontsize=16)
    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Pérdida de Entrenamiento')
    ax1.plot(epochs_range, history['val_loss'], 'o-', label='Pérdida de Validación')
    ax1.set_title('Pérdida (Loss) vs. Épocas')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax2.plot(epochs_range, history['val_acc'], 'o-', label='Accuracy de Validación')
    ax2.plot(epochs_range, history['val_f1'], 'o-', label='F1-Score (Weighted) de Validación')
    ax2.set_title('Métricas de Validación vs. Épocas')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Puntuación')
    ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Gráficos de curvas de aprendizaje guardados en {plot_path}")

# --- 6. SCRIPT PRINCIPAL ---
def main():
    parser = setup_parser()
    args = parser.parse_args()
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(args.output_dir, args.target_column, run_timestamp)
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_dir, run_timestamp, args.target_column)

    # --- NUEVO: Guardar hiperparámetros para reproducibilidad ---
    with open(os.path.join(run_output_dir, 'hyp.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    logging.info(f"Hiperparámetros guardados en {os.path.join(run_output_dir, 'hyp.yaml')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- INICIANDO ENTRENAMIENTO PARA: {args.target_column} ---")
    logging.info(f"Usando dispositivo: {device}")
    logging.info(f"Parámetros de ejecución: {args}")

    encoder_path = os.path.join(run_output_dir, f"label_encoder_{args.target_column}.pkl")
    model_path = os.path.join(run_output_dir, "risk_predictor_model.bin")
    # --- NUEVO: Ruta para guardar el tokenizer ---
    tokenizer_path = os.path.join(run_output_dir, "tokenizer")
    report_path = os.path.join(run_output_dir, "classification_report.txt")
    cm_path = os.path.join(run_output_dir, "confusion_matrix.png")
    plot_path = os.path.join(run_output_dir, "learning_curves.png")

    try:
        narratives, labels, encoder = load_and_preprocess_data(args.data_file, args.target_column, encoder_path)
        class_names = encoder.classes_
        label_ids = np.arange(len(class_names))
        X_train, X_val, y_train, y_val = train_test_split(narratives, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels)
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
        train_dataset = SingleLabelDataset(X_train.to_numpy(), y_train, tokenizer, args.max_len)
        val_dataset = SingleLabelDataset(X_val.to_numpy(), y_val, tokenizer, args.max_len)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        model = RiskClassifier(n_classes=len(class_names), model_name=args.model_name).to(device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_f1 = 0
        epochs_no_improve = 0 # --- NUEVO: Contador para Early Stopping ---
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(args.epochs):
            logging.info(f'--- Época {epoch + 1}/{args.epochs} ---')
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
            val_loss, val_f1, val_acc, report, cm = eval_model(model, val_loader, loss_fn, device, class_names, label_ids)
            logging.info(f'Pérdida de Entrenamiento: {train_loss:.4f}')
            logging.info(f'Pérdida de Validación: {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1-Score (weighted): {val_f1:.4f}')
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0 # Reiniciar contador
                torch.save(model.state_dict(), model_path)
                # --- NUEVO: Guardar el tokenizer junto al mejor modelo ---
                tokenizer.save_pretrained(tokenizer_path)
                logging.info(f"Nuevo mejor modelo y tokenizer guardados con F1-Score: {best_f1:.4f}")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(f"Reporte de Clasificación para la Época {epoch + 1}\n")
                    f.write(f"F1-Score (weighted): {val_f1:.4f} | Accuracy: {val_acc:.4f}\n\n")
                    f.write(report)
                logging.info(f"Reporte de clasificación detallado guardado en {report_path}")
                plt.figure(figsize=(20, 20))
                sns.heatmap(cm, annot=False, fmt='g') # annot=False es mejor para muchas clases
                plt.title('Matriz de Confusión (Mejor Modelo)')
                plt.ylabel('Etiqueta Real')
                plt.xlabel('Etiqueta Predicha')
                plt.savefig(cm_path)
                plt.close()
                logging.info(f"Matriz de confusión actualizada guardada en {cm_path}")
            else:
                # --- NUEVO: Lógica de Early Stopping ---
                epochs_no_improve += 1
                logging.info(f"No hubo mejora en el F1-Score. Paciencia: {epochs_no_improve}/{args.patience}")
                if epochs_no_improve >= args.patience:
                    logging.info(f"Parada temprana activada después de {epoch + 1} épocas.")
                    break

        save_plots(history, plot_path)
        logging.info(f"--- ENTRENAMIENTO PARA '{args.target_column}' COMPLETADO ---")
        logging.info(f"El mejor modelo y sus artefactos se encuentran en: {run_output_dir}")

    except Exception as e:
        logging.critical(f"Ha ocurrido un error fatal durante el entrenamiento: {e}", exc_info=True)

if __name__ == "__main__":
    main()
