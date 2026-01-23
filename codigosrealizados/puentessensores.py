# -*- coding: utf-8 -*-
"""
Script de entrenamiento con Muestras Coherentes.
Versión 5: Trata cada archivo .txt como una serie temporal independiente
para generar ventanas de datos coherentes y evitar aprender de saltos artificiales.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json


# --- PASO 1: INGENIERÍA DE CARACTERÍSTICAS (Ahora opera sobre una sola serie) ---

def create_features_from_series(data_series, window_size=50):
    df = pd.DataFrame(data_series, columns=['acceleration'])
    df['mean'] = df['acceleration'].rolling(window=window_size).mean()
    df['std'] = df['acceleration'].rolling(window=window_size).std()
    df['skew'] = df['acceleration'].rolling(window=window_size).skew()
    df['kurt'] = df['acceleration'].rolling(window=window_size).kurt()
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df.values


# --- NUEVA LÓGICA DE DATASET ---

class TimeSeriesWindowDataset(Dataset):
    """Crea ventanas a partir de una ÚNICA serie temporal continua."""

    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1
        if self.n_samples < 0:
            self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# --- MODELO (Sin Cambios) ---

class PredictiveFeatureModel(nn.Module):
    def __init__(self, pred_len, input_size, hidden_size, num_layers, dropout, model_type='lstm'):
        super(PredictiveFeatureModel, self).__init__()
        self.pred_len = pred_len
        self.input_size = input_size
        print(f"Usando modelo avanzado para {input_size} características.")

        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                               dropout=dropout)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                              dropout=dropout)

        self.fc_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size * pred_len)
        )

    def forward(self, x_enc):
        rnn_outputs, _ = self.rnn(x_enc)
        last_output = rnn_outputs[:, -1, :]
        prediction = self.fc_net(last_output)
        return prediction.view(x_enc.size(0), self.pred_len, self.input_size)


# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO (Reescrita) ---

def run_experiment(data_directory, output_dir, hp):
    # 1. Cargar cada archivo como una serie separada
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if
                 f.endswith('.txt') and f.startswith('1_')]

    # Solo usaremos los datos del sensor 1
    print(f"Encontrados {len(all_files)} archivos para el sensor 1.")

    all_series_features = []
    for file_path in tqdm(all_files, desc="Procesando archivos"):
        # Cargar la serie cruda
        series_df = pd.read_csv(file_path, sep='\s+', header=None, names=['timestamp', 'acceleration'])
        # Crear características para esta serie
        series_features = create_features_from_series(series_df['acceleration'].values,
                                                      window_size=hp['feature_window'])
        all_series_features.append(series_features)

    # 2. Dividir las SERIES en train/val/test
    train_series, test_series = train_test_split(all_series_features, test_size=0.15, shuffle=True, random_state=42)
    train_series, val_series = train_test_split(train_series, test_size=0.176, shuffle=True, random_state=42)

    # 3. Escalar los datos usando SOLO el conjunto de entrenamiento
    scaler = StandardScaler()
    # Concatenar todas las series de entrenamiento para ajustar el scaler
    concatenated_train_data = np.concatenate(train_series, axis=0)
    scaler.fit(concatenated_train_data)

    # 4. Crear datasets de ventanas para cada conjunto (train, val, test)
    def create_concatenated_dataset(series_list, scaler, seq_len, pred_len):
        datasets = []
        for series in series_list:
            scaled_series = scaler.transform(series)
            datasets.append(TimeSeriesWindowDataset(scaled_series, seq_len, pred_len))
        return ConcatDataset(datasets)

    train_dataset = create_concatenated_dataset(train_series, scaler, hp['seq_len'], hp['pred_len'])
    val_dataset = create_concatenated_dataset(val_series, scaler, hp['seq_len'], hp['pred_len'])
    test_dataset = create_concatenated_dataset(test_series, scaler, hp['seq_len'], hp['pred_len'])

    print(f"Total de ventanas de entrenamiento: {len(train_dataset)}")
    print(f"Total de ventanas de validación: {len(val_dataset)}")
    print(f"Total de ventanas de prueba: {len(test_dataset)}")

    # 5. DataLoaders y Entrenamiento (el resto es muy similar)
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = all_series_features[0].shape[1]

    model = PredictiveFeatureModel(
        pred_len=hp['pred_len'], input_size=num_features,
        hidden_size=hp['hidden_size'], num_layers=hp['num_layers'],
        dropout=hp['dropout'], model_type=hp['model_type']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    history = {'train_loss': [], 'val_loss': []}

    print("\n--- Iniciando Entrenamiento con Muestras Coherentes ---")
    for epoch in range(hp['epochs']):
        model.train()
        avg_train_loss = 0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Entrenando]', leave=False)
        for seq_x, seq_y in progress_bar_train:
            seq_x, seq_y = seq_x.to(device), seq_y.to(device)
            optimizer.zero_grad()
            output = model(seq_x)
            loss = criterion(output, seq_y)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        history['train_loss'].append(avg_train_loss / len(train_loader))

        model.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for seq_x, seq_y in val_loader:
                seq_x, seq_y = seq_x.to(device), seq_y.to(device)
                output = model(seq_x)
                loss = criterion(output, seq_y)
                avg_val_loss += loss.item()
        history['val_loss'].append(avg_val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{hp['epochs']} -> Train Loss: {history['train_loss'][-1]:.6f}, Val Loss: {history['val_loss'][-1]:.6f}")

        scheduler.step(history['val_loss'][-1])
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"   -> Nuevo mejor modelo guardado con Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"   -> Val Loss no mejoró. Paciencia: {patience_counter}/{hp['patience']}")

        if patience_counter >= hp['patience']:
            print("--- Parada Temprana (Early Stopping) activada ---")
            break

    print("--- Entrenamiento Finalizado ---\n")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # ... (El código de evaluación y guardado es igual al anterior)
    idx = np.random.randint(0, len(test_dataset))
    seq_x, seq_y_true = test_dataset[idx]

    with torch.no_grad():
        input_tensor = seq_x.unsqueeze(0).to(device)
        prediction_scaled = model(input_tensor).squeeze(0).cpu().numpy()

    prediction_unscaled = scaler.inverse_transform(prediction_scaled)
    actual_unscaled = scaler.inverse_transform(seq_y_true.numpy())

    num_features = actual_unscaled.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 4 * num_features), sharex=True)
    feature_names = ['Aceleración', 'Media Móvil', 'Desv. Estándar Móvil', 'Asimetría Móvil', 'Curtosis Móvil']

    for i in range(num_features):
        ax = axes[i]
        ax.plot(actual_unscaled[:, i], label=f'{feature_names[i]} Real', marker='o', markersize=4)
        ax.plot(prediction_unscaled[:, i], label=f'{feature_names[i]} Predicha', linestyle='--')
        ax.set_title(f'Comparación de Característica: {feature_names[i]}')
        ax.legend()
        ax.grid(True)

    plt.xlabel('Paso de Tiempo Futuro')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_test_prediction_all_features.png'))
    plt.show()

    final_metrics = {'best_val_loss': best_val_loss}
    final_results = {"hiperparametros": hp, "metricas_finales": final_metrics}
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
    print("Resultados finales guardados.")


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == '__main__':
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    HP = {
        "model_type": "gru",
        "feature_window": 100,
        "seq_len": 128,
        "pred_len": 32,
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 0.001,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "patience": 10
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = os.path.join("resultados_entrenamiento", f"run_coherent_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)
    print(f"Los resultados se guardarán en: {output_directory}")

    if not os.path.isdir(data_folder_path):
        print(f"Error: El directorio '{data_folder_path}' no existe.")
    else:
        run_experiment(data_folder_path, output_directory, HP)

