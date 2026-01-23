import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm import tqdm
import json
import joblib
import logging
from torch_geometric.nn import GATConv
from torch.nn import BatchNorm1d


# --- CONFIGURACIÓN DE LOGGING ---
def setup_logging(log_dir):
    log_filename = os.path.join(log_dir, 'training_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )


# --- ESTRUCTURA DEL GRAFO ---
def define_bridge_graph():
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    return edge_index


# --- FASE 1: PRE-PROCESAMIENTO PARA DATOS TEMPORALES ---
def run_preprocessing(data_directory, output_dir, hp):
    logging.info("--- Iniciando Fase de Pre-procesamiento para Dominio del Tiempo ---")
    all_sensor_data = {}
    for sensor_id in range(1, 6):
        logging.info(f"Procesando datos para el sensor {sensor_id}...")
        sensor_files = sorted(
            [f for f in os.listdir(data_directory) if f.endswith('.txt') and f.startswith(f'{sensor_id}_')],
            key=lambda f: f.split('_')[1] + f.split('_')[2].replace('.txt', '')
        )
        if not sensor_files: continue
        signal_parts = [pd.read_csv(os.path.join(data_directory, f), sep='\s+', header=None, usecols=[1]).values for f
                        in sensor_files]
        all_sensor_data[sensor_id] = np.concatenate(signal_parts).flatten()

    if len(all_sensor_data) != 5:
        logging.error("No se encontraron datos para los 5 sensores. Abortando.")
        return None

    logging.info("Ajustando StandardScaler en todos los datos del dominio del tiempo...")
    scaler = StandardScaler()
    scaler.fit(np.concatenate(list(all_sensor_data.values())).reshape(-1, 1))

    min_len = min(len(data) for data in all_sensor_data.values())
    data_stack = np.stack(
        [scaler.transform(data[:min_len].reshape(-1, 1)).flatten() for sid, data in sorted(all_sensor_data.items())],
        axis=1)

    n_samples = (min_len - hp['window_size']) // hp['stride'] + 1
    time_windows = np.array([data_stack[i * hp['stride']: i * hp['stride'] + hp['window_size']] for i in
                             tqdm(range(n_samples), desc="Generando ventanas")])

    # Shape: (n_windows, window_size, n_sensors) -> Necesitamos (n_windows, n_sensors, window_size)
    time_windows_transposed = np.transpose(time_windows, (0, 2, 1))

    preprocessed_dir = os.path.join(output_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)
    features_path = os.path.join(preprocessed_dir, 'time_windows.npy')
    np.save(features_path, time_windows_transposed)
    joblib.dump(scaler, os.path.join(output_dir, 'time_domain_scaler.gz'))

    logging.info(
        f"Pre-procesamiento completo. {time_windows_transposed.shape[0]} muestras guardadas en '{features_path}'")
    return preprocessed_dir


# --- DATASET TEMPORAL ---
class TimeSeriesWindowDataset(Dataset):
    def __init__(self, features_path):
        self.features = np.load(features_path).astype(np.float32)
        logging.info("Dataset de ventanas temporales cargado.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Devuelve [Sensores, Longitud de Ventana]
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.features[idx])


# --- ARQUITECTURA DEL MODELO TCN + GAT ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, features, length)
        return self.network(x)[:, :, -1]  # Tomamos la última salida de la secuencia


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, num_sensors, window_size, tcn_channels, latent_dim=32):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_sensors = num_sensors
        # TCN procesará cada sensor individualmente
        self.tcn_encoder = TCNEncoder(1, tcn_channels)
        tcn_output_dim = tcn_channels[-1]

        # GAT para fusionar la información espacial
        self.gat_encoder = GATConv(tcn_output_dim, latent_dim, heads=1)

        # Decoder
        self.gat_decoder = GATConv(latent_dim, tcn_output_dim, heads=1)
        self.temporal_decoder = nn.Linear(tcn_output_dim, window_size)

    def forward(self, x, edge_index):
        # x shape: (batch_size * num_sensors, 1, window_size)

        # 1. Encoder Temporal (TCN)
        tcn_out = self.tcn_encoder(x)  # Shape: (batch*sensors, tcn_out)

        # 2. Encoder Espacial (GAT)
        latent = self.gat_encoder(tcn_out, edge_index).relu()

        # 3. Decoder Espacial (GAT)
        gat_decoded = self.gat_decoder(latent, edge_index).relu()

        # 4. Decoder Temporal (Linear)
        reconstructed = self.temporal_decoder(gat_decoded)

        # Re-añadir la dimensión de canal para que coincida
        return reconstructed.unsqueeze(1)


# --- FASE 2: ENTRENAMIENTO ---
def run_training(preprocessed_dir, output_dir, hp):
    logging.info("\n--- Iniciando Fase de Entrenamiento Final con TCN+GAT ---")

    dataset = TimeSeriesWindowDataset(os.path.join(preprocessed_dir, 'time_windows.npy'))
    train_len = int(0.85 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = SpatioTemporalAutoencoder(
        num_sensors=5,
        window_size=hp['window_size'],
        tcn_channels=hp['tcn_channels']
    ).to(device)
    base_edge_index = define_bridge_graph().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_tcn_gat_model.pth')

    logging.info(f"Iniciando entrenamiento en {device}...")
    for epoch in range(hp['epochs']):
        model.train()
        avg_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hp['epochs']} [Train]")
        for inputs_batch, targets_batch in progress_bar:
            batch_size, num_sensors, window_len = inputs_batch.shape

            inputs = inputs_batch.reshape(batch_size * num_sensors, 1, window_len).to(device)
            targets = targets_batch.reshape(batch_size * num_sensors, 1, window_len).to(device)

            edge_indices = [base_edge_index + i * num_sensors for i in range(batch_size)]
            batch_edge_index = torch.cat(edge_indices, dim=1)

            optimizer.zero_grad()
            outputs = model(inputs, batch_edge_index)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()

        model.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for inputs_batch, targets_batch in val_loader:
                batch_size, num_sensors, window_len = inputs_batch.shape
                inputs = inputs_batch.reshape(batch_size * num_sensors, 1, window_len).to(device)
                targets = targets_batch.reshape(batch_size * num_sensors, 1, window_len).to(device)
                edge_indices = [base_edge_index + i * num_sensors for i in range(batch_size)]
                batch_edge_index = torch.cat(edge_indices, dim=1)

                outputs = model(inputs, batch_edge_index)
                loss = criterion(outputs, targets)
                avg_val_loss += loss.item()

        avg_train_loss /= len(train_loader)
        avg_val_loss /= len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f"Epoch {epoch + 1}/{hp['epochs']} -> Train Loss: {avg_train_loss:.8f}, Val Loss: {avg_val_loss:.8f}, LR: {current_lr:.1e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logging.info(f"   -> Nuevo mejor modelo guardado con Val Loss: {best_val_loss:.8f}")
        else:
            patience_counter += 1
        if patience_counter >= hp['patience']:
            logging.info("--- Parada Temprana ---")
            break
    logging.info("--- Entrenamiento Finalizado ---")


# --- EJECUCIÓN DEL EXPERIMENTO ---
if __name__ == '__main__':
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    HP = {
        "window_size": 512, "stride": 512,
        "tcn_channels": [16, 32, 32],  # Canales de salida para cada capa TCN
        "epochs": 100, "batch_size": 64,  # Batch size más pequeño para un modelo más complejo
        "learning_rate": 1e-3,
        "patience": 15
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = os.path.join("resultados_entrenamiento", f"run_tcn_gat_final_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)
    setup_logging(output_directory)

    if not os.path.isdir(data_folder_path):
        logging.error(f"Error: El directorio de datos no existe: {data_folder_path}")
    else:
        preprocessed_data_dir = run_preprocessing(data_folder_path, output_directory, HP)
        if preprocessed_data_dir:
            run_training(preprocessed_data_dir, output_directory, HP)

        hp_path = os.path.join(output_directory, 'hyperparameters.json')
        with open(hp_path, 'w') as f:
            json.dump(HP, f, indent=4)
        logging.info(f"Hiperparámetros guardados en: {hp_path}")
