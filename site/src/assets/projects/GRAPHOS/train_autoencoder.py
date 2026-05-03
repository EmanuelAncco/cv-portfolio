
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import joblib


from torch_geometric.nn import GCNConv


# --- ESTRUCTURA DEL GRAFO Y LÓGICA DE DATOS ---

def define_bridge_graph():

    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    return edge_index


class SpatioTemporalWindowDataset(Dataset):
    def __init__(self, data_dict, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride
        min_len = min(len(data) for data in data_dict.values())
        self.data = np.stack([data[:min_len] for sid, data in sorted(data_dict.items())], axis=1)
        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples < 0: self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        return torch.FloatTensor(window), torch.FloatTensor(window)


# --- ARQUITECTURA DEL GNN AUTOENCODER ---

class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden=32, gnn_out=16, rnn_hidden=64):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(input_size=gnn_out * num_nodes, hidden_size=rnn_hidden, batch_first=True,
                                  num_layers=2)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=gnn_out * num_nodes, batch_first=True,
                                  num_layers=2)
        self.gnn_decoder = GNNLayer(gnn_out, gnn_hidden, num_features)

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        gnn_encoded_steps = []
        for t in range(self.window_size):
            snapshot = x[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(batch_size,
                                                                               device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            gnn_out = self.gnn_encoder(snapshot, batch_edge_index)
            gnn_encoded_steps.append(gnn_out.reshape(batch_size, self.num_nodes, -1))

        gnn_encoded = torch.stack(gnn_encoded_steps, dim=1)
        gnn_encoded_flat = gnn_encoded.reshape(batch_size, self.window_size, -1)
        _, hidden_state = self.rnn_encoder(gnn_encoded_flat)
        decoder_input = hidden_state.permute(1, 0, 2).repeat(1, self.window_size, 1)
        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        rnn_decoded_unflat = rnn_decoded.reshape(batch_size, self.window_size, self.num_nodes, -1)
        reconstructed_steps = []
        for t in range(self.window_size):
            snapshot = rnn_decoded_unflat[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(batch_size,
                                                                               device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            reconstructed_snapshot = self.gnn_decoder(snapshot, batch_edge_index)
            reconstructed_steps.append(reconstructed_snapshot.reshape(batch_size, self.num_nodes, -1))
        return torch.stack(reconstructed_steps, dim=1)


# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO ---

def run_experiment(data_directory, output_dir, hp):
    # --- Carga de Datos ---
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    sensor_data = {i: [] for i in range(1, 6)}
    for f in all_files:
        try:
            sid = int(os.path.basename(f).split('_')[0])
            if sid in sensor_data:
                sensor_data[sid].append(pd.read_csv(f, sep='\s+', header=None, usecols=[1]).values)
        except:
            continue

    # --- División y Escalado ---
    train_files, test_files = {}, {}
    for sid, files in sensor_data.items():
        if not files: continue
        tv, t = train_test_split(files, test_size=0.15, shuffle=True, random_state=42)
        train_files[sid] = tv
        test_files[sid] = t

    scaler = StandardScaler()
    concatenated_train_data = np.concatenate([item for sublist in train_files.values() for item in sublist])
    scaler.fit(concatenated_train_data)

    # --- Creación de Datasets ---
    def create_spatio_temporal_dataset(files_dict, scaler, window_size, stride):
        if len(files_dict) != 5: return None
        scaled_dict = {}
        for sid in sorted(files_dict.keys()):
            if not files_dict[sid]: return None
            scaled_dict[sid] = scaler.transform(files_dict[sid][0])
        return SpatioTemporalWindowDataset(scaled_dict, window_size, stride)

    train_dataset = create_spatio_temporal_dataset(train_files, scaler, hp['window_size'], hp['stride'])
    test_dataset = create_spatio_temporal_dataset(test_files, scaler, hp['window_size'], hp['stride'])

    if not train_dataset or not test_dataset:
        print("No se pudieron crear los datasets.")
        return

    train_len = int(0.85 * len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])

    print(
        f"Ventanas de entrenamiento: {len(train_dataset)}, validación: {len(val_dataset)}, prueba: {len(test_dataset)}")

    # --- Bucle de Entrenamiento ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=num_workers)

    num_nodes = len(sensor_data)
    model = SpatioTemporalAutoencoder(num_nodes=num_nodes, num_features=1, window_size=hp['window_size']).to(device)
    edge_index = define_bridge_graph().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    history = {'train_loss': [], 'val_loss': []}

    print("\n--- Iniciando Entrenamiento del GNN Autoencoder ---")
    for epoch in range(hp['epochs']):
        model.train()
        avg_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Train]', leave=False)
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, edge_index)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        history['train_loss'].append(avg_train_loss / len(train_loader))

        model.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs, edge_index)
                loss = criterion(outputs, inputs)
                avg_val_loss += loss.item()
        history['val_loss'].append(avg_val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{hp['epochs']} -> Train Loss: {history['train_loss'][-1]:.6f}, Val Loss: {history['val_loss'][-1]:.6f}")
        scheduler.step(history['val_loss'][-1])
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"   -> Nuevo mejor modelo guardado.")
        else:
            patience_counter += 1
        if patience_counter >= hp['patience']:
            print("--- Parada Temprana ---")
            break

    print("--- Entrenamiento Finalizado ---")


    print("Guardando artefactos finales...")
    scaler_path = os.path.join(output_dir, 'scaler.gz')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler guardado en: {scaler_path}")

    hp_path = os.path.join(output_dir, 'hyperparameters.json')
    with open(hp_path, 'w') as f:
        json.dump(hp, f, indent=4)
    print(f"Hiperparámetros guardados en: {hp_path}")


#EJECUCIÓN#
if __name__ == '__main__':
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    HP = {"window_size": 64, "stride": 32, "epochs": 50, "batch_size": 32, "learning_rate": 0.001, "patience": 10}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = os.path.join("resultados_entrenamiento", f"run_gnn_{timestamp}")
    os.makedirs(output_directory, exist_ok=True)

    print(f"Los resultados se guardarán en: {output_directory}")
    if not os.path.isdir(data_folder_path):
        print("Error: Directorio no encontrado.")
    else:
        run_experiment(data_folder_path, output_directory, HP)

