# -*- coding: utf-8 -*-
"""
Script de evaluación para el Autoencoder de Grafos Espacio-Temporal.
Versión 3.0: Añadido análisis estadístico sobre todo el dataset de prueba para
generar un mapa de localización de daño. Se crean gráficos de cajas (boxplots)
y un factor de amplificación de daño para una visualización científica robusta.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Se necesita torch_geometric. Instalar con: pip install torch-geometric
from torch_geometric.nn import GCNConv

# --- CONFIGURACIÓN ---
RUN_DIRECTORY = r"resultados_entrenamiento\run_gnn_20250910-020756"
DATA_FOLDER_HEALTHY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
DATA_FOLDER_DAMAGED = r"C:\Users\Emanuel\Downloads\articulo tesis delgadillo\Aceleraciones con daño\Aceleraciones"

MODEL_PATH = os.path.join(RUN_DIRECTORY, "best_model.pth")
HYPERPARAMETERS_PATH = os.path.join(RUN_DIRECTORY, "hyperparameters.json")
OUTPUT_DIR = os.path.join(RUN_DIRECTORY, "evaluacion_sismo_real")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- DEFINICIONES DE CLASES (Copiadas para ser autocontenido) ---

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
        min_len = min(len(data) for data in data_dict.values() if len(data) > 0)
        valid_data = {sid: data for sid, data in data_dict.items() if len(data) >= min_len}
        if len(valid_data) != len(data_dict):
            print(f"Advertencia: Algunos sensores tenían datos insuficientes y fueron ignorados.")
        self.data = np.stack([data[:min_len] for sid, data in sorted(valid_data.items())], axis=1)
        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples < 0: self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        return torch.FloatTensor(self.data[start:end])


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
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=gnn_hidden * num_nodes, batch_first=True,
                                  num_layers=2)
        self.gnn_decoder = GNNLayer(gnn_hidden, gnn_hidden, num_features)

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        num_features = x.size(3)
        gnn_hidden_dim = self.gnn_decoder.conv1.in_channels
        x_flat = x.reshape(-1, num_features)
        num_graphs_in_batch = batch_size * self.window_size
        batch_edge_index = edge_index.repeat(1, num_graphs_in_batch) + torch.arange(num_graphs_in_batch,
                                                                                    device=x.device).repeat_interleave(
            edge_index.size(1)) * self.num_nodes
        gnn_out = self.gnn_encoder(x_flat, batch_edge_index)
        gnn_encoded_flat = gnn_out.reshape(batch_size, self.window_size, -1)
        _, hidden_state = self.rnn_encoder(gnn_encoded_flat)
        context_vector = hidden_state[-1].unsqueeze(1)
        decoder_input = context_vector.repeat(1, self.window_size, 1)
        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        rnn_decoded_flat = rnn_decoded.reshape(-1, gnn_hidden_dim)
        reconstructed_flat = self.gnn_decoder(rnn_decoded_flat, batch_edge_index)
        reconstruction = reconstructed_flat.reshape(batch_size, self.window_size, self.num_nodes, -1)
        return reconstruction


# --- FUNCIONES DE EVALUACIÓN ---

def prepare_evaluation_environment(healthy_path, damaged_path, hp):
    print("Cargando datos SANOS para ajustar el scaler...")
    all_healthy_files = [os.path.join(healthy_path, f) for f in os.listdir(healthy_path) if f.endswith('.txt')]
    sensor_data_healthy = {i: [] for i in range(1, 6)}
    for f in all_healthy_files:
        try:
            sid = int(os.path.basename(f).split('_')[0])
            if sid in sensor_data_healthy:
                data = pd.read_csv(f, sep='\s+', header=None, usecols=[1], engine='python').values
                if data.size > 0: sensor_data_healthy[sid].append(data)
        except Exception:
            continue
    train_files, test_files = {}, {}
    for sid, files in sensor_data_healthy.items():
        if not files: continue
        tv, t = train_test_split(files, test_size=0.15, shuffle=True, random_state=42)
        train_files[sid], test_files[sid] = tv, t
    scaler = StandardScaler()
    concatenated_train_data = np.concatenate([item for sublist in train_files.values() for item in sublist])
    scaler.fit(concatenated_train_data)
    print("Scaler ajustado con datos de entrenamiento sanos.")
    print("Creando dataset de prueba con datos SANOS...")
    scaled_healthy_test = {sid: scaler.transform(np.concatenate(test_files[sid])) for sid in sorted(test_files.keys())
                           if test_files[sid]}
    healthy_dataset = SpatioTemporalWindowDataset(scaled_healthy_test, hp['window_size'], hp['stride'])
    print(f"Cargando datos CON DAÑO desde: {damaged_path}")
    all_damaged_files = [os.path.join(damaged_path, f) for f in os.listdir(damaged_path) if f.endswith('.txt')]
    sensor_data_damaged = {i: [] for i in range(1, 6)}
    for f in all_damaged_files:
        try:
            sid = int(os.path.basename(f).split('_')[0])
            if sid in sensor_data_damaged:
                data = pd.read_csv(f, sep='\s+', header=None, usecols=[0], engine='python').values
                if data.size > 0: sensor_data_damaged[sid].append(data)
        except Exception as e:
            print(f"No se pudo leer {f}: {e}"); continue
    print("Creando dataset de prueba con datos CON DAÑO...")
    scaled_damaged_test = {sid: scaler.transform(np.concatenate(sensor_data_damaged[sid])) for sid in
                           sorted(sensor_data_damaged.keys()) if sensor_data_damaged[sid]}
    damaged_dataset = SpatioTemporalWindowDataset(scaled_damaged_test, hp['window_size'], hp['stride'])
    return healthy_dataset, damaged_dataset


def analyze_full_dataset(model, dataset, device):
    """Itera sobre todo un dataset y calcula el error de reconstrucción para cada ventana."""
    model.eval()
    all_errors = []
    edge_index = define_bridge_graph().to(device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analizando dataset"):
            inputs = batch.to(device)
            reconstruction = model(inputs, edge_index)
            error = torch.mean((inputs - reconstruction) ** 2, dim=(1, 3))  # Error por ventana
            all_errors.append(error.cpu().numpy())

    return np.vstack(all_errors)


def plot_damage_localization_map(healthy_errors, damaged_errors, output_path):
    """
    Genera un gráfico de cajas y un factor de amplificación para localizar el daño.
    """
    num_nodes = healthy_errors.shape[1]
    node_labels = [f"Sensor {i + 1}" for i in range(num_nodes)]

    healthy_median_error = np.median(healthy_errors, axis=0)
    damaged_median_error = np.median(damaged_errors, axis=0)

    # Evitar división por cero si el error sano es muy bajo
    amplification_factor = damaged_median_error / (healthy_median_error + 1e-9)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Análisis de Localización de Daño por Sensor', fontsize=18)

    # Gráfico de cajas (Box Plot)
    bplot = ax1.boxplot([damaged_errors[:, i] for i in range(num_nodes)],
                        vert=True, patch_artist=True, labels=node_labels)
    ax1.set_yscale('log')
    ax1.set_title('Distribución del Error de Reconstrucción (Datos con Daño)')
    ax1.set_ylabel('Error Cuadrático Medio (MSE) - Escala Log')
    ax1.grid(True, linestyle='--', alpha=0.6)
    for patch in bplot['boxes']:
        patch.set_facecolor('orangered')

    # Gráfico de Factor de Amplificación
    ax2.bar(node_labels, amplification_factor, color='crimson')
    ax2.set_title('Factor de Amplificación del Error Mediano (Daño vs. Sano)')
    ax2.set_ylabel('Error con Daño / Error Sano')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"Mapa de localización de daño guardado en: {output_path}")


# --- EJECUCIÓN DEL SCRIPT DE EVALUACIÓN ---
if __name__ == '__main__':
    print("--- Iniciando Script de Evaluación con Análisis de Localización ---")

    with open(HYPERPARAMETERS_PATH, 'r') as f:
        hp = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = 5
    model = SpatioTemporalAutoencoder(
        num_nodes=num_nodes, num_features=1, window_size=hp['window_size'],
        gnn_hidden=hp['gnn_hidden'], gnn_out=hp['gnn_out'], rnn_hidden=hp['rnn_hidden']
    ).to(device)

    print(f"Cargando pesos del modelo desde: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    healthy_dataset, damaged_dataset = prepare_evaluation_environment(DATA_FOLDER_HEALTHY, DATA_FOLDER_DAMAGED, hp)

    if not healthy_dataset or not damaged_dataset or len(healthy_dataset) == 0 or len(damaged_dataset) == 0:
        print("Error: No se pudieron crear los datasets. Verifica las rutas y los archivos de datos.")
        exit()

    print("\n--- Analizando la totalidad de los datasets para localización ---")
    healthy_errors_full = analyze_full_dataset(model, healthy_dataset, device)
    damaged_errors_full = analyze_full_dataset(model, damaged_dataset, device)

    # --- Generación de Gráfico de Localización ---
    localization_map_path = os.path.join(OUTPUT_DIR, 'eval_damage_localization_map.png')
    plot_damage_localization_map(healthy_errors_full, damaged_errors_full, localization_map_path)

    print("\n--- Script de Evaluación y Localización Finalizado ---")

