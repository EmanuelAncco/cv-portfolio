# -*- coding: utf-8 -*-
"""
Script de simulación para el Autoencoder de Grafos Espacio-Temporal.
Versión 4.0 FINAL: Genera una visualización de dos columnas que muestra tanto la
reconstrucción de la señal como el error residual a lo largo del tiempo para
cada sensor, probando de forma concluyente el efecto de localización contextual.
Utiliza la topología del grafo proporcionada por el usuario.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Se necesita torch_geometric. Instalar con: pip install torch-geometric
from torch_geometric.nn import GCNConv

# --- CONFIGURACIÓN DE LA SIMULACIÓN ---
RUN_DIRECTORY = r"resultados_entrenamiento\run_gnn_20250910-020756"
DATA_FOLDER_HEALTHY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

# --- Parámetros de la simulación ---
TARGET_SENSOR_INDEX = 2  # Sensor 3 (los índices van de 0 a 4)
ANOMALY_MAGNITUDE = 3.5  # Magnitud sutil y realista para un daño incipiente

MODEL_PATH = os.path.join(RUN_DIRECTORY, "best_model.pth")
HYPERPARAMETERS_PATH = os.path.join(RUN_DIRECTORY, "hyperparameters.json")
OUTPUT_DIR = os.path.join(RUN_DIRECTORY, "simulacion_dano_localizado")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- DEFINICIONES DE CLASES (Copiadas para ser autocontenido) ---

def define_bridge_graph():
    """Define la conectividad del puente basada en el bosquejo del usuario."""
    edge_index = torch.tensor([
        # Conexiones bidireccionales
        [0, 2], [2, 0],  # Sensor 1 con 3
        [1, 3], [3, 1],  # Sensor 2 con 4
        [2, 3], [3, 2],  # Sensor 3 con 4
        [3, 4], [4, 3]  # Sensor 4 con 5
    ], dtype=torch.long).t().contiguous()
    return edge_index


class SpatioTemporalWindowDataset(Dataset):
    def __init__(self, data_dict, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride
        min_len = min(len(data) for data in data_dict.values() if len(data) > 0)
        valid_data = {sid: data for sid, data in data_dict.items() if len(data) >= min_len}
        self.data = np.stack([data[:min_len] for sid, data in sorted(valid_data.items())], axis=1)
        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples < 0: self.n_samples = 0

    def __len__(self): return self.n_samples

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
        # Corregido: Usar -1 para que PyTorch infiera la dimensión correcta dinámicamente
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
        # Corregido: Usar -1 para que PyTorch infiera la dimensión correcta dinámicamente
        rnn_decoded_flat = rnn_decoded.reshape(-1, gnn_hidden_dim)
        reconstructed_flat = self.gnn_decoder(rnn_decoded_flat, batch_edge_index)
        reconstruction = reconstructed_flat.reshape(batch_size, self.window_size, self.num_nodes, -1)
        return reconstruction


# --- FUNCIONES DE SIMULACIÓN ---

def get_healthy_test_window(healthy_path, hp):
    """Carga los datos sanos, ajusta el scaler y extrae una ventana de prueba."""
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

    scaled_healthy_test = {sid: scaler.transform(np.concatenate(test_files[sid])) for sid in sorted(test_files.keys())
                           if test_files[sid]}
    healthy_dataset = SpatioTemporalWindowDataset(scaled_healthy_test, hp['window_size'], hp['stride'])

    random_idx = np.random.randint(len(healthy_dataset))
    return healthy_dataset[random_idx]


def inject_anomaly(window_data, sensor_index, magnitude):
    """Inyecta una anomalía de tipo 'spike' en un sensor específico."""
    damaged_window = window_data.clone()
    spike_position = damaged_window.shape[0] // 2
    damaged_window[spike_position, sensor_index, 0] += magnitude
    return damaged_window


def plot_contextual_effect_analysis(damaged_window, model, device, output_path):
    """
    Visualiza el efecto dominó contextual mostrando la reconstrucción y el error residual por sensor.
    """
    model.eval()
    edge_index = define_bridge_graph().to(device)
    with torch.no_grad():
        damaged_input = damaged_window.unsqueeze(0).to(device)
        reconstruction = model(damaged_input, edge_index).squeeze(0).cpu()

    # Calcula el error residual
    residual_error = damaged_window - reconstruction

    fig, axes = plt.subplots(5, 2, figsize=(18, 22), sharex=True)
    fig.suptitle('Análisis del Efecto Contextual de la GNN en la Detección de Anomalías', fontsize=22, y=0.96)

    time_steps = np.arange(damaged_window.shape[0])
    num_nodes = damaged_window.shape[1]

    # Encuentra los límites del error para una escala Y consistente
    max_error = np.max(np.abs(residual_error.numpy())) * 1.1

    for i in range(num_nodes):
        # --- Columna Izquierda: Reconstrucción ---
        ax_recon = axes[i, 0]
        title_suffix = ""
        if i == TARGET_SENSOR_INDEX:
            title_suffix = " (Epicentro)"
        elif abs(i - TARGET_SENSOR_INDEX) == 1:  # Asumiendo vecinos directos en el grafo
            title_suffix = " (Vecino)"

        ax_recon.set_title(f"Sensor {i + 1}{title_suffix}: Señal vs. Reconstrucción", fontsize=14)
        ax_recon.plot(time_steps, damaged_window[:, i, 0].numpy(), label='Entrada', color='dodgerblue', linewidth=2)
        ax_recon.plot(time_steps, reconstruction[:, i, 0].numpy(), label='Reconstrucción', color='darkorange',
                      linestyle='--', linewidth=2)
        ax_recon.set_ylabel('Valor Normalizado')
        ax_recon.legend()
        ax_recon.grid(True, linestyle='--', alpha=0.6)

        # --- Columna Derecha: Error Residual ---
        ax_error = axes[i, 1]
        ax_error.set_title(f"Sensor {i + 1}: Error de Reconstrucción (Residual)", fontsize=14)
        ax_error.plot(time_steps, residual_error[:, i, 0].numpy(), label='Error (Entrada - Reconstrucción)',
                      color='crimson', linewidth=2)
        ax_error.set_ylabel('Error')
        ax_error.set_ylim(-max_error, max_error)  # Escala Y consistente para comparar
        ax_error.legend()
        ax_error.grid(True, linestyle='--', alpha=0.6)

    axes[-1, 0].set_xlabel('Paso de Tiempo')
    axes[-1, 1].set_xlabel('Paso de Tiempo')
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico de análisis contextual guardado en: {output_path}")


# --- EJECUCIÓN DEL SCRIPT DE SIMULACIÓN ---
if __name__ == '__main__':
    print("--- Iniciando Simulación Detallada de Daño Localizado ---")

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

    print("Preparando ventana de datos sanos para la simulación...")
    healthy_window = get_healthy_test_window(DATA_FOLDER_HEALTHY, hp)

    print(f"Inyectando anomalía sutil en el Sensor {TARGET_SENSOR_INDEX + 1}...")
    damaged_window = inject_anomaly(healthy_window, TARGET_SENSOR_INDEX, ANOMALY_MAGNITUDE)

    print("Generando visualización detallada del efecto contextual...")
    simulation_plot_path = os.path.join(OUTPUT_DIR, f'sim_contextual_effect_at_sensor_{TARGET_SENSOR_INDEX + 1}.png')
    plot_contextual_effect_analysis(damaged_window, model, device, simulation_plot_path)

    print("\n--- Simulación Finalizada ---")

