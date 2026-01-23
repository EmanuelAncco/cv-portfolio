# -*- coding: utf-8 -*-
"""
Script de evaluación para el Autoencoder de Grafos Espacio-Temporal.
Versión 1.1: Corregido el NameError al importar la clase `Dataset` de torch.utils.data.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset  # <-- CORRECCIÓN AÑADIDA AQUÍ
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import logging

# Se necesita torch_geometric. Instalar con: pip install torch-geometric
from torch_geometric.nn import GCNConv

# --- CONFIGURACIÓN ---
# Asegúrate de que estas rutas apunten a la carpeta de resultados de tu mejor entrenamiento
RUN_DIRECTORY = r"resultados_entrenamiento\run_gnn_20250910-020756"
MODEL_PATH = os.path.join(RUN_DIRECTORY, "best_model.pth")
HYPERPARAMETERS_PATH = os.path.join(RUN_DIRECTORY, "hyperparameters.json")
DATA_FOLDER_PATH = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
OUTPUT_DIR = os.path.join(RUN_DIRECTORY, "evaluacion")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- DEFINICIONES DE CLASES (Copiadas de puentessen_final.py para ser autocontenido) ---

def define_bridge_graph():
    """Define la conectividad del puente basada en la disposición física de los sensores."""
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    return edge_index


class SpatioTemporalWindowDataset(Dataset):
    """Crea ventanas espacio-temporales a partir de datos de múltiples sensores."""

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
        return torch.FloatTensor(window)


class GNNLayer(nn.Module):
    """Capa GNN reutilizable para codificador y decodificador."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


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
        batch_edge_index = edge_index.repeat(1, num_graphs_in_batch) + torch.arange(
            num_graphs_in_batch, device=x.device
        ).repeat_interleave(edge_index.size(1)) * self.num_nodes
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

def load_data_and_scaler(data_path, hp):
    """
    Carga los datos y, crucialmente, re-crea y ajusta el scaler
    exactamente como se hizo en el entrenamiento original.
    """
    print("Cargando datos para ajustar el scaler y crear el dataset de prueba...")
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    sensor_data = {i: [] for i in range(1, 6)}
    for f in all_files:
        try:
            sid = int(os.path.basename(f).split('_')[0])
            if sid in sensor_data:
                data = pd.read_csv(f, sep='\s+', header=None, usecols=[1], engine='python').values
                if data.size > 0:
                    sensor_data[sid].append(data)
        except Exception:
            continue

    train_files, test_files = {}, {}
    for sid, files in sensor_data.items():
        if not files: continue
        tv, t = train_test_split(files, test_size=0.15, shuffle=True, random_state=42)
        train_files[sid] = tv
        test_files[sid] = t

    # 1. Re-crear el scaler
    concatenated_train_data = np.concatenate([item for sublist in train_files.values() for item in sublist])
    scaler = StandardScaler()
    scaler.fit(concatenated_train_data)

    # 2. Crear el dataset de prueba con el scaler ajustado
    scaled_dict = {}
    for sid in sorted(test_files.keys()):
        if not test_files[sid]: return None, None
        concatenated_sensor_data = np.concatenate(test_files[sid])
        scaled_dict[sid] = scaler.transform(concatenated_sensor_data)

    test_dataset = SpatioTemporalWindowDataset(scaled_dict, hp['window_size'], hp['stride'])
    return test_dataset, scaler


def inject_anomaly(window_tensor, sensor_index, magnitude=2.0, anomaly_type='spike'):
    """
    Inyecta una anomalía en una ventana de datos.
    - window_tensor: Tensor de shape (window_size, num_nodes, num_features)
    - sensor_index: El índice del sensor a afectar (0 a 4).
    - magnitude: Cuán fuerte es la anomalía.
    - anomaly_type: 'spike' (un pulso corto) o 'noise' (ruido añadido).
    """
    anomalous_window = window_tensor.clone()
    window_size = anomalous_window.shape[0]

    if anomaly_type == 'spike':
        # Añade un pico agudo en el medio de la ventana temporal
        spike_start = window_size // 2
        spike_end = spike_start + 5  # Un pico de 5 pasos de tiempo
        anomalous_window[spike_start:spike_end, sensor_index, :] += magnitude
    elif anomaly_type == 'noise':
        # Añade ruido gaussiano a lo largo de toda la ventana para ese sensor
        noise = torch.randn(window_size, 1) * magnitude
        anomalous_window[:, sensor_index, :] += noise

    return anomalous_window


def get_reconstruction_error(model, data_window, device):
    """Pasa una única ventana por el modelo y calcula el error por nodo."""
    model.eval()
    with torch.no_grad():
        # Añadir una dimensión de batch (batch_size=1)
        inputs = data_window.unsqueeze(0).to(device)
        edge_index = define_bridge_graph().to(device)
        reconstruction = model(inputs, edge_index)

        # Calcular error MSE por nodo
        error_per_node = torch.mean((inputs - reconstruction) ** 2, dim=(0, 1, 3))
        return error_per_node.detach().cpu().numpy()


def plot_comparison(normal_error, anomalous_error, sensor_index, output_path):
    """Genera y guarda un gráfico de barras comparando los errores."""
    num_nodes = len(normal_error)
    node_labels = [f"Sensor {i + 1}" for i in range(num_nodes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle('Comparación de Error de Reconstrucción: Normal vs. Anómalo', fontsize=16)

    # Gráfico de datos normales
    ax1.bar(node_labels, normal_error, color='skyblue')
    ax1.set_title('Ventana de Datos Normal')
    ax1.set_ylabel('Error Cuadrático Medio (MSE)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Gráfico de datos anómalos
    colors = ['orangered' if i == sensor_index else 'skyblue' for i in range(num_nodes)]
    ax2.bar(node_labels, anomalous_error, color=colors)
    ax2.set_title(f'Ventana con Anomalía en Sensor {sensor_index + 1}')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico de comparación guardado en: {output_path}")


# --- EJECUCIÓN DEL SCRIPT DE EVALUACIÓN ---
if __name__ == '__main__':
    print("--- Iniciando Script de Evaluación ---")

    # 1. Cargar Hiperparámetros
    print(f"Cargando hiperparámetros desde: {HYPERPARAMETERS_PATH}")
    with open(HYPERPARAMETERS_PATH, 'r') as f:
        hp = json.load(f)

    # 2. Configurar dispositivo y cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    num_nodes = 5  # Asumimos 5 sensores como en el entrenamiento
    model = SpatioTemporalAutoencoder(
        num_nodes=num_nodes, num_features=1, window_size=hp['window_size'],
        gnn_hidden=hp['gnn_hidden'], gnn_out=hp['gnn_out'], rnn_hidden=hp['rnn_hidden']
    ).to(device)

    print(f"Cargando pesos del modelo desde: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    # 3. Cargar datos de prueba y el scaler
    test_dataset, scaler = load_data_and_scaler(DATA_FOLDER_PATH, hp)
    if not test_dataset:
        print("Error: No se pudo crear el dataset de prueba.")
        exit()

    # 4. Realizar la prueba de inyección de anomalía
    print("\n--- Realizando Prueba de Inyección de Anomalía ---")
    # Seleccionar una ventana de prueba al azar
    sample_idx = np.random.randint(len(test_dataset))
    normal_window = test_dataset[sample_idx]

    # Sensor al que se le inyectará la anomalía (0-4)
    sensor_to_affect = 3
    # Inyectar una anomalía de tipo 'pico' con una magnitud de 3.0
    anomalous_window = inject_anomaly(normal_window, sensor_index=sensor_to_affect, magnitude=3.0, anomaly_type='spike')

    # 5. Calcular errores de reconstrucción
    error_normal = get_reconstruction_error(model, normal_window, device)
    error_anomalous = get_reconstruction_error(model, anomalous_window, device)

    print("\nResultados de la Detección:")
    print("  Vector de Error (Normal):")
    for i, err in enumerate(error_normal):
        print(f"    - Sensor {i + 1}: {err:.6f}")

    print("\n  Vector de Error (Anómalo):")
    for i, err in enumerate(error_anomalous):
        highlight = "<--- ANOMALÍA DETECTADA" if i == sensor_to_affect else ""
        print(f"    - Sensor {i + 1}: {err:.6f} {highlight}")

    # 6. Visualizar y guardar resultados
    output_plot_path = os.path.join(OUTPUT_DIR, f'eval_sensor_{sensor_to_affect + 1}.png')
    plot_comparison(error_normal, error_anomalous, sensor_to_affect, output_plot_path)

    print("\n--- Script de Evaluación Finalizado ---")

