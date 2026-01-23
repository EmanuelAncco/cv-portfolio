# -*- coding: utf-8 -*-
"""
Script de entrenamiento de un Autoencoder de Grafos Espacio-Temporal.
Versión 9.0: Corregido el RuntimeError de multiplicación de matrices ajustando
la arquitectura del decodificador para que sea simétrica con el codificador
en términos de dimensiones de características.
"""
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
import logging

# Se necesita torch_geometric. Instalar con: pip install torch-geometric
from torch_geometric.nn import GCNConv


# --- CONFIGURACIÓN DE LOGGING ---
def setup_logging(log_path):
    """Configura el logging para guardar en archivo y mostrar en consola."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


# --- ESTRUCTURA DEL GRAFO Y LÓGICA DE DATOS ---

def define_bridge_graph():
    """Define la conectividad del puente basada en la disposición física de los sensores."""
    # Grafo que conecta nodos adyacentes
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
        # Asegura que todos los sensores tengan la misma longitud para el apilado
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


# --- ARQUITECTURA DEL GNN AUTOENCODER (CORREGIDA) ---

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

        # Codificador (Sin cambios)
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(input_size=gnn_out * num_nodes, hidden_size=rnn_hidden, batch_first=True,
                                  num_layers=2)

        # --- Decodificador (ARQUITECTURA CORREGIDA) ---
        # 1. El RNN Decoder ahora produce la dimensión oculta del GNN (gnn_hidden) por cada nodo.
        #    Esto asegura que la entrada al GNN Decoder tenga las dimensiones correctas.
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=gnn_hidden * num_nodes, batch_first=True,
                                  num_layers=2)
        # 2. El GNN Decoder ahora acepta la dimensión oculta (gnn_hidden) como entrada para iniciar la reconstrucción.
        #    Se usa gnn_hidden como capa oculta interna también por simplicidad.
        self.gnn_decoder = GNNLayer(gnn_hidden, gnn_hidden, num_features)

    def forward(self, x, edge_index):
        batch_size = x.size(0)

        # --- Codificador ---
        gnn_encoded_steps = []
        for t in range(self.window_size):
            # Procesa cada 'snapshot' temporal del grafo
            snapshot = x[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            # Crea un `edge_index` para todo el batch de grafos
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(edge_index.size(1)) * self.num_nodes

            gnn_out = self.gnn_encoder(snapshot, batch_edge_index)
            gnn_encoded_steps.append(gnn_out.reshape(batch_size, self.num_nodes, -1))

        gnn_encoded = torch.stack(gnn_encoded_steps, dim=1)
        gnn_encoded_flat = gnn_encoded.reshape(batch_size, self.window_size, -1)
        _, hidden_state = self.rnn_encoder(gnn_encoded_flat)

        # --- Decodificador ---
        # Usamos el último estado oculto del encoder como el 'vector de pensamiento' o contexto inicial.
        # Lo expandimos para alimentar cada paso de tiempo del decodificador.
        context_vector = hidden_state[-1].unsqueeze(1)  # Tomamos solo el de la última capa
        decoder_input = context_vector.repeat(1, self.window_size, 1)

        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        rnn_decoded_unflat = rnn_decoded.reshape(batch_size, self.window_size, self.num_nodes, -1)

        reconstructed_steps = []
        for t in range(self.window_size):
            snapshot = rnn_decoded_unflat[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            # Reutilizamos el mismo batch_edge_index
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(
                batch_size, device=x.device
            ).repeat_interleave(edge_index.size(1)) * self.num_nodes

            reconstructed_snapshot = self.gnn_decoder(snapshot, batch_edge_index)
            reconstructed_steps.append(reconstructed_snapshot.reshape(batch_size, self.num_nodes, -1))

        reconstruction = torch.stack(reconstructed_steps, dim=1)
        return reconstruction


# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO ---

def run_experiment(data_directory, output_dir, hp):
    log_file = os.path.join(output_dir, 'training_log.txt')
    setup_logging(log_file)

    # Guardar hiperparámetros
    with open(os.path.join(output_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hp, f, indent=4)
    logging.info(f"Hiperparámetros guardados en {output_dir}")

    logging.info("Cargando y preprocesando datos...")
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    sensor_data = {i: [] for i in range(1, 6)}
    for f in all_files:
        try:
            sid = int(os.path.basename(f).split('_')[0])
            if sid in sensor_data:
                # Lee solo la segunda columna (índice 1)
                data = pd.read_csv(f, sep='\s+', header=None, usecols=[1], engine='python').values
                if data.size > 0:
                    sensor_data[sid].append(data)
        except Exception as e:
            logging.warning(f"No se pudo procesar el archivo {f}: {e}")
            continue

    train_files, test_files = {}, {}
    for sid, files in sensor_data.items():
        if not files:
            logging.warning(f"No se encontraron datos para el sensor {sid}.")
            continue
        tv, t = train_test_split(files, test_size=0.15, shuffle=True, random_state=42)
        train_files[sid] = tv
        test_files[sid] = t

    logging.info("Ajustando el escalador con los datos de entrenamiento...")
    concatenated_train_data = np.concatenate([item for sublist in train_files.values() for item in sublist])
    scaler = StandardScaler()
    scaler.fit(concatenated_train_data)

    def create_spatio_temporal_dataset(files_dict, scaler, window_size, stride):
        if len(files_dict) != 5: return None
        scaled_dict = {}
        # Usar todos los archivos disponibles para cada sensor
        for sid in sorted(files_dict.keys()):
            if not files_dict[sid]: return None
            concatenated_sensor_data = np.concatenate(files_dict[sid])
            scaled_dict[sid] = scaler.transform(concatenated_sensor_data)
        return SpatioTemporalWindowDataset(scaled_dict, window_size, stride)

    logging.info("Creando datasets y dataloaders...")
    train_dataset = create_spatio_temporal_dataset(train_files, scaler, hp['window_size'], hp['stride'])
    test_dataset = create_spatio_temporal_dataset(test_files, scaler, hp['window_size'], hp['stride'])

    if not train_dataset or not test_dataset or len(train_dataset) == 0:
        logging.error("No se pudieron crear los datasets o están vacíos. Revisa los datos de entrada.")
        return

    train_len = int(0.85 * len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])

    logging.info(
        f"Tamaños -> Entrenamiento: {len(train_dataset)}, Validación: {len(val_dataset)}, Prueba: {len(test_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")
    num_workers = 0 if os.name == 'nt' else 4
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=num_workers)

    num_nodes = len(sensor_data)
    model = SpatioTemporalAutoencoder(
        num_nodes=num_nodes, num_features=1, window_size=hp['window_size'],
        gnn_hidden=hp['gnn_hidden'], gnn_out=hp['gnn_out'], rnn_hidden=hp['rnn_hidden']
    ).to(device)
    edge_index = define_bridge_graph().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    history = {'train_loss': [], 'val_loss': []}

    logging.info("--- Iniciando Entrenamiento del Autoencoder ---")
    for epoch in range(hp['epochs']):
        model.train()
        avg_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Train]', leave=False)
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)  # Aunque sean iguales, es buena práctica
            optimizer.zero_grad()
            outputs = model(inputs, edge_index)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        history['train_loss'].append(avg_train_loss / len(train_loader))

        model.eval()
        avg_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs, edge_index)
                loss = criterion(outputs, targets)
                avg_val_loss += loss.item()
        history['val_loss'].append(avg_val_loss / len(val_loader))

        logging.info(
            f"Epoch {epoch + 1}/{hp['epochs']} -> Train Loss: {history['train_loss'][-1]:.6f}, Val Loss: {history['val_loss'][-1]:.6f}")
        scheduler.step(history['val_loss'][-1])
        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            logging.info(f"   -> Nuevo mejor modelo guardado con Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
        if patience_counter >= hp['patience']:
            logging.info("--- Parada Temprana (Early Stopping) ---")
            break

    logging.info("--- Entrenamiento Finalizado ---")

    # Guardar curvas de aprendizaje
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Curvas de Aprendizaje')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

    logging.info("\n--- Simulación de Localización con Datos de Prueba ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    if len(test_dataset) > 0:
        inputs, _ = test_dataset[np.random.randint(len(test_dataset))]
        inputs = inputs.unsqueeze(0).to(device)

        reconstruction = model(inputs, edge_index)

        error_per_node = torch.mean((inputs - reconstruction) ** 2, dim=(0, 1, 3))
        error_per_node = error_per_node.detach().cpu().numpy()

        logging.info("\nVector de Error de Reconstrucción por Nodo:")
        for i, err in enumerate(error_per_node):
            logging.info(f"  - Sensor {i + 1}: Error = {err:.6f}")

        plt.figure(figsize=(10, 6))
        plt.bar([f"Sensor {i + 1}" for i in range(num_nodes)], error_per_node, color='cyan')
        plt.title("Error de Reconstrucción por Nodo (Localización de Anomalía)")
        plt.ylabel("Error Cuadrático Medio (MSE)")
        plt.savefig(os.path.join(output_dir, 'localization_vector.png'))
        # plt.show() # Descomentar si se ejecuta interactivamente
        plt.close()
    else:
        logging.warning("El dataset de prueba está vacío, no se puede realizar la simulación.")


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == '__main__':
    # Validar que la ruta de datos exista
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    if not os.path.isdir(data_folder_path):
        print(f"Error Crítico: El directorio de datos '{data_folder_path}' no existe.")
    else:
        # Definir todos los hiperparámetros en un solo lugar
        HP = {
            "window_size": 64,
            "stride": 32,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": 10,
            "gnn_hidden": 32,
            "gnn_out": 16,
            "rnn_hidden": 64
        }

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_directory = os.path.join("resultados_entrenamiento", f"run_gnn_{timestamp}")
        os.makedirs(output_directory, exist_ok=True)

        print(f"Los resultados se guardarán en: {output_directory}")
        run_experiment(data_folder_path, output_directory, HP)
