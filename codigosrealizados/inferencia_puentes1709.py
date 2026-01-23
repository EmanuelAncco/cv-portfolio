import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- ARQUITECTURA DEL MODELO Y CLASES DE DATOS (COMO EN EL ENTRENAMIENTO) ---
from torch_geometric.nn import GCNConv


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
        if self.n_samples <= 0: self.n_samples = 0

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        return torch.FloatTensor(window), torch.FloatTensor(window)


class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden=32, gnn_out=16, rnn_hidden=64,
                 gnn_out_decoder=32):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes, self.window_size, self.gnn_out_decoder = num_nodes, window_size, gnn_out_decoder
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(input_size=gnn_out * num_nodes, hidden_size=rnn_hidden, batch_first=True,
                                  num_layers=2)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=gnn_out_decoder * num_nodes, batch_first=True,
                                  num_layers=2)
        self.gnn_decoder = GNNLayer(gnn_out_decoder, gnn_hidden, num_features)

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
        context_vector = hidden_state[-1]
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.window_size, 1)
        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        rnn_decoded_unflat = rnn_decoded.reshape(batch_size, self.window_size, self.num_nodes, self.gnn_out_decoder)
        reconstructed_steps = []
        for t in range(self.window_size):
            snapshot = rnn_decoded_unflat[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(batch_size,
                                                                               device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            reconstructed_snapshot = self.gnn_decoder(snapshot, batch_edge_index)
            reconstructed_steps.append(reconstructed_snapshot.reshape(batch_size, self.num_nodes, -1))
        return torch.stack(reconstructed_steps, dim=1)


def run_validation_and_visualization(training_run_dir, healthy_data_dir, damage_data_dir, output_plot_dir):
    print("--- Iniciando Proceso Completo de Validación y Visualización ---")
    os.makedirs(output_plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Cargar artefactos y modelo
    try:
        scaler = joblib.load(os.path.join(training_run_dir, 'scaler.gz'))
        with open(os.path.join(training_run_dir, 'hyperparameters.json'), 'r') as f:
            hp = json.load(f)
        model_path = os.path.join(training_run_dir, 'best_model.pth')
        num_nodes = 5
        model = SpatioTemporalAutoencoder(
            num_nodes=num_nodes, num_features=1, window_size=hp['window_size'],
            gnn_hidden=hp.get('gnn_hidden', 32), gnn_out=hp.get('gnn_out', 16),
            rnn_hidden=hp.get('rnn_hidden', 64), gnn_out_decoder=hp.get('gnn_out_decoder', 32)
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Modelo, scaler e hiperparámetros cargados.")
    except Exception as e:
        print(f"Error fatal al cargar artefactos: {e}");
        return

    edge_index = define_bridge_graph().to(device)
    criterion = nn.MSELoss(reduction='none')

    # 2. Análisis de datos SANOS (VALIDACIÓN)
    print("\nProcesando datos SANOS para establecer línea base...")
    healthy_files = [os.path.join(healthy_data_dir, f) for f in os.listdir(healthy_data_dir) if f.endswith('.txt')]
    np.random.shuffle(healthy_files)
    # Seleccionar 5 archivos al azar para representar los 5 sensores
    sample_healthy_files = healthy_files[:num_nodes]
    healthy_data = {i: pd.read_csv(f, sep='\s+', header=None, usecols=[1]).values for i, f in
                    enumerate(sample_healthy_files, 1)}
    scaled_healthy_data = {sid: scaler.transform(data) for sid, data in healthy_data.items()}
    healthy_dataset = SpatioTemporalWindowDataset(scaled_healthy_data, hp['window_size'], hp['stride'])
    healthy_loader = DataLoader(healthy_dataset, batch_size=hp['batch_size'], shuffle=True)

    healthy_errors, original_healthy_sample, reconstructed_healthy_sample = [], None, None
    with torch.no_grad():
        for i, (inputs, _) in enumerate(healthy_loader):
            inputs = inputs.to(device)
            outputs = model(inputs, edge_index)
            if i == 0:
                original_healthy_sample = inputs[0].cpu().numpy()
                reconstructed_healthy_sample = outputs[0].cpu().numpy()
            healthy_errors.append(criterion(outputs, inputs).cpu().numpy())

    healthy_errors = np.concatenate(healthy_errors, axis=0)
    healthy_error_per_window = np.mean(healthy_errors, axis=(1, 2, 3))
    print(f"Error de reconstrucción medio en datos sanos: {np.mean(healthy_error_per_window):.6f}")

    # Gráfico 0: Reconstrucción en Estado Sano
    fig, axes = plt.subplots(num_nodes, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Gráfico 0: Reconstrucción de Señal en Estado Sano (Muestra Aleatoria)', fontsize=16)
    for i in range(num_nodes):
        axes[i].plot(original_healthy_sample[:, i, 0], label='Señal Original (Sana)', color='green')
        axes[i].plot(reconstructed_healthy_sample[:, i, 0], label='Señal Reconstruida', color='black', linestyle=':')
        axes[i].set_title(f'Sensor {i + 1}');
        axes[i].set_ylabel('Aceleración Normalizada')
        axes[i].legend();
        axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[-1].set_xlabel('Paso de Tiempo en la Ventana')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(output_plot_dir, "0_reconstruccion_sana.png"));
    plt.close()
    print("Gráfico 0 guardado.")

    # 3. Análisis de datos CON DAÑO (INFERENCIA)
    print("\nProcesando datos CON DAÑO para detección...")
    damage_data = {}
    all_files_found = True
    for i in range(1, num_nodes + 1):
        # --- LÓGICA DE BÚSQUEDA DE ARCHIVOS MEJORADA ---
        possible_names = [f'{i}_Aceleraciones.txt', f'{i}_sismo.txt', f'{i}.txt']
        found_path = None
        for name in possible_names:
            path = os.path.join(damage_data_dir, name)
            if os.path.exists(path):
                found_path = path
                break

        if found_path:
            print(f"Archivo encontrado para sensor {i}: {os.path.basename(found_path)}")
            damage_data[i] = pd.read_csv(found_path, sep='\s+', header=None, usecols=[1]).values
        else:
            print(f"ERROR: No se encontró ningún archivo para el sensor {i} en '{damage_data_dir}'")
            all_files_found = False

    if not all_files_found:
        print("Proceso detenido. Faltan archivos de datos de daño.")
        return

    scaled_damage_data = {sid: scaler.transform(data) for sid, data in damage_data.items()}
    damage_dataset = SpatioTemporalWindowDataset(scaled_damage_data, hp['window_size'], hp['stride'])
    damage_loader = DataLoader(damage_dataset, batch_size=hp['batch_size'], shuffle=False)

    damage_errors, all_originals, all_reconstructed = [], [], []
    with torch.no_grad():
        for inputs, _ in tqdm(damage_loader, desc="Procesando datos de daño"):
            inputs = inputs.to(device)
            outputs = model(inputs, edge_index)
            all_originals.append(inputs.cpu().numpy());
            all_reconstructed.append(outputs.cpu().numpy())
            damage_errors.append(criterion(outputs, inputs).cpu().numpy())

    damage_errors = np.concatenate(damage_errors, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    mean_error_per_node = np.mean(damage_errors, axis=(0, 1, 3))
    damage_error_per_window = np.mean(damage_errors, axis=(1, 2, 3))

    # Gráfico 1: Comparación Dañada (Peor Caso)
    idx_max_error = np.argmax(damage_error_per_window)
    original_sample = all_originals[idx_max_error]
    reconstructed_sample = all_reconstructed[idx_max_error]
    fig, axes = plt.subplots(num_nodes, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Gráfico 1: Comparación Original vs. Reconstruida (Ventana con Mayor Error de Daño)', fontsize=16)
    for i in range(num_nodes):
        axes[i].plot(original_sample[:, i, 0], label='Señal Original (Dañada)', color='blue', alpha=0.8)
        axes[i].plot(reconstructed_sample[:, i, 0], label='Señal Reconstruida', color='red', linestyle='--')
        axes[i].set_title(f'Sensor {i + 1}');
        axes[i].set_ylabel('Aceleración Normalizada')
        axes[i].legend();
        axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[-1].set_xlabel('Paso de Tiempo en la Ventana')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(output_plot_dir, "1_comparacion_danada.png"));
    plt.close()
    print("Gráfico 1 guardado.")

    # Gráfico 2: Distribución de Errores con Escala Logarítmica
    plt.figure(figsize=(12, 7))
    sns.kdeplot(healthy_error_per_window[healthy_error_per_window > 0], label='Estado Sano (Datos de Validación)',
                color='green', fill=True, cut=0, log_scale=True)
    sns.kdeplot(damage_error_per_window[damage_error_per_window > 0], label='Estado con Daño (Detectado)', color='red',
                fill=True, cut=0, log_scale=True)
    plt.title('Gráfico 2: Distribución de Errores de Reconstrucción (Escala Logarítmica)', fontsize=16)
    plt.xlabel('Error Cuadrático Medio (MSE) por Ventana');
    plt.ylabel('Densidad')
    plt.legend();
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_plot_dir, "2_distribucion_errores_log.png"));
    plt.close()
    print("Gráfico 2 (Log) guardado.")

    # Gráfico 3: Mapa de Calor
    sensor_coords = {1: (1, 5), 2: (1, 1), 3: (3, 5), 4: (3, 1), 5: (5, 4)}
    x_coords = [sensor_coords[i + 1][0] for i in range(num_nodes)]
    y_coords = [sensor_coords[i + 1][1] for i in range(num_nodes)]
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_coords, y_coords, c=mean_error_per_node, cmap='coolwarm', s=2000, edgecolors='k', alpha=0.9)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f'S{i + 1}\nE: {mean_error_per_node[i]:.4f}', ha='center', va='center', color='white',
                 fontsize=10, weight='bold')
    plt.title('Gráfico 3: Mapa de Calor para Localización de Anomalías', fontsize=16)
    plt.xlabel('Eje X Esquemático');
    plt.ylabel('Eje Y Esquemático')
    cbar = plt.colorbar(scatter);
    cbar.set_label('Error de Reconstrucción Medio (MSE)')
    plt.grid(True, linestyle='--', alpha=0.5);
    plt.xlim(0, 6);
    plt.ylim(0, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_plot_dir, "3_heatmap_localizacion.png"));
    plt.close()
    print("Gráfico 3 guardado.")
    print("\n--- Proceso Finalizado ---")


if __name__ == '__main__':
    PATH_AL_MODELO_ENTRENADO = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756"
    PATH_A_DATOS_SANOS = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    PATH_A_DATOS_CON_DAÑO = r"D:\descargas 2025-2\articulo tesis delgadillo\Aceleraciones con daño\Aceleraciones"
    PATH_PARA_GUARDAR_GRAFICOS = "resultados_validacion_completos"

    run_validation_and_visualization(
        training_run_dir=PATH_AL_MODELO_ENTRENADO,
        healthy_data_dir=PATH_A_DATOS_SANOS,
        damage_data_dir=PATH_A_DATOS_CON_DAÑO,
        output_plot_dir=PATH_PARA_GUARDAR_GRAFICOS
    )

