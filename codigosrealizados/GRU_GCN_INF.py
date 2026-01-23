import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv


# --- CONFIGURACIÓN DE LOGGING ---
def setup_logging(log_dir):
    """Configura el logging para guardar en archivo y mostrar en consola."""
    log_filename = os.path.join(log_dir, 'inference_log.log')
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )


# --- ARQUITECTURA DEL MODELO (Idéntica al entrenamiento) ---
class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden, gnn_out, rnn_hidden):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size

        # Las dimensiones de las capas se definen exclusivamente por los hiperparámetros de entrada
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
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(
                batch_size, device=x.device).repeat_interleave(edge_index.size(1)) * self.num_nodes

            gnn_out_val = self.gnn_encoder(snapshot, batch_edge_index)
            gnn_encoded_steps.append(gnn_out_val.reshape(batch_size, self.num_nodes, -1))

        gnn_encoded = torch.stack(gnn_encoded_steps, dim=1)
        gnn_encoded_flat = gnn_encoded.view(batch_size, self.window_size, -1)
        _, hidden_state = self.rnn_encoder(gnn_encoded_flat)

        decoder_input = hidden_state[-1].unsqueeze(1).repeat(1, self.window_size, 1)

        rnn_decoded, _ = self.rnn_decoder(decoder_input)

        gnn_out_channels = self.gnn_encoder.conv2.out_channels
        gnn_in_decoder = rnn_decoded.view(batch_size, self.window_size, self.num_nodes, gnn_out_channels)

        reconstructed_steps = []
        for t in range(self.window_size):
            snapshot = gnn_in_decoder[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(
                batch_size, device=x.device).repeat_interleave(edge_index.size(1)) * self.num_nodes

            reconstructed_snapshot = self.gnn_decoder(snapshot, batch_edge_index)
            reconstructed_steps.append(reconstructed_snapshot.reshape(batch_size, self.num_nodes, -1))

        return torch.stack(reconstructed_steps, dim=1)


# --- DATASET Y GRAFO ---
class InferenceDataset(Dataset):
    def __init__(self, data_windows):
        self.features = data_windows.astype(np.float32)
        logging.info(f"Dataset de inferencia cargado con {len(self.features)} muestras.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx])


def define_bridge_graph():
    edge_index = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]],
        dtype=torch.long).t().contiguous()
    return edge_index


# ...(El resto de tus funciones de ayuda como `load_damaged_data`, `plot_reconstruction`, etc., no necesitan cambios)...
# --- FUNCIONES PARA MANEJAR DATOS REALES DE DAÑO ---
def create_windows_from_array(data_array, window_size, stride):
    n_samples = (data_array.shape[0] - window_size) // stride + 1
    if n_samples <= 0:
        return np.array([])
    windows = np.array([data_array[i * stride: i * stride + window_size] for i in range(n_samples)])
    return np.transpose(windows, (0, 2, 1))


def load_damaged_data(data_dir, scaler, hp):
    logging.info(f"Cargando datos con daño desde: {data_dir}")
    damaged_data_dict = {}
    for sensor_id in range(1, 6):
        file_path = os.path.join(data_dir, f'{sensor_id}_sismo.txt')
        if not os.path.exists(file_path):
            logging.error(f"No se encontró el archivo de daño: {file_path}")
            return None
        signal = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python', usecols=[1]).values
        damaged_data_dict[sensor_id] = signal

    try:
        min_len = min(len(data) for data in damaged_data_dict.values())
        data_stack = np.hstack([data[:min_len] for sid, data in sorted(damaged_data_dict.items())])
    except ValueError:
        logging.error("No se pudieron cargar los datos de daño para todos los sensores.")
        return None

    scaled_data = scaler.transform(data_stack)
    windows = create_windows_from_array(scaled_data, hp['window_size'], hp['stride'])
    logging.info(f"Se generaron {len(windows)} ventanas a partir de los datos de daño.")
    return windows


# --- FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN ---
def get_reconstruction_errors(model, dataloader, device):
    model.eval()
    all_errors = []
    per_sensor_errors = []
    criterion = nn.MSELoss(reduction='none')
    base_edge_index = define_bridge_graph().to(device)

    with torch.no_grad():
        for inputs_batch in dataloader:
            inputs = inputs_batch.permute(0, 2, 1).unsqueeze(-1).to(device)
            outputs = model(inputs, base_edge_index)
            loss_tensor = criterion(outputs, inputs)
            loss_per_node = loss_tensor.mean(dim=(1, 3))
            sample_error = loss_per_node.mean(dim=1)
            all_errors.extend(sample_error.cpu().numpy())
            per_sensor_errors.extend(loss_per_node.cpu().numpy())

    return np.array(all_errors), np.array(per_sensor_errors)


def plot_reconstruction(original, reconstructed, error, title, output_path):
    num_sensors = original.shape[0]
    fig, axes = plt.subplots(num_sensors, 1, figsize=(15, 2.5 * num_sensors), sharex=True)
    fig.suptitle(f'{title}\nError MSE Global: {error:.6f}', fontsize=16)

    for i in range(num_sensors):
        ax = axes[i] if num_sensors > 1 else axes
        ax.plot(original[i], 'b-', label='Señal Original')
        ax.plot(reconstructed[i], 'r--', label='Señal Reconstruida')
        ax.set_title(f'Sensor {i + 1}')
        ax.legend()
        ax.grid(True)

    plt.xlabel('Paso de Tiempo')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Gráfico de reconstrucción guardado en: {output_path}")


def plot_error_distribution(errors, threshold, output_path, title_prefix="Sanos"):
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=50, kde=True, color='skyblue' if title_prefix == "Sanos" else "salmon")
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Umbral de Anomalía ({threshold:.6f})')
    plt.title(f'Distribución del Error de Reconstrucción en Datos {title_prefix}')
    plt.xlabel('Error MSE')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Gráfico de distribución de error guardado en: {output_path}")


def plot_damage_localization_heatmap(per_sensor_error, title, output_path):
    sensor_coords = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (2, 1), 5: (2, -1)}
    x_coords = [v[0] for v in sensor_coords.values()]
    y_coords = [v[1] for v in sensor_coords.values()]
    plt.figure(figsize=(12, 5))
    plt.scatter(x_coords, y_coords, c=per_sensor_error, cmap='Reds', s=500, edgecolors='black')
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f"S{i + 1}\n{per_sensor_error[i]:.4f}", ha='center', va='center', color='black')
    cbar = plt.colorbar();
    cbar.set_label('Error de Reconstrucción MSE')
    plt.title(title, fontsize=16)
    plt.xlabel('Posición Longitudinal del Puente (simplificado)');
    plt.ylabel('Posición Transversal (simplificado)')
    plt.xticks(ticks=[0, 1, 2], labels=['Extremo Miraflores', 'Centro', 'Extremo Surquillo']);
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.6);
    plt.savefig(output_path);
    plt.close()
    logging.info(f"Mapa de calor de localización guardado en: {output_path}")


# --- SCRIPT PRINCIPAL DE INFERENCIA ---
if __name__ == '__main__':
    # --- 1. CONFIGURACIÓN ---
    training_results_dir = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756"
    damaged_data_dir = r"D:\descargas 2025-2\articulo tesis delgadillo\Aceleraciones con daño\Aceleraciones"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    inference_output_dir = os.path.join(training_results_dir, f"inference_analysis_{timestamp}")
    os.makedirs(inference_output_dir, exist_ok=True)
    setup_logging(inference_output_dir)

    hp_path = os.path.join(training_results_dir, 'hyperparameters.json')
    try:
        with open(hp_path, 'r') as f:
            HP = json.load(f)
            logging.info(f"Hiperparámetros cargados desde JSON: {HP}")
    except FileNotFoundError:
        logging.error(f"No se pudo encontrar el archivo de hiperparámetros en: {hp_path}")
        exit()

    model_path = os.path.join(training_results_dir, 'best_model.pth')
    scaler_path = os.path.join(training_results_dir, 'scaler.gz')

    if not all(os.path.exists(p) for p in [model_path, scaler_path, damaged_data_dir]):
        logging.error("Error: No se encontró el modelo, el scaler o la carpeta de datos. Verifica las rutas.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    # --- 2. CARGAR MODELO Y SCALER ---
    logging.info("Inicializando la arquitectura del modelo según 'hyperparameters.json'...")
    try:
        model = SpatioTemporalAutoencoder(
            num_nodes=5,
            num_features=1,
            # Se leen todos los parámetros de la arquitectura desde el archivo de hiperparámetros.
            window_size=HP['window_size'],
            gnn_hidden=HP['gnn_hidden'],
            gnn_out=HP['gnn_out'],
            rnn_hidden=HP['rnn_hidden']
        ).to(device)
    except KeyError as e:
        logging.error(f"Error: La clave {e} no se encontró en 'hyperparameters.json'. "
                      f"Asegúrate de que el archivo contenga 'window_size', 'gnn_hidden', 'gnn_out' y 'rnn_hidden'.")
        exit()

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    logging.info("Modelo GCN+GRU entrenado cargado exitosamente.")

    scaler = joblib.load(scaler_path)
    logging.info("Scaler cargado exitosamente.")

    # --- 3. UMBRAL DE ANOMALÍA ---
    logging.info("Generando datos sintéticos 'sanos' para establecer umbral...")
    num_healthy_samples = 5000
    healthy_windows = np.random.randn(num_healthy_samples, 5, HP['window_size'])
    healthy_dataset = InferenceDataset(healthy_windows)
    healthy_loader = DataLoader(healthy_dataset, batch_size=HP.get('batch_size', 32), shuffle=False)

    logging.info("Calculando errores de reconstrucción en datos sanos...")
    healthy_errors, _ = get_reconstruction_errors(model, healthy_loader, device)

    anomaly_threshold = np.mean(healthy_errors) + 3 * np.std(healthy_errors)
    logging.info(f"Umbral de anomalía establecido en: {anomaly_threshold:.6f}")
    plot_error_distribution(healthy_errors, anomaly_threshold,
                            os.path.join(inference_output_dir, "healthy_error_distribution.png"))

    # --- 4. ANÁLISIS DE DATOS CON DAÑO ---
    logging.info("\n--- Iniciando análisis con datos de sismo ---")
    damaged_windows = load_damaged_data(damaged_data_dir, scaler, HP)

    if damaged_windows is not None and len(damaged_windows) > 0:
        damaged_dataset = InferenceDataset(damaged_windows)
        damaged_loader = DataLoader(damaged_dataset, batch_size=HP.get('batch_size', 32), shuffle=False)
        damaged_errors, per_sensor_damaged_errors = get_reconstruction_errors(model, damaged_loader, device)

        # --- 5. RESULTADOS ---
        anomalous_windows_count = np.sum(damaged_errors > anomaly_threshold)
        logging.info(
            f"Análisis completo. {anomalous_windows_count} de {len(damaged_errors)} ventanas superaron el umbral.")

        if anomalous_windows_count > 0:
            logging.warning(f"¡ANOMALÍA SIGNIFICATIVA DETECTADA!")
            # ...(resto del código de ploteo)...
        else:
            logging.info("No se detectaron anomalías significativas.")

    logging.info("\n--- Análisis de Inferencia Finalizado ---")
