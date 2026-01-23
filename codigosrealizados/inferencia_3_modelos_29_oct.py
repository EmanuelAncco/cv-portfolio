import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize  # Importar Normalize para heatmap lineal
import matplotlib.cm as cm  # Para mapas de color
import seaborn as sns
import json
import joblib
import logging
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from datetime import datetime
import glob
import random  # Para seleccionar colores
import re  # Para parsear el log
# --- NUEVA IMPORTACIÓN ---
# Necesaria para calcular el Structural Similarity Index (SSIM)
from skimage.metrics import structural_similarity
# --- NUEVA IMPORTACIÓN --- Para dibujar el grafo
import networkx as nx

# --- Configuración del Logging (Simplificada) ---
# Configuramos el logger raíz UNA SOLA VEZ.
# El FileHandler se añadirá dinámicamente en run_inference_and_plot
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()  # Logger Raíz

# Evitar añadir handlers duplicados si se re-ejecuta en un notebook
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # Consola
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)


# --- DEFINICIONES (IDÉNTICAS AL SCRIPT DE ENTRENAMIENTO) ---

def define_bridge_graph():
    """Define la estructura del grafo del puente (nodos 0-4)."""
    # Nodos: 0, 1, 2, 3, 4 (corresponden a Sensores 1 a 5)
    # Conexiones definidas en el script original
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    if edge_index.max() >= 5:
        raise ValueError(f"Índice de nodo inválido en edge_index: {edge_index.max()}. Se esperan nodos 0-4.")
    return edge_index


class SpatioTemporalWindowDataset(Dataset):
    """Dataset para cargar ventanas espacio-temporales."""

    def __init__(self, data_dict, window_size, stride=1, sensor_ids=None):
        self.window_size = window_size
        self.stride = stride
        local_logger = logging.getLogger(self.__class__.__name__)

        if not data_dict:
            raise ValueError("El diccionario de datos de entrada está vacío.")

        # Si sensor_ids no se provee, los infiere del diccionario
        expected_ids = set(sensor_ids) if sensor_ids else set(data_dict.keys())
        provided_ids = set(data_dict.keys())

        if sensor_ids and not expected_ids.issubset(provided_ids):
            missing = expected_ids - provided_ids
            local_logger.warning(
                f"Faltan datos para IDs de sensor: {missing}. Se usarán los disponibles: {provided_ids.intersection(expected_ids)}")

        ids_to_use = sorted(list(expected_ids.intersection(provided_ids)))
        if not ids_to_use:
            raise ValueError("No hay datos válidos para los IDs de sensor requeridos.")

        self.data_dict = {sid: data_dict[sid] for sid in ids_to_use}
        self.num_nodes = len(self.data_dict)

        # Validar que todos los datos (arrays de np) no sean None
        valid_data_values = [data for data in self.data_dict.values() if data is not None and len(data) > 0]
        if not valid_data_values:
            raise ValueError("El diccionario de datos no contiene arrays válidos.")

        min_len = min(len(data) for data in valid_data_values)

        if min_len < window_size:
            raise ValueError(
                f"Longitud mínima de datos ({min_len}) es menor que window_size ({window_size}). No se pueden crear ventanas.")

        processed_data = []
        valid_ids_used = []  # Track which sensors actually contribute data

        # Iterar sobre los IDs que decidimos usar
        for sid in ids_to_use:
            sensor_data = self.data_dict.get(sid)  # Usar .get para seguridad
            if sensor_data is not None and len(sensor_data) >= window_size:
                sensor_data = sensor_data[:min_len]  # Truncate to min_len

                # Asegurar que sea (N, 1)
                if sensor_data.ndim == 1:
                    sensor_data = sensor_data[:, np.newaxis]
                elif sensor_data.ndim != 2 or sensor_data.shape[1] != 1:
                    local_logger.error(f"Forma inesperada para sensor {sid}: {sensor_data.shape}. Omitiendo.")
                    continue

                processed_data.append(sensor_data)
                valid_ids_used.append(sid)
            else:
                local_logger.warning(f"Datos inválidos o insuficientes para sensor {sid}. Omitiendo.")

        if not processed_data:
            raise ValueError("No se procesaron datos válidos de ningún sensor.")

        # Shape: (min_len, num_valid_nodes)
        self.data = np.concatenate(processed_data, axis=1)
        self.num_nodes = self.data.shape[1]  # Update num_nodes based on actual data used
        local_logger.info(f"Datos concatenados con shape: {self.data.shape}")

        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples <= 0:
            local_logger.warning(
                f"Número de muestras calculado es {self.n_samples} (<=0). Verifique window_size, stride y longitud de datos ({len(self.data)}).")
            self.n_samples = 0

        local_logger.info(
            f"Dataset creado con {self.num_nodes} sensores válidos ({valid_ids_used}), {len(self.data)} puntos de tiempo, {self.n_samples} ventanas.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        if start < 0 or end > len(self.data):
            # Log detailed error info
            logger.error(
                f"Índice {idx} genera rango [{start}, {end}) fuera de límites para datos de longitud {len(self.data)}. Stride={self.stride}, WinSize={self.window_size}, N_Samples={self.n_samples}")
            raise IndexError(f"Índice {idx} fuera de rango.")

        window = self.data[start:end]  # Shape: (window_size, num_nodes)
        window = window[:, :, np.newaxis]  # Shape: (window_size, num_nodes, 1)
        return torch.FloatTensor(window), torch.FloatTensor(window)


class GNNLayer(nn.Module):
    """Bloque de capas GCN."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index):
        # Asegurarse de que edge_index esté en el mismo dispositivo que x
        edge_index = edge_index.to(x.device)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SpatioTemporalAutoencoder(nn.Module):
    """Arquitectura ST-GAE."""

    def __init__(self, num_nodes, num_features, window_size, gnn_hidden=32, gnn_out=16, rnn_hidden=64):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_features = num_features
        self.gnn_hidden_dim = gnn_hidden
        self.gnn_encoder_out_dim = gnn_out
        self.rnn_encoder_hidden_dim = rnn_hidden

        local_logger = logging.getLogger(self.__class__.__name__)

        self.rnn_decoder_output_dim = self.gnn_hidden_dim * num_nodes

        local_logger.info(f"Initializing STAutoencoder: N={num_nodes}, F={num_features}, T={window_size}")
        local_logger.info(f"  GNN Encoder: {num_features} -> {self.gnn_hidden_dim} -> {self.gnn_encoder_out_dim}")
        local_logger.info(
            f"  RNN Encoder: Input={self.gnn_encoder_out_dim * num_nodes}, Hidden={self.rnn_encoder_hidden_dim}, Layers=2")
        local_logger.info(
            f"  RNN Decoder: Input={self.rnn_encoder_hidden_dim}, Hidden={self.rnn_decoder_output_dim}, Layers=2")
        local_logger.info(f"  GNN Decoder: {self.gnn_hidden_dim} -> {self.gnn_hidden_dim} -> {num_features}")

        self.gnn_encoder = GNNLayer(num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)

        self.rnn_encoder = nn.GRU(input_size=self.gnn_encoder_out_dim * num_nodes,
                                  hidden_size=self.rnn_encoder_hidden_dim,
                                  batch_first=True, num_layers=2)

        self.rnn_decoder = nn.GRU(input_size=self.rnn_encoder_hidden_dim,
                                  hidden_size=self.rnn_decoder_output_dim,
                                  batch_first=True, num_layers=2)

        self.gnn_decoder = GNNLayer(self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index):
        # x shape: [B, T, N, F]
        batch_size, _, _, _ = x.shape

        # 1. Preparar datos para GNN Encoder
        # Reshape de [B, T, N, F] -> [B*T, N, F]
        x_reshaped = x.view(batch_size * self.window_size, self.num_nodes, self.num_features)

        # Mover edge_index al dispositivo de x
        edge_index = edge_index.to(x.device)

        # 2. GNN Encoder (Vectorizado)
        # Aplicar GNN a todos los (B*T) frames a la vez
        # gnn_encoded shape: [B*T, N, gnn_encoder_out_dim]
        gnn_encoded = self.gnn_encoder(x_reshaped, edge_index)

        # 3. Preparar datos para RNN Encoder
        # Reshape de [B*T, N, gnn_out] -> [B, T, N * gnn_out]
        gnn_encoded = gnn_encoded.view(batch_size, self.window_size, self.num_nodes, self.gnn_encoder_out_dim)
        rnn_input = gnn_encoded.view(batch_size, self.window_size, -1)  # Shape: [B, T, N*gnn_out]

        # 4. RNN Encoder
        _, h_n = self.rnn_encoder(rnn_input)  # h_n shape: [num_layers, B, rnn_hidden]

        # 5. RNN Decoder
        # Usar el último estado oculto
        latent_vector = h_n[-1].unsqueeze(1).repeat(1, self.window_size, 1)  # Shape: [B, T, rnn_hidden]
        rnn_decoded, _ = self.rnn_decoder(latent_vector)  # Shape: [B, T, rnn_decoder_output_dim (N*gnn_hidden)]

        # 6. Preparar datos para GNN Decoder
        # rnn_decoder_output_dim es self.gnn_hidden_dim * self.num_nodes
        # Reshape de [B, T, N*gnn_hidden] -> [B*T, N, gnn_hidden]
        gnn_input_decoder = rnn_decoded.contiguous().view(batch_size * self.window_size, self.num_nodes,
                                                          self.gnn_hidden_dim)

        # 7. GNN Decoder (Vectorizado)
        # Aplicar GNN a todos los (B*T) frames decodificados a la vez
        # reconstructed_frames shape: [B*T, N, F]
        reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index)

        # 8. Reshape final
        # Reshape de [B*T, N, F] -> [B, T, N, F]
        reconstructed_x = reconstructed_frames.view(batch_size, self.window_size, self.num_nodes, self.num_features)

        return reconstructed_x


# --- FUNCIONES DE CARGA Y PROCESAMIENTO ---

def load_data_from_dir(directory, num_nodes, max_len=None):
    """
    Carga datos de sensores desde un directorio, buscando archivos que comiencen con '<id>_'.
    Concatena los datos de múltiples archivos para un mismo sensor.
    Devuelve un array (max_len, num_nodes) y max_len.
    """
    local_logger = logging.getLogger(f"{__name__}.load_data")
    local_logger.info(f"Searching files in: {directory} with pattern '<id>_*'")
    all_sensor_data = {}
    loaded_files_count = 0
    min_length = float('inf')

    # Los sensores están indexados desde 1
    sensor_ids_to_find = list(range(1, num_nodes + 1))

    for i in sensor_ids_to_find:
        search_pattern = os.path.join(directory, f"{i}_*")
        file_list = glob.glob(search_pattern)

        if not file_list:
            local_logger.warning(f"No files found for sensor {i} with pattern: {search_pattern}")
            continue

        file_list.sort()
        sensor_df_list = []
        for filepath in file_list:
            try:
                # Asumimos una sola columna de datos
                # ¡¡¡ LA CORRECCIÓN CLAVE ESTÁ AQUÍ !!!
                # Cambiamos usecols=[0] por usecols=[1] para que coincida con el script de entrenamiento
                # (La columna 0 es el índice/tiempo, la Columna 1 es la aceleración)
                df = pd.read_csv(filepath, header=None, sep=r'\s+', usecols=[1], engine='python')
                if df.empty:
                    local_logger.warning(f"Empty file skipped: {filepath}")
                    continue
                sensor_df_list.append(df)
            except Exception as e:
                local_logger.error(f"Error reading or processing file {filepath}: {e}")

        if not sensor_df_list:
            local_logger.warning(
                f"Could not read any valid data for sensor {i} despite finding files.")
            continue

        full_sensor_df = pd.concat(sensor_df_list, ignore_index=True)
        all_sensor_data[i] = full_sensor_df.iloc[:, 0].values
        loaded_files_count += len(file_list)

        if len(all_sensor_data[i]) < min_length:
            min_length = len(all_sensor_data[i])

    if not all_sensor_data:
        local_logger.error("No data loaded from any sensor.")
        return None, 0

    if len(all_sensor_data) < num_nodes:
        local_logger.warning(
            f"Expected data for {num_nodes} sensors, but only loaded {len(all_sensor_data)}. Continuing...")
        # Llenar los datos faltantes con ceros o manejar de otra forma
        for i in sensor_ids_to_find:
            if i not in all_sensor_data:
                local_logger.warning(f"Filling missing data for sensor {i} with zeros.")
                # Usar min_length (de los sensores encontrados) o un valor por defecto
                fill_length = min_length if min_length != float('inf') else 1
                all_sensor_data[i] = np.zeros(fill_length)

    # Si ningún archivo se cargó, min_length sigue siendo inf
    if min_length == float('inf'):
        if max_len:  # Si se pasó un max_len, usarlo
            min_length = max_len
        else:
            local_logger.error("Could not determine minimum data length.")
            return None, 0

    local_logger.info(
        f"Load completed. Files loaded: {loaded_files_count}, Sensors with data: {len(all_sensor_data)}.")

    # Truncar todos los arrays a la longitud mínima o max_len
    # Si max_len no se proporciona, se usa min_length (la más corta encontrada)
    if max_len is None:
        max_len = min_length

    local_logger.info(f"Normalizing all sensors to length: {max_len}")

    processed_data = np.zeros((max_len, num_nodes))
    for i in sensor_ids_to_find:
        sensor_data = all_sensor_data.get(i)
        if sensor_data is None:
            local_logger.warning(f"No data for sensor {i} in final dictionary. Using zeros.")
            # 'processed_data' ya está inicializado a ceros, así que no hacemos nada
            continue

        # Truncar o rellenar (pad)
        if len(sensor_data) >= max_len:
            processed_data[:, i - 1] = sensor_data[:max_len]
        else:
            # Rellenar con el último valor o con ceros si está vacío
            pad_value = sensor_data[-1] if len(sensor_data) > 0 else 0
            padding = np.full(max_len - len(sensor_data), pad_value)
            processed_data[:, i - 1] = np.concatenate((sensor_data, padding))

    local_logger.info(f"Processed data with final shape: {processed_data.shape}.")

    return processed_data, max_len


def perform_inference(model, dataloader, device, edge_index):
    """
    Ejecuta la inferencia y calcula errores (MSE) y SSIM.
    --- MODIFICADO ---
    Ahora devuelve también `all_ssim_per_sensor_np`.
    """
    local_logger = logging.getLogger(f"{__name__}.inference")
    model.eval()
    all_inputs_np = []
    all_outputs_np = []
    all_losses_np = []  # Pérdida promedio por ventana (MSE)
    all_losses_per_sensor_np = []  # Array (n_samples, n_nodes) (MSE)
    all_ssim_per_sensor_np = []  # --- NUEVO --- Array (n_samples, n_nodes) (SSIM)

    criterion_none = nn.MSELoss(reduction='none')

    total_batches = len(dataloader)
    if total_batches == 0:
        local_logger.error("DataLoader is empty. No data for inference.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    processed_windows = 0
    num_nodes = dataloader.dataset.num_nodes  # Obtener num_nodes del dataset

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Inference', leave=False, total=total_batches)
        for batch_idx, (inputs, _) in enumerate(progress_bar):
            if inputs is None or len(inputs) == 0:
                local_logger.warning(f"Batch {batch_idx + 1}/{total_batches} empty. Skipping.")
                continue

            inputs = inputs.to(device)  # Shape: (batch, time, nodes, feats)

            # Asegurarse de que edge_index esté en el dispositivo correcto
            edge_index = edge_index.to(device)

            try:
                outputs = model(inputs, edge_index)  # Shape: (batch, time, nodes, feats)

                # --- Cálculo de MSE ---
                loss_elementwise = criterion_none(outputs, inputs)  # Shape: (batch, time, nodes, feats)
                loss_per_window = torch.mean(loss_elementwise, dim=(1, 2, 3))  # Shape: (batch,)
                loss_per_sensor_window = torch.mean(loss_elementwise, dim=(1, 3))  # Shape: (batch, nodes)

                all_losses_np.append(loss_per_window.cpu().numpy())
                all_losses_per_sensor_np.append(loss_per_sensor_window.cpu().numpy())
                all_inputs_np.append(inputs.cpu().numpy())
                all_outputs_np.append(outputs.cpu().numpy())

                # --- NUEVO: Cálculo de SSIM ---
                # Mover tensores a CPU/numpy para cálculo de SSIM
                inputs_cpu = inputs.cpu().numpy()
                outputs_cpu = outputs.cpu().numpy()

                batch_size = inputs_cpu.shape[0]
                window_len = inputs_cpu.shape[1]

                # Pre-alocar array para SSIM del lote
                batch_ssim_per_sensor = np.zeros((batch_size, num_nodes))

                for i in range(batch_size):
                    for n in range(num_nodes):
                        sig_in = inputs_cpu[i, :, n, 0]  # Señal original (ventana, sensor n)
                        sig_out = outputs_cpu[i, :, n, 0]  # Señal reconstruida (ventana, sensor n)

                        # data_range es crucial para SSIM. Se calcula sobre la señal original.
                        data_range = sig_in.max() - sig_in.min()
                        if data_range == 0:
                            # Si la señal es plana, SSIM es 1 si son iguales, 0 si no (o cercano)
                            # Poner data_range=1 y dejar que skimage decida es una opción segura
                            data_range = 1.0

                            # win_size (para SSIM) no puede ser > longitud de la señal.
                        # El default (7) es usualmente seguro, pero añadimos un check.
                        # Debe ser impar y >= 3.
                        current_win_size = min(7, window_len)
                        if current_win_size < 3:  # Si la ventana es muy corta (1 o 2 puntos)
                            ssim_val = 1.0 if np.allclose(sig_in, sig_out) else 0.0  # Comparación directa
                        else:
                            if current_win_size % 2 == 0:  # Asegurar que sea impar
                                current_win_size -= 1
                            try:
                                ssim_val = structural_similarity(sig_in, sig_out,
                                                                 data_range=data_range,
                                                                 win_size=current_win_size)
                            except ValueError as ve:
                                # Posible error si win_size > image extent
                                local_logger.warning(
                                    f"SSIM calculation failed for window {processed_windows + i}, sensor {n + 1}. Setting to 0. Error: {ve}")
                                ssim_val = 0.0  # O un valor que indique error

                        batch_ssim_per_sensor[i, n] = ssim_val

                all_ssim_per_sensor_np.append(batch_ssim_per_sensor)
                # --- FIN DE CÁLCULO SSIM ---

                processed_windows += len(inputs)
                progress_bar.set_postfix({'Windows': processed_windows})

            except Exception as e:
                local_logger.error(f"Error during inference on batch {batch_idx + 1}/{total_batches}: {e}",
                                   exc_info=True)
                continue  # Saltar al siguiente lote

    # Concatenar resultados al final
    if not all_losses_np:
        local_logger.error("No windows were processed successfully.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    all_inputs_np = np.concatenate(all_inputs_np, axis=0)
    all_outputs_np = np.concatenate(all_outputs_np, axis=0)
    all_losses_np = np.concatenate(all_losses_np, axis=0)  # Shape: (n_samples,)
    all_losses_per_sensor_np = np.concatenate(all_losses_per_sensor_np, axis=0)  # Shape: (n_samples, n_nodes)
    all_ssim_per_sensor_np = np.concatenate(all_ssim_per_sensor_np, axis=0)  # --- NUEVO --- Shape: (n_samples, n_nodes)

    local_logger.info(f"Inference completed. Processed {processed_windows} windows.")
    local_logger.info(f"Final shape of losses per window: {all_losses_np.shape}")
    local_logger.info(f"Final shape of losses per sensor: {all_losses_per_sensor_np.shape}")
    local_logger.info(f"Final shape of SSIM per sensor: {all_ssim_per_sensor_np.shape}")  # --- NUEVO ---

    return all_inputs_np, all_outputs_np, all_losses_np, all_losses_per_sensor_np, all_ssim_per_sensor_np


# --- FUNCIONES DE PLOTEO ---
local_logger_plot = logging.getLogger(f"{__name__}.plotting")
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    local_logger_plot.warning("Style 'seaborn-v0_8-whitegrid' not found, using 'ggplot'.")
    plt.style.use('ggplot')

# --- Paletas de colores más vibrantes --- MODIFICADO ---
# Para Healthy vs Damage (Azul y Naranja de tab10 o Set1)
palette_scenario = sns.color_palette("Set1", 2)

# Para Sensores (usaremos un mapa de color perceptualmente uniforme como 'viridis' o 'plasma')
sensor_colormap = plt.cm.viridis  # Verde-Azul a Amarillo (más vibrante que coolwarm)
num_sensors_global = 5  # Asumimos 5 sensores, ajustar si es necesario
sensor_colors_mapped = sensor_colormap(np.linspace(0.1, 0.9, num_sensors_global))  # Evitar extremos muy claros/oscuros


# --- (OPCIONAL) VERSIÓN TRADUCIDA DE LA FUNCIÓN ANTIGUA ---
# --- Esta función ya no se llama en run_inference_and_plot ---
def plot_reconstruction_sample_translated(original_scaled, reconstructed_scaled, scaler, window_idx, output_dir,
                                          prefix):
    """
    Genera gráfico de reconstrucción para una muestra específica (VENTANA ÚNICA).
    Aplica inverse_transform del scaler para mostrar las unidades originales.
    Usa colores distintos por sensor.
    original_scaled, reconstructed_scaled: shape (time, nodes, feats)
    """
    if original_scaled is None or reconstructed_scaled is None or original_scaled.size == 0 or reconstructed_scaled.size == 0:
        local_logger_plot.warning(
            f"Original or reconstructed data missing/empty for window {window_idx} ({prefix}). Plot will not be generated.")
        return

    # Asegurar shapes correctos
    if original_scaled.ndim != 3 or reconstructed_scaled.ndim != 3 or original_scaled.shape != reconstructed_scaled.shape:
        local_logger_plot.error(
            f"Inconsistent shapes for plotting: Orig {original_scaled.shape}, Rec {reconstructed_scaled.shape}")
        return
    if original_scaled.shape[2] != 1:  # Asumiendo F=1
        local_logger_plot.error(f"Expected 1 feature, but found {original_scaled.shape[2]}")
        return

    num_sensors = original_scaled.shape[1]
    time_steps = original_scaled.shape[0]

    try:
        # --- INICIO DE LA TRANSFORMACIÓN INVERSA ---
        original_flat = original_scaled.reshape(-1, 1)
        reconstructed_flat = reconstructed_scaled.reshape(-1, 1)
        original_inv_flat = scaler.inverse_transform(original_flat)
        reconstructed_inv_flat = scaler.inverse_transform(reconstructed_flat)
        original_inv = original_inv_flat.reshape(original_scaled.shape)
        reconstructed_inv = reconstructed_inv_flat.reshape(reconstructed_scaled.shape)
        # --- FIN DE LA TRANSFORMACIÓN INVERSA ---

    except Exception as e:
        local_logger_plot.error(f"Error during inverse_transform in plotting: {e}. Plotting scaled data.")
        original_inv = original_scaled
        reconstructed_inv = reconstructed_scaled

    # Calcular error sobre los datos invertidos (en unidades reales)
    error_signal = original_inv.squeeze(-1) - reconstructed_inv.squeeze(-1)  # Shape (time, nodes)

    fig, axes = plt.subplots(num_sensors, 2, figsize=(15, 3 * num_sensors), sharex=True, squeeze=False)
    # --- TEXTO TRADUCIDO ---
    fig.suptitle(f'Sample Reconstruction {prefix.capitalize()} (Window Index: {window_idx})', fontsize=16)

    for i in range(num_sensors):
        color = sensor_colors_mapped[i]  # Usar color mapeado
        ax_sig = axes[i, 0]
        ax_err = axes[i, 1]

        # Usar el color asignado
        # --- TEXTO TRADUCIDO ---
        ax_sig.plot(original_inv[:, i, 0], label='Original', color=color, linewidth=1.5, alpha=0.8)
        ax_sig.plot(reconstructed_inv[:, i, 0], label='Reconstructed', color=color, linestyle='--', linewidth=1.5,
                    alpha=1.0)
        ax_sig.set_title(f'Sensor {i + 1}: Original vs. Reconstructed Signal')
        ax_sig.set_ylabel('Original Value (Inverted)')
        ax_sig.legend(fontsize='small')
        ax_sig.grid(True, linestyle=':')

        # Usar un color estándar (rojo) para el error
        # --- TEXTO TRADUCIDO ---
        ax_err.plot(error_signal[:, i], label='Error', color='red', linewidth=1.5)
        ax_err.set_title(f'Sensor {i + 1}: Reconstruction Error')
        ax_err.set_ylabel('Error (Original Units)')
        ax_err.grid(True, linestyle=':')
        ax_err.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax_err.legend(fontsize='small')

    # --- TEXTO TRADUCIDO ---
    axes[num_sensors - 1, 0].set_xlabel('Time Step in Window')
    axes[num_sensors - 1, 1].set_xlabel('Time Step in Window')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, f"{prefix}_reconstruction_sample_{window_idx}_OLD.png")  # Renombrado
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Reconstruction graph (old style) saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving graph {filename}: {e}")
    plt.close(fig)


# --- NUEVO GRÁFICO DE RECONSTRUCCIÓN POR SENSOR ---
def plot_sensor_reconstruction_samples(all_originals, all_reconstructions, scaler, num_sensors, output_dir, prefix):
    """
    Genera un gráfico por CADA sensor, mostrando la reconstrucción de
    DOS muestras aleatorias (ventanas) y sus errores.
    all_originals, all_reconstructions: shape (n_samples, time, nodes, feats)
    """
    local_logger_plot.info(f"Generating per-sensor reconstruction plots for {prefix} data...")
    if all_originals is None or all_reconstructions is None or all_originals.size == 0 or all_reconstructions.size == 0:
        local_logger_plot.warning(
            f"Original or reconstructed data missing/empty ({prefix}). No per-sensor graphs will be generated.")
        return

    num_samples = all_originals.shape[0]
    if num_samples < 2:
        local_logger_plot.warning(f"Need at least 2 samples to plot, but found {num_samples} ({prefix}). Skipping.")
        return

    # Definir los dos colores distintos para las muestras
    color1 = '#377eb8'  # Azul EDAQ
    color2 = '#4daf4a'  # Verde EDAQ
    error_color = '#e41a1c'  # Rojo EDAQ

    for i in range(num_sensors):
        # Seleccionar dos índices de ventana aleatorios y distintos
        try:
            idx1, idx2 = np.random.choice(num_samples, 2, replace=False)
        except ValueError as e:
            local_logger_plot.error(
                f"Error selecting random indices for sensor {i + 1} ({prefix}): {e}. Skipping sensor.")
            continue

        # --- Preparar datos (Muestra 1) ---
        original_sample1_scaled = all_originals[idx1, :, i, 0]  # Shape (time,)
        reconstructed_sample1_scaled = all_reconstructions[idx1, :, i, 0]  # Shape (time,)

        # --- Preparar datos (Muestra 2) ---
        original_sample2_scaled = all_originals[idx2, :, i, 0]  # Shape (time,)
        reconstructed_sample2_scaled = all_reconstructions[idx2, :, i, 0]  # Shape (time,)

        try:
            # --- INICIO DE LA TRANSFORMACIÓN INVERSA ---
            # Reshape a (N, 1) para el scaler
            original_sample1_inv = scaler.inverse_transform(original_sample1_scaled.reshape(-1, 1)).squeeze()
            reconstructed_sample1_inv = scaler.inverse_transform(reconstructed_sample1_scaled.reshape(-1, 1)).squeeze()

            original_sample2_inv = scaler.inverse_transform(original_sample2_scaled.reshape(-1, 1)).squeeze()
            reconstructed_sample2_inv = scaler.inverse_transform(reconstructed_sample2_scaled.reshape(-1, 1)).squeeze()
            # --- FIN DE LA TRANSFORMACIÓN INVERSA ---

        except Exception as e:
            local_logger_plot.error(
                f"Error during inverse_transform in plot_sensor_reconstruction_samples: {e}. Plotting scaled data.")
            original_sample1_inv = original_sample1_scaled
            reconstructed_sample1_inv = reconstructed_sample1_scaled
            original_sample2_inv = original_sample2_scaled
            reconstructed_sample2_inv = reconstructed_sample2_scaled

        # Calcular error sobre los datos invertidos (en unidades reales)
        error_signal_1 = original_sample1_inv - reconstructed_sample1_inv  # Shape (time,)
        error_signal_2 = original_sample2_inv - reconstructed_sample2_inv  # Shape (time,)

        # --- Crear Gráfico ---
        # 2 filas (para 2 muestras), 2 columnas (para señal y error)
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, squeeze=False)
        fig.suptitle(f'Sensor {i + 1}: Reconstruction of Two Random Samples ({prefix.capitalize()} Data)', fontsize=16)

        # --- Muestra 1 (Fila 0) ---
        ax_sig1 = axes[0, 0]
        ax_err1 = axes[0, 1]

        ax_sig1.plot(original_sample1_inv, label=f'Original (Sample {idx1})', color=color1, linewidth=1.5, alpha=0.8)
        ax_sig1.plot(reconstructed_sample1_inv, label=f'Reconstructed (Sample {idx1})', color=color1, linestyle='--',
                     linewidth=1.5, alpha=1.0)
        ax_sig1.set_title(f'Sample 1 (Window: {idx1}) - Original vs. Reconstructed')
        ax_sig1.set_ylabel('Original Value (Inverted)')
        ax_sig1.legend(fontsize='small')
        ax_sig1.grid(True, linestyle=':')

        ax_err1.plot(error_signal_1, label='Error', color=error_color, linewidth=1.5)
        ax_err1.set_title(f'Sample 1 (Window: {idx1}) - Reconstruction Error')
        ax_err1.set_ylabel('Error (Original Units)')
        ax_err1.grid(True, linestyle=':')
        ax_err1.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax_err1.legend(fontsize='small')

        # --- Muestra 2 (Fila 1) ---
        ax_sig2 = axes[1, 0]
        ax_err2 = axes[1, 1]

        ax_sig2.plot(original_sample2_inv, label=f'Original (Sample {idx2})', color=color2, linewidth=1.5, alpha=0.8)
        ax_sig2.plot(reconstructed_sample2_inv, label=f'Reconstructed (Sample {idx2})', color=color2, linestyle='--',
                     linewidth=1.5, alpha=1.0)
        ax_sig2.set_title(f'Sample 2 (Window: {idx2}) - Original vs. Reconstructed')
        ax_sig2.set_ylabel('Original Value (Inverted)')
        ax_sig2.legend(fontsize='small')
        ax_sig2.grid(True, linestyle=':')

        ax_err2.plot(error_signal_2, label='Error', color=error_color, linewidth=1.5)  # Mismo color de error
        ax_err2.set_title(f'Sample 2 (Window: {idx2}) - Reconstruction Error')
        ax_err2.set_ylabel('Error (Original Units)')
        ax_err2.grid(True, linestyle=':')
        ax_err2.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax_err2.legend(fontsize='small')

        # --- Etiquetas X ---
        axes[1, 0].set_xlabel('Time Step in Window')
        axes[1, 1].set_xlabel('Time Step in Window')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        filename = os.path.join(output_dir, f"{prefix}_sensor_{i + 1}_reconstruction_samples.png")
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            local_logger_plot.info(f"Graph of reconstruction by sensor saved in: {filename}")
        except Exception as e:
            local_logger_plot.error(f"Error saving graph {filename}: {e}")
        plt.close(fig)


def plot_mse_comparison(mse_healthy_per_sensor_avg, mse_damage_per_sensor_avg, output_dir):
    """Genera gráfico comparativo de MSE promedio por sensor (sano vs. daño)."""
    if mse_healthy_per_sensor_avg is None or mse_damage_per_sensor_avg is None or \
            mse_healthy_per_sensor_avg.size == 0 or mse_damage_per_sensor_avg.size == 0 or \
            len(mse_healthy_per_sensor_avg) != len(mse_damage_per_sensor_avg):
        local_logger_plot.error("Invalid or inconsistent average MSE data per sensor for comparison.")
        return

    num_sensors = len(mse_healthy_per_sensor_avg)
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # --- TEXTO TRADUCIDO ---
    fig.suptitle('Damage Detection: Average Reconstruction Error per Sensor', fontsize=16)

    # Panel Izquierdo: Datos Sanos (Escala Lineal)
    axes[0].bar(sensor_labels, mse_healthy_per_sensor_avg, color=palette_scenario[0], edgecolor='black')  # Usar paleta
    axes[0].set_title('Healthy Data (Baseline)')
    axes[0].set_ylabel('Mean Squared Error (MSE)')
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, axis='y', linestyle=':')
    max_healthy_val = np.max(mse_healthy_per_sensor_avg) if mse_healthy_per_sensor_avg.size > 0 else 1
    for i, v in enumerate(mse_healthy_per_sensor_avg):
        axes[0].text(i, v + 0.03 * max(max_healthy_val, 1e-9), f"{v:.4e}", ha='center', va='bottom',
                     fontsize=9)  # Usar .4e

    # Panel Derecho: Datos del Sismo (Escala Logarítmica)
    valid_damage_mse = mse_damage_per_sensor_avg[mse_damage_per_sensor_avg > 0]
    use_log_scale = True
    plot_data_damage = mse_damage_per_sensor_avg
    # --- TEXTO TRADUCIDO ---
    y_label_damage = 'Mean Squared Error (MSE) (Log Scale)'

    # Si hay valores <= 0, no se puede usar escala log directamente
    if valid_damage_mse.size != num_sensors:
        local_logger_plot.warning(
            "Non-positive damage MSE errors found for some sensors. Using linear scale.")
        use_log_scale = False
        # --- TEXTO TRADUCIDO ---
        y_label_damage = 'Mean Squared Error (MSE)'

    axes[1].bar(sensor_labels, plot_data_damage, color=palette_scenario[1], edgecolor='black')  # Usar paleta
    # --- TEXTO TRADUCIDO ---
    axes[1].set_title('Damage Data' + (' (Log Scale)' if use_log_scale else ''))  # Titulo corregido
    axes[1].set_ylabel(y_label_damage)

    min_val_plot = 1e-9  # Valor mínimo para escala log si es aplicable

    if use_log_scale:
        axes[1].set_yscale('log')
        # Asegurarse de que el límite inferior sea visible y positivo
        min_positive_mse = np.min(valid_damage_mse) if valid_damage_mse.size > 0 else min_val_plot
        axes[1].set_ylim(bottom=max(min_positive_mse * 0.1, min_val_plot))
    else:
        axes[1].set_ylim(bottom=0)

    axes[1].grid(True, axis='y', linestyle=':')

    max_damage_val = np.max(plot_data_damage) if plot_data_damage.size > 0 else 1
    for i, v in enumerate(plot_data_damage):
        if use_log_scale:
            # Colocar texto encima de la barra, asegurando que sea visible en log scale
            text_pos = max(v, axes[1].get_ylim()[0]) * 1.5
        else:
            text_pos = v + 0.03 * max(max_damage_val, 1e-9)

        axes[1].text(i, text_pos, f"{v:.3e}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, "damage_detection_mse_comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"MSE comparison graph saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving graph {filename}: {e}")
    plt.close(fig)


def plot_damage_localization(losses_damage_per_sensor, losses_healthy_per_sensor, output_dir):
    """
    Genera gráfico de localización (violin+strip y factor de amplificación).
    Recibe las distribuciones completas de pérdidas: (n_samples, n_nodes)
    """
    if losses_damage_per_sensor is None or losses_healthy_per_sensor is None or \
            losses_damage_per_sensor.size == 0 or losses_healthy_per_sensor.size == 0 or \
            losses_damage_per_sensor.shape[1] != losses_healthy_per_sensor.shape[1]:
        local_logger_plot.error("Invalid or inconsistent loss data per sensor for localization.")
        return

    num_sensors = losses_damage_per_sensor.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))  # Ajustar tamaño
    # --- TEXTO TRADUCIDO ---
    fig.suptitle('Damage Localization Analysis per Sensor', fontsize=16)

    # --- Panel Superior: Violin Plot + Strip Plot Error con Daño (Escala Log) --- MODIFICADO ---

    # Preparar datos para Seaborn
    data_list = []
    for i in range(num_sensors):
        sensor_name = sensor_labels[i]
        # Filtrar no positivos para escala log
        positive_losses = losses_damage_per_sensor[:, i][losses_damage_per_sensor[:, i] > 0]
        if len(positive_losses) > 0:
            for loss_val in positive_losses:
                data_list.append({'Sensor': sensor_name, 'MSE': loss_val})  # Quitar color individual aqui
        else:
            # Añadir un placeholder pequeño si no hay datos positivos
            data_list.append({'Sensor': sensor_name, 'MSE': 1e-12})

    if not data_list or pd.DataFrame(data_list)['MSE'].max() <= 1e-11:  # Chequear si hay datos válidos
        local_logger_plot.error("No positive damage error data to plot violinplot.")
        # --- TEXTO TRADUCIDO ---
        axes[0].set_title('Damage Error Distribution (No positive data)')
    else:
        df_damage = pd.DataFrame(data_list)

        # Crear el violin plot
        sns.violinplot(x='Sensor', y='MSE', data=df_damage, ax=axes[0],
                       palette=sensor_colors_mapped,  # Usar colores mapeados para violines
                       inner='quartile', cut=0, scale='width', linewidth=1.5)  # Línea más gruesa
        # Superponer el strip plot
        # *** CAMBIO DE COLOR Y AJUSTES ***
        sns.stripplot(x='Sensor', y='MSE', data=df_damage, ax=axes[0],
                      color='black', alpha=0.1, size=2.5, jitter=0.3)  # Puntos negros semitransparentes

        # --- TEXTO TRADUCIDO ---
        axes[0].set_title('Reconstruction Error Distribution (Damage Data - with points)')
        axes[0].set_ylabel('Mean Squared Error (MSE) - Log Scale')
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle=':')
        # Ajustar límites Y si es necesario
        # Filtrar inf/nan antes de min/max
        valid_mse = df_damage['MSE'][np.isfinite(df_damage['MSE'])]
        if len(valid_mse) > 0:
            min_mse_plot = max(valid_mse.min() * 0.8, 1e-9)
            # Ajustar límite superior basado en percentil para ignorar outliers extremos
            max_mse_plot = np.percentile(valid_mse, 99.8) * 1.5
            axes[0].set_ylim(bottom=min_mse_plot, top=max_mse_plot)

    # --- Panel Inferior: Factor de Amplificación Mediano ---
    median_error_damage = np.median(losses_damage_per_sensor, axis=0)
    median_error_healthy = np.median(losses_healthy_per_sensor, axis=0)
    # Evitar división por cero o por valores muy pequeños
    median_error_healthy_safe = np.maximum(median_error_healthy, 1e-10)

    amplification_factor = median_error_damage / median_error_healthy_safe

    # Usar colores mapeados también en barras
    bar_colors = [sensor_colors_mapped[i] for i in range(num_sensors)]
    axes[1].bar(sensor_labels, amplification_factor, color=bar_colors, edgecolor='black', alpha=0.85)  # Más opaco
    # --- TEXTO TRADUCIDO ---
    axes[1].set_title('Median Error Amplification Factor (Damage vs. Healthy)')
    axes[1].set_ylabel('Median Error Damage / Median Error Healthy')
    axes[1].grid(True, axis='y', linestyle=':')  # Grid solo en Y para barras

    # Filtrar inf/nan antes de min/max para amp factor
    valid_amp = amplification_factor[np.isfinite(amplification_factor)]
    if len(valid_amp) > 0:
        max_amp = np.max(valid_amp)
        # Asegurar que el límite inferior sea 0 o un poco menos
        axes[1].set_ylim(bottom=min(0, np.min(valid_amp) * 0.9))
        for i, v in enumerate(amplification_factor):
            if np.isfinite(v):  # Solo añadir texto si el valor es finito
                axes[1].text(i, v + 0.02 * max_amp, f"{v:.2f}", ha='center', va='bottom',
                             fontsize=9.5)  # Ajustar posición y tamaño texto
    else:
        axes[1].set_ylim(bottom=0)  # Default si no hay datos válidos

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "damage_localization_analysis.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Damage localization graph saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving graph {filename}: {e}")
    plt.close(fig)


# --- NUEVOS GRÁFICOS PARA ANÁLISIS DE ARTÍCULO ---

def plot_error_distribution_kde(losses_healthy, losses_damage, output_dir):
    """
    Genera un gráfico KDE (Kernel Density Estimation) de la distribución
    de los errores de reconstrucción (MSE global por ventana).
    losses_healthy, losses_damage: shape (n_samples,)
    """
    if losses_healthy is None or losses_damage is None or losses_healthy.size == 0 or losses_damage.size == 0:
        local_logger_plot.warning("Empty loss data (healthy or damage), skipping KDE plot.")
        return

    local_logger_plot.info("Generating error distribution graph (KDE)...")
    plt.figure(figsize=(12, 7))

    # Usar log del error para mejor visualización si los rangos son muy amplios
    # Añadir un epsilon pequeño para evitar log(0)
    epsilon = 1e-12
    log_losses_healthy = np.log10(losses_healthy[losses_healthy > 0] + epsilon)  # Filtrar no positivos
    log_losses_damage = np.log10(losses_damage[losses_damage > 0] + epsilon)  # Filtrar no positivos

    if len(log_losses_healthy) > 1:  # KDE necesita al menos 2 puntos
        # --- TEXTO TRADUCIDO ---
        sns.kdeplot(log_losses_healthy, label='Healthy', color=palette_scenario[0], fill=True, bw_adjust=0.5,
                    alpha=0.6)  # Usar paleta
    else:
        local_logger_plot.warning("Not enough positive healthy data to plot KDE.")

    if len(log_losses_damage) > 1:  # KDE necesita al menos 2 puntos
        # --- TEXTO TRADUCIDO ---
        sns.kdeplot(log_losses_damage, label='Damage', color=palette_scenario[1], fill=True, bw_adjust=0.5,
                    alpha=0.6)  # Usar paleta
    else:
        local_logger_plot.warning("Not enough positive damage data to plot KDE.")

    # --- TEXTO TRADUCIDO ---
    plt.title('Reconstruction Error Distribution (MSE per Window)')
    plt.xlabel('Log10(Mean Squared Error)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':')

    filename = os.path.join(output_dir, "error_distribution_kde.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"KDE graph saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving KDE graph {filename}: {e}")
    plt.close()


def plot_error_timeseries(losses_healthy, losses_damage, output_dir):
    """
    Genera un gráfico del error de reconstrucción (MSE) a lo largo del tiempo (ventanas).
    losses_healthy, losses_damage: shape (n_samples,)
    """
    local_logger_plot.info("Generating error time series graph...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    # --- TEXTO TRADUCIDO ---
    fig.suptitle('Reconstruction Error (MSE) over Time (Windows)', fontsize=16)

    # Filtrar valores no positivos para escala logarítmica
    epsilon = 1e-12

    if losses_healthy is not None and losses_healthy.size > 0:
        valid_healthy_indices = np.where(losses_healthy > 0)[0]
        if len(valid_healthy_indices) > 0:
            # --- TEXTO TRADUCIDO ---
            axes[0].plot(valid_healthy_indices, losses_healthy[valid_healthy_indices], label='Healthy Error',
                         color=palette_scenario[0],  # Usar paleta
                         alpha=0.7, linewidth=0.5)  # Línea más fina
            axes[0].set_yscale('log')
            axes[0].set_ylabel('MSE (Log Scale)')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, linestyle=':')
        else:
            local_logger_plot.warning("No positive healthy errors to plot in log scale.")
            axes[0].plot([], [], label='Healthy Error')  # Para que aparezca la leyenda
        # --- TEXTO TRADUCIDO ---
        axes[0].set_title('Healthy Data (Baseline)')

    else:
        # --- TEXTO TRADUCIDO ---
        axes[0].set_title('Healthy Data (Not available)')

    if losses_damage is not None and losses_damage.size > 0:
        valid_damage_indices = np.where(losses_damage > 0)[0]
        if len(valid_damage_indices) > 0:
            # --- TEXTO TRADUCIDO ---
            axes[1].plot(valid_damage_indices, losses_damage[valid_damage_indices], label='Damage Error',
                         color=palette_scenario[1],  # Usar paleta
                         alpha=0.7, linewidth=0.5)  # Línea más fina
            axes[1].set_yscale('log')
            axes[1].set_ylabel('MSE (Log Scale)')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, linestyle=':')
        else:
            local_logger_plot.warning("No positive damage errors to plot in log scale.")
            axes[1].plot([], [], label='Damage Error')  # Para leyenda
        # --- TEXTO TRADUCIDO ---
        axes[1].set_title('Damage Data')
        axes[1].set_xlabel('Window Index (Time)')

    else:
        # --- TEXTO TRADUCIDO ---
        axes[1].set_title('Damage Data (Not available)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "error_timeseries.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Error time series graph saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving time series graph {filename}: {e}")
    plt.close()


def plot_error_heatmap(losses_healthy_per_sensor, losses_damage_per_sensor, output_dir):
    """
    Genera mapas de calor que muestran el error por sensor a lo largo del tiempo.
    losses_healthy_per_sensor, losses_damage_per_sensor: shape (n_samples, n_nodes)
    """
    local_logger_plot.info("Generating error heatmaps...")

    # Manejar el caso donde uno de los arrays está vacío
    data_list = []
    has_healthy = losses_healthy_per_sensor is not None and losses_healthy_per_sensor.size > 0
    has_damage = losses_damage_per_sensor is not None and losses_damage_per_sensor.size > 0

    if has_healthy:
        # Filtrar valores infinitos/NaN antes de calcular min/max
        valid_healthy = losses_healthy_per_sensor[np.isfinite(losses_healthy_per_sensor)]
        if len(valid_healthy[valid_healthy > 0]) > 0:
            data_list.append(valid_healthy[valid_healthy > 0])

    if has_damage:
        # Filtrar valores infinitos/NaN antes de calcular min/max
        valid_damage = losses_damage_per_sensor[np.isfinite(losses_damage_per_sensor)]
        if len(valid_damage[valid_damage > 0]) > 0:
            data_list.append(valid_damage[valid_damage > 0])

    if not data_list:
        local_logger_plot.warning("No finite positive data available for heatmap plotting.")
        return

    all_data_positive = np.concatenate(data_list)
    if all_data_positive.size == 0:
        local_logger_plot.warning("No finite positive data available for heatmap plotting.")
        return

    vmin = np.min(all_data_positive)
    vmax = np.max(all_data_positive)

    # Asegurar vmin > 0 para LogNorm
    vmin = max(vmin, 1e-12)
    # Considerar ajustar vmax si es extremadamente alto debido a outliers
    vmax_plot = np.percentile(all_data_positive, 99.5)  # Usar percentil 99.5 como límite superior

    # Usar LogNorm para mejor visualización si hay outliers
    # Manejar caso donde vmin y vmax_plot son iguales o muy cercanos
    if vmin >= vmax_plot:
        # Si vmin y vmax son casi iguales, usar escala lineal simple podría ser mejor
        norm = plt.Normalize(vmin=vmin * 0.9, vmax=vmin * 1.1)
        cbar_label = 'MSE (Linear Scale)'
        local_logger_plot.warning(f"Heatmap vmin ~= vmax_plot ({vmin:.2e}). Using linear scale near value.")
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax_plot)  # Usar vmax_plot
        cbar_label = 'MSE (Log Scale)'

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    # --- TEXTO TRADUCIDO ---
    fig.suptitle('Error Heatmap (Sensor vs. Time)', fontsize=16)

    # Determinar etiquetas de sensor del conjunto que no esté vacío
    num_sensors = losses_healthy_per_sensor.shape[1] if has_healthy else losses_damage_per_sensor.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]

    # Colormaps
    healthy_cmap = "Blues"
    damage_cmap = "Reds"

    if has_healthy:
        # Transponer para que los sensores queden en el eje Y
        # --- TEXTO TRADUCIDO ---
        sns.heatmap(losses_healthy_per_sensor.T, ax=axes[0], cmap=healthy_cmap, norm=norm,
                    cbar_kws={'label': cbar_label}, xticklabels=10000)  # Mostrar menos etiquetas X
        axes[0].set_title('Healthy Data (Baseline)')
        axes[0].set_yticklabels(sensor_labels, rotation=0)
        axes[0].set_ylabel('Sensor')
    else:
        # --- TEXTO TRADUCIDO ---
        axes[0].set_title('Healthy Data (Not available)')

    if has_damage:
        # --- TEXTO TRADUCIDO ---
        sns.heatmap(losses_damage_per_sensor.T, ax=axes[1], cmap=damage_cmap, norm=norm,
                    cbar_kws={'label': cbar_label}, xticklabels=10000)  # Mostrar menos etiquetas X
        axes[1].set_title('Damage Data')
        axes[1].set_yticklabels(sensor_labels, rotation=0)
        axes[1].set_ylabel('Sensor')
        axes[1].set_xlabel('Window Index (Approx. Time)')  # Cambiar label
    else:
        # --- TEXTO TRADUCIDO ---
        axes[1].set_title('Damage Data (Not available)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "error_heatmap_sensor_vs_time.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Error heatmap saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving heatmap {filename}: {e}")
    plt.close()


# --- GRÁFICO ESTADÍSTICO DETALLADO ---
# (Modificado para usar Violinplots + Stripplots con puntos negros semitransparentes)
def plot_error_statistics_per_sensor(losses_healthy_per_sensor, losses_damage_per_sensor, output_dir):
    """
    Genera gráficos de violín+strip comparativos de la distribución de errores por sensor
    para datos sanos vs. datos con daño.
    losses_healthy_per_sensor, losses_damage_per_sensor: shape (n_samples, n_nodes)
    """
    if losses_healthy_per_sensor is None or losses_damage_per_sensor is None or \
            losses_healthy_per_sensor.size == 0 or losses_damage_per_sensor.size == 0 or \
            losses_healthy_per_sensor.shape[1] != losses_damage_per_sensor.shape[1]:
        local_logger_plot.warning("Insufficient per-sensor loss data for statistical graph.")
        return

    local_logger_plot.info(
        "Generating statistical error graph per sensor (Violinplots + Stripplots)...")  # Mensaje actualizado
    num_sensors = losses_healthy_per_sensor.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]

    # --- Preparar datos para Seaborn ---
    data_list_healthy = []
    data_list_damage = []

    for i in range(num_sensors):
        sensor_name = sensor_labels[i]

        # Datos Sanos
        healthy_positive = losses_healthy_per_sensor[:, i][losses_healthy_per_sensor[:, i] > 0]
        if len(healthy_positive) > 0:
            for loss_val in healthy_positive:
                data_list_healthy.append({'Sensor': sensor_name, 'MSE': loss_val})  # No necesita color aquí
        else:
            data_list_healthy.append({'Sensor': sensor_name, 'MSE': 1e-12})  # Placeholder

        # Datos con Daño
        damage_positive = losses_damage_per_sensor[:, i][losses_damage_per_sensor[:, i] > 0]
        if len(damage_positive) > 0:
            for loss_val in damage_positive:
                data_list_damage.append({'Sensor': sensor_name, 'MSE': loss_val})  # No necesita color aquí
        else:
            data_list_damage.append({'Sensor': sensor_name, 'MSE': 1e-12})  # Placeholder

    df_healthy = pd.DataFrame(data_list_healthy)
    df_damage = pd.DataFrame(data_list_damage)
    # --- FIN DE PREPARACIÓN ---

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)  # Compartir eje Y
    # --- TEXTO TRADUCIDO ---
    fig.suptitle('Statistical Distribution of Reconstruction Error per Sensor', fontsize=16)

    # --- Panel Datos Sanos: Violinplot + Stripplot --- MODIFICADO ---
    if not df_healthy.empty and df_healthy['MSE'].max() > 1e-11:  # Chequear datos válidos
        sns.violinplot(x='Sensor', y='MSE', data=df_healthy, ax=axes[0],
                       palette=sensor_colors_mapped,  # Usar colores mapeados para violines
                       inner='quartile', cut=0, scale='width', linewidth=1.5)  # Linea violin más gruesa
        # *** CAMBIO DE COLOR Y AJUSTES ***
        sns.stripplot(x='Sensor', y='MSE', data=df_healthy, ax=axes[0],
                      color='black', alpha=0.1, size=2.5, jitter=0.3)  # Puntos negros semitransparentes
        # --- TEXTO TRADUCIDO ---
        axes[0].set_title('Healthy Data (Baseline - with points)')
        axes[0].set_ylabel('Mean Squared Error (MSE) - Log Scale')
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle=':')
        # Ajustar límites Y
        # Filtrar inf/nan antes de min/max
        valid_mse_h = df_healthy['MSE'][np.isfinite(df_healthy['MSE'])]
        if len(valid_mse_h) > 0:
            min_mse_plot = max(valid_mse_h.min() * 0.8, 1e-9)
            # Ajustar límite superior basado en percentil para ignorar outliers extremos
            max_mse_plot = np.percentile(valid_mse_h, 99.8) * 1.5
            axes[0].set_ylim(bottom=min_mse_plot, top=max_mse_plot)
    else:
        axes[0].set_title('Healthy Data (No positive data or too small)')

    # --- Panel Datos con Daño: Violinplot + Stripplot --- MODIFICADO ---
    if not df_damage.empty and df_damage['MSE'].max() > 1e-11:  # Chequear datos válidos
        sns.violinplot(x='Sensor', y='MSE', data=df_damage, ax=axes[1],
                       palette=sensor_colors_mapped,  # Usar colores mapeados para violines
                       inner='quartile', cut=0, scale='width', linewidth=1.5)  # Linea violin más gruesa
        # *** CAMBIO DE COLOR Y AJUSTES ***
        sns.stripplot(x='Sensor', y='MSE', data=df_damage, ax=axes[1],
                      color='black', alpha=0.1, size=2.5, jitter=0.3)  # Puntos negros semitransparentes
        # --- TEXTO TRADUCIDO ---
        axes[1].set_title('Damage Data (with points)')
        axes[1].set_yscale('log')  # Mantener escala log
        axes[1].grid(True, linestyle=':')
        # Ajustar límites Y (compartido con el eje izquierdo si sharey=True)
        if not axes[0].get_shared_y_axes().joined(axes[0], axes[1]):
            valid_mse_d = df_damage['MSE'][np.isfinite(df_damage['MSE'])]
            if len(valid_mse_d) > 0:
                min_mse_plot_d = max(valid_mse_d.min() * 0.8, 1e-9)
                max_mse_plot_d = np.percentile(valid_mse_d, 99.8) * 1.5
                axes[1].set_ylim(bottom=min_mse_plot_d, top=max_mse_plot_d)
    else:
        axes[1].set_title('Damage Data (No positive data or too small)')
    # --- FIN DE MODIFICACIÓN ---

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Cambiar nombre de archivo para reflejar el nuevo tipo de gráfico
    filename = os.path.join(output_dir, "error_statistics_per_sensor_violinplot_with_points.png")  # Nombre actualizado
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Statistical error violinplot with points saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving statistical violinplot: {e}")
    plt.close(fig)


# --- ************ NUEVAS FUNCIONES DE PLOTEO (PAPER FIGS. 9 & 10) ************ ---

def plot_evaluation_scatters_and_boxes(healthy_losses_global, healthy_ssim_global, damage_losses_global,
                                       damage_ssim_global, output_dir):
    """
    Replica la estructura de la Figura 9 del paper:
    (a) Scatter Error vs. Samples
    (b) Scatter SSIM vs. Samples
    (c) Violinplot+Stripplot Error vs. Scenario  <--- MODIFICADO (Escala Log Y, puntos negros)
    (d) Violinplot+Stripplot SSIM vs. Scenario  <--- MODIFICADO (puntos negros)

    Usa "Healthy" y "Damage" como los dos escenarios.
    """
    local_logger_plot.info("Generating global evaluation scatter and combined distribution plots (Fig. 9 style)...")

    # Comprobar si alguno de los arrays necesarios es None o está vacío
    if healthy_losses_global is None or healthy_ssim_global is None or \
            damage_losses_global is None or damage_ssim_global is None or \
            healthy_losses_global.size == 0 or damage_losses_global.size == 0:
        local_logger_plot.warning("Missing global MSE or SSIM data for Healthy or Damage, skipping Fig. 9 plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Overall Evaluation Results (Healthy vs. Damage)', fontsize=16)

    num_healthy = len(healthy_losses_global)
    num_damage = len(damage_losses_global)

    # --- (a) Scatter Error vs. Samples ---
    indices_healthy = np.arange(num_healthy)
    indices_damage = np.arange(num_healthy, num_healthy + num_damage)

    axes[0, 0].scatter(indices_healthy, healthy_losses_global, label='Healthy', alpha=0.5, s=10,
                       color=palette_scenario[0])  # Usar paleta
    axes[0, 0].scatter(indices_damage, damage_losses_global, label='Damage', alpha=0.5, s=10,
                       color=palette_scenario[1])  # Usar paleta
    axes[0, 0].set_title('(a) Reconstruction Error (MSE) vs. Samples')
    axes[0, 0].set_xlabel('The number of samples')
    axes[0, 0].set_ylabel('Error (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=':')
    # Considerar escala log si los picos son muy altos
    # Filtrar inf y nan antes de calcular max/min
    valid_losses = np.concatenate([healthy_losses_global, damage_losses_global])
    valid_losses = valid_losses[np.isfinite(valid_losses)]
    if len(valid_losses) > 0:
        max_error = np.max(valid_losses)
        min_error = max(np.min(valid_losses), 1e-9)  # Evitar 0
        if max_error / min_error > 1000:  # Si hay más de 3 órdenes de magnitud
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_ylabel('Error (MSE) - Log Scale')

    # --- (b) Scatter SSIM vs. Samples ---
    axes[0, 1].scatter(indices_healthy, healthy_ssim_global, label='Healthy', alpha=0.5, s=10,
                       color=palette_scenario[0])  # Usar paleta
    axes[0, 1].scatter(indices_damage, damage_ssim_global, label='Damage', alpha=0.5, s=10,
                       color=palette_scenario[1])  # Usar paleta
    axes[0, 1].set_title('(b) Structural Similarity (SSIM) vs. Samples')
    axes[0, 1].set_xlabel('The number of samples')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=':')
    axes[0, 1].set_ylim(-0.05, 1.05)  # Ajustar límites para SSIM

    # --- (c) Violinplot + Stripplot Error vs. Scenario --- MODIFICADO (Escala Log Y, Puntos Negros) ---
    # Crear DataFrame para ploteo con Seaborn
    healthy_df = pd.DataFrame({'MSE': healthy_losses_global, 'SSIM': healthy_ssim_global, 'Scenario': 'Healthy'})
    damage_df = pd.DataFrame({'MSE': damage_losses_global, 'SSIM': damage_ssim_global, 'Scenario': 'Damage'})
    full_df = pd.concat([healthy_df, damage_df], ignore_index=True)

    # Filtrar valores no positivos para escala log
    full_df_pos_mse = full_df[full_df['MSE'] > 0].copy()
    if full_df_pos_mse.empty or full_df_pos_mse['MSE'].max() <= 1e-11:
        axes[1, 0].set_title('(c) Error Distribution (No positive data or too small)')
    else:
        # Violin plot primero (detrás)
        sns.violinplot(x='Scenario', y='MSE', data=full_df_pos_mse, ax=axes[1, 0],
                       palette=palette_scenario,  # Usar paleta Healthy/Damage
                       inner='quartile', cut=0, scale='width', linewidth=1.5)  # Línea más gruesa
        # Strip plot encima (delante) con puntos negros semitransparentes
        # *** CAMBIO DE COLOR Y AJUSTES ***
        sns.stripplot(x='Scenario', y='MSE', data=full_df_pos_mse, ax=axes[1, 0],
                      color='black', alpha=0.1, size=2.5, jitter=0.3)  # Puntos negros semitransparentes

        axes[1, 0].set_title('(c) Error Distribution by Scenario (with points)')
        axes[1, 0].set_xlabel('Scenarios')
        axes[1, 0].set_ylabel('Error (MSE) - Log Scale')  # Etiqueta actualizada
        axes[1, 0].set_yscale('log')  # Aplicar escala logarítmica
        axes[1, 0].grid(True, linestyle=':')
        # Ajustar límite inferior si es necesario para log scale
        # Filtrar inf/nan antes de min/max
        valid_mse_c = full_df_pos_mse['MSE'][np.isfinite(full_df_pos_mse['MSE'])]
        if len(valid_mse_c) > 0:
            min_mse_plot = max(valid_mse_c.min() * 0.8, 1e-9)
            # Ajustar límite superior basado en percentil para ignorar outliers extremos
            max_mse_plot = np.percentile(valid_mse_c, 99.8) * 1.5  # Usar percentil alto
            axes[1, 0].set_ylim(bottom=min_mse_plot, top=max_mse_plot)

            # --- (d) Violinplot + Stripplot SSIM vs. Scenario --- MODIFICADO (Puntos Negros) ---
    # Violin plot primero
    sns.violinplot(x='Scenario', y='SSIM', data=full_df, ax=axes[1, 1],
                   palette=palette_scenario,  # Usar paleta Healthy/Damage
                   inner='quartile', cut=0, scale='width', linewidth=1.5)  # Línea más gruesa
    # Strip plot encima
    # *** CAMBIO DE COLOR Y AJUSTES ***
    sns.stripplot(x='Scenario', y='SSIM', data=full_df, ax=axes[1, 1],
                  color='black', alpha=0.1, size=2.5, jitter=0.3)  # Puntos negros semitransparentes

    axes[1, 1].set_title('(d) SSIM Distribution by Scenario (with points)')
    axes[1, 1].set_xlabel('Scenarios')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].grid(True, linestyle=':')
    axes[1, 1].set_ylim(bottom=-0.05, top=1.05)  # SSIM suele estar entre 0 y 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "evaluation_scatters_and_distributions_fig9_style.png")  # Nombre actualizado
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Fig. 9 style plot saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving Fig. 9 style plot {filename}: {e}")
    plt.close(fig)


def plot_ssim_vs_error_cluster(healthy_losses_per_sensor, healthy_ssim_per_sensor, damage_losses_per_sensor,
                               damage_ssim_per_sensor, num_nodes, output_dir):
    """
    Replica la estructura de la Figura 10 y 11 del paper (un gráfico por sensor):
    (a) Scatter SSIM vs. Error, coloreado por escenario (Healthy vs. Damage)
    (b) Gráfico de Tendencia (promedio de Healthy vs. promedio de Damage)
    """
    local_logger_plot.info("Generating SSIM vs. Error cluster plots (Fig. 10 style)...")

    # Comprobar si alguno de los arrays necesarios es None o está vacío
    if healthy_losses_per_sensor is None or healthy_ssim_per_sensor is None or \
            damage_losses_per_sensor is None or damage_ssim_per_sensor is None or \
            healthy_losses_per_sensor.size == 0 or damage_losses_per_sensor.size == 0:
        local_logger_plot.warning("Missing per-sensor MSE or SSIM data for Healthy or Damage, skipping Fig. 10 plots.")
        return

    for i in range(num_nodes):
        sensor_id = i + 1
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Progressive Damage Detection Results of Sensor {sensor_id}', fontsize=16)

        # --- (a) Scatter Plot de Clusters ---
        h_loss = healthy_losses_per_sensor[:, i]
        h_ssim = healthy_ssim_per_sensor[:, i]
        d_loss = damage_losses_per_sensor[:, i]
        d_ssim = damage_ssim_per_sensor[:, i]

        axes[0].scatter(h_loss, h_ssim, label='Scenario: Healthy', color=palette_scenario[0], alpha=0.3,
                        s=15)  # Usar paleta
        axes[0].scatter(d_loss, d_ssim, label='Scenario: Damage', color=palette_scenario[1], alpha=0.3,
                        s=15)  # Usar paleta

        axes[0].set_title('(a) Cluster Analysis')
        axes[0].set_xlabel('Error (MSE)')
        axes[0].set_ylabel('SSIM')
        axes[0].legend()
        axes[0].grid(True, linestyle=':')
        # Considerar escala log para el eje X (Error) si los rangos son muy amplios
        all_losses_sensor = np.concatenate([h_loss, d_loss])
        all_losses_sensor = all_losses_sensor[np.isfinite(all_losses_sensor)]
        if len(all_losses_sensor) > 0:
            max_loss_s = np.max(all_losses_sensor)
            min_loss_s = max(np.min(all_losses_sensor), 1e-9)
            if max_loss_s / min_loss_s > 100:
                axes[0].set_xscale('log')
                axes[0].set_xlabel('Error (MSE) - Log Scale')

        # Ajustar límites Y para SSIM
        axes[0].set_ylim(-0.05, 1.05)

        # --- (b) Gráfico de Tendencia (Promedios) ---
        h_loss_mean = np.mean(h_loss)
        h_ssim_mean = np.mean(h_ssim)
        d_loss_mean = np.mean(d_loss)
        d_ssim_mean = np.mean(d_ssim)

        # Usar colores de la paleta para los puntos promedio
        axes[1].plot([h_loss_mean, d_loss_mean], [h_ssim_mean, d_ssim_mean], marker='o', markersize=10, linestyle='-',
                     linewidth=2, color='darkgrey')  # Linea gris oscuro
        axes[1].scatter(h_loss_mean, h_ssim_mean, label=f'Healthy Avg (E: {h_loss_mean:.2f}, S: {h_ssim_mean:.2f})',
                        color=palette_scenario[0], s=120, zorder=5, edgecolor='black')  # Puntos más grandes
        axes[1].scatter(d_loss_mean, d_ssim_mean, label=f'Damage Avg (E: {d_loss_mean:.2f}, S: {d_ssim_mean:.2f})',
                        color=palette_scenario[1], s=120, zorder=5, edgecolor='black')  # Puntos más grandes

        axes[1].set_title('(b) Trend of Average Values')
        axes[1].set_xlabel('Error (MSE)')
        axes[1].set_ylabel('SSIM')
        axes[1].legend()
        axes[1].grid(True, linestyle=':')
        # Ajustar límites Y para SSIM
        axes[1].set_ylim(-0.05, 1.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = os.path.join(output_dir, f"ssim_vs_error_cluster_sensor_{sensor_id}.png")
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            local_logger_plot.info(f"Fig. 10 style plot for sensor {sensor_id} saved in: {filename}")
        except Exception as e:
            local_logger_plot.error(f"Error saving Fig. 10 style plot {filename}: {e}")
        plt.close(fig)


# --- ************ NUEVA FUNCION DE PLOTEO (PAPER FIG. 7) ************ ---
def plot_training_history(log_file_path, output_dir):
    """
    Parsea el archivo training_log.txt y genera un gráfico de las curvas de pérdida
    de entrenamiento y validación por época, similar a la Fig. 7 del paper.
    """
    local_logger_plot.info("Generating training history plot (Fig. 7 style)...")

    epochs = []
    train_losses = []
    val_losses = []

    # Patrón regex para extraer los datos de las líneas de log relevantes
    # Ej: Epoch 1/50 -> Train Loss: 0.737716, Val Loss: 0.637364
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ -> Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+)")  # Mejorado para notación científica

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:  # Añadir encoding
            for line in f:
                match = epoch_pattern.search(line)
                if match:
                    try:
                        epoch = int(match.group(1))
                        train_loss = float(match.group(2))
                        val_loss = float(match.group(3))
                        epochs.append(epoch)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                    except ValueError:
                        local_logger_plot.warning(
                            f"Could not parse loss values in line: {line.strip()}. Skipping line.")
                        continue
    except FileNotFoundError:
        local_logger_plot.error(f"Training log file not found at: {log_file_path}. Skipping training history plot.")
        return
    except Exception as e:
        local_logger_plot.error(
            f"Error parsing training log file {log_file_path}: {e}. Skipping training history plot.")
        return

    if not epochs:
        local_logger_plot.warning("No epoch data found in training log file. Skipping training history plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-', color='#e41a1c',
             markersize=4)  # Rojo (Set1)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--', color='#377eb8',
             markersize=5)  # Azul (Set1)

    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle=':')

    # Determinar si usar escala logarítmica basado en el rango de pérdidas
    all_losses = train_losses + val_losses
    # Filtrar posibles inf/nan antes de min/max
    all_losses = [l for l in all_losses if np.isfinite(l)]
    if not all_losses:  # Si no quedan pérdidas válidas
        min_loss = 1e-6
        max_loss = 1.0
    else:
        min_loss = min(all_losses)
        max_loss = max(all_losses)

    # Poner escala log si hay > 2 órdenes de magnitud O si el mínimo es muy pequeño
    if (max_loss / max(min_loss, 1e-9) > 100) or min_loss < 0.01:
        plt.yscale('log')
        plt.ylabel('Loss (MSE) - Log Scale')  # Actualizar etiqueta Y
        # Ajustar límite inferior para log, asegurando que no sea cero o negativo
        plot_min_y = max(min_loss * 0.8, 1e-9)  # Bajar un poco pero no demasiado
        plt.ylim(bottom=plot_min_y)
    else:
        plt.ylim(bottom=0)  # Escala lineal empieza en 0

    filename = os.path.join(output_dir, "training_history_loss_curves.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Training history plot saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving training history plot {filename}: {e}")
    plt.close()


# --- ************ NUEVA FUNCION DE PLOTEO (GRAFO) ************ ---
def plot_bridge_graph(edge_index_numpy, output_dir, node_values=None, cmap=plt.cm.viridis, title_suffix="",
                      filename_suffix="", vmin=None, vmax=None):
    """
    Dibuja la estructura del grafo del puente (sensores y conexiones).
    Opcionalmente colorea los nodos según node_values.
    edge_index_numpy: Array numpy con las conexiones (shape [2, num_edges]).
    node_values: Lista o array con valores para colorear los nodos (len debe ser 5).
    cmap: Colormap a usar si se proveen node_values.
    title_suffix: Sufijo para añadir al título del gráfico.
    filename_suffix: Sufijo para añadir al nombre del archivo.
    vmin, vmax: Límites para la barra de color.
    """
    local_logger_plot.info(f"Generating bridge graph structure plot ({title_suffix})...")

    # Crear un grafo de NetworkX a partir de las aristas
    G = nx.Graph()
    # Los nodos van de 0 a 4. Los añadimos explícitamente.
    G.add_nodes_from(range(5))
    # Añadir aristas.
    edges = set()
    for i in range(edge_index_numpy.shape[1]):
        u, v = sorted(edge_index_numpy[:, i])  # Ordenar para evitar duplicados A->B, B->A
        edges.add((u, v))
    G.add_edges_from(list(edges))

    # Definir posiciones manualmente para claridad
    pos = {0: (0, 0), 1: (1, 0.1), 2: (1, -0.1), 3: (2, 0.1), 4: (2, -0.1)}

    # Mapear etiquetas de nodo de 0-4 a 1-5 para mostrar
    labels = {i: str(i + 1) for i in G.nodes()}

    fig, ax = plt.subplots(figsize=(8, 4))  # Obtener ax para la barra de color

    # Determinar colores de nodo
    node_color_param = 'skyblue'  # Default
    if node_values is not None and len(node_values) == len(G.nodes()):
        node_color_param = node_values
        # Ajustar vmin/vmax si no se proporcionan
        if vmin is None: vmin = np.min(node_values)
        if vmax is None: vmax = np.max(node_values)
        # Asegurar que vmin y vmax no sean iguales
        if vmin == vmax:
            vmin -= 0.1 * abs(vmin) + 1e-6
            vmax += 0.1 * abs(vmax) + 1e-6

        norm = Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        node_color_param = [mapper.to_rgba(val) for val in node_values]  # Convertir valores a colores RGBA

        # Añadir barra de color
        cbar = fig.colorbar(mapper, ax=ax, shrink=0.7, aspect=10)
        cbar.set_label('Average MSE')  # Etiqueta barra de color

    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels, node_color=node_color_param,
            node_size=800, font_size=12, font_color='white', font_weight='bold', edge_color='gray',
            width=1.5)  # Ajustes visuales

    ax.set_title(f"Bridge Sensor Graph Structure {title_suffix}")

    filename = os.path.join(output_dir, f"bridge_graph{filename_suffix}.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Bridge graph plot saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving bridge graph plot {filename}: {e}")
    plt.close(fig)


# --- ************ NUEVA FUNCION HEATMAP SOBRE GRAFO ************ ---
def plot_graph_heatmap_comparison(edge_index_numpy, healthy_values, damage_values, output_dir):
    """
    Dibuja dos grafos lado a lado, coloreando los nodos según los valores
    promedio de MSE (o similar) para Healthy y Damage.
    """
    local_logger_plot.info("Generating graph heatmap comparison plot...")

    if healthy_values is None or damage_values is None or len(healthy_values) != 5 or len(damage_values) != 5:
        local_logger_plot.warning("Invalid or missing average sensor values for graph heatmap. Skipping plot.")
        return

    # Filtrar inf/nan
    healthy_values = np.nan_to_num(healthy_values, nan=0.0, posinf=np.finfo(np.float32).max,
                                   neginf=np.finfo(np.float32).min)
    damage_values = np.nan_to_num(damage_values, nan=0.0, posinf=np.finfo(np.float32).max,
                                  neginf=np.finfo(np.float32).min)

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # Más ancho para 2 grafos
    fig.suptitle('Average MSE on Graph Nodes', fontsize=16)

    # --- Grafo Izquierdo: Healthy ---
    cmap_healthy = plt.cm.Blues  # Mapa de color para sano
    # Encontrar min/max global para la barra de color
    all_values = np.concatenate([healthy_values, damage_values])
    vmin = np.min(all_values)
    vmax = np.max(all_values)
    # Asegurar que vmin y vmax no sean iguales
    if vmin == vmax:
        vmin -= 0.1 * abs(vmin) + 1e-6 if vmin != 0 else 1e-6
        vmax += 0.1 * abs(vmax) + 1e-6 if vmax != 0 else 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper_healthy = cm.ScalarMappable(norm=norm, cmap=cmap_healthy)
    node_colors_healthy = [mapper_healthy.to_rgba(val) for val in healthy_values]

    # Dibujar grafo
    G = nx.Graph()
    G.add_nodes_from(range(5))
    edges = set()
    for i in range(edge_index_numpy.shape[1]):
        u, v = sorted(edge_index_numpy[:, i])
        edges.add((u, v))
    G.add_edges_from(list(edges))
    pos = {0: (0, 0), 1: (1, 0.1), 2: (1, -0.1), 3: (2, 0.1), 4: (2, -0.1)}
    labels = {i: str(i + 1) for i in G.nodes()}

    nx.draw(G, pos, ax=axes[0], with_labels=True, labels=labels, node_color=node_colors_healthy,
            node_size=900, font_size=12, font_color='black', font_weight='bold', edge_color='darkgray', width=2)
    axes[0].set_title("Healthy Data")
    # Añadir barra de color
    cbar_h = fig.colorbar(mapper_healthy, ax=axes[0], shrink=0.8, aspect=15)
    cbar_h.set_label('Average MSE')

    # --- Grafo Derecho: Damage ---
    cmap_damage = plt.cm.Reds  # Mapa de color para daño
    mapper_damage = cm.ScalarMappable(norm=norm, cmap=cmap_damage)  # Usar la misma norma
    node_colors_damage = [mapper_damage.to_rgba(val) for val in damage_values]

    nx.draw(G, pos, ax=axes[1], with_labels=True, labels=labels, node_color=node_colors_damage,
            node_size=900, font_size=12, font_color='black', font_weight='bold', edge_color='darkgray', width=2)
    axes[1].set_title("Damage Data")
    # Añadir barra de color
    cbar_d = fig.colorbar(mapper_damage, ax=axes[1], shrink=0.8, aspect=15)
    cbar_d.set_label('Average MSE')

    # Guardar figura
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajustar layout
    filename = os.path.join(output_dir, "graph_heatmap_comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Graph heatmap comparison plot saved in: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error saving graph heatmap comparison plot {filename}: {e}")
    plt.close(fig)


# --- ************ FIN DE NUEVA FUNCION HEATMAP SOBRE GRAFO ************ ---


# --- FUNCIÓN PRINCIPAL DE INFERENCIA ---

def run_inference_and_plot(model_dir, base_healthy_dir, damage_data_dir=None):
    """
    Función principal que orquesta la carga del modelo, la inferencia y el ploteo de resultados.
    """
    file_handler = None
    try:
        # --- Configuración de directorios y logging de archivo ---
        inference_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(model_dir, f"inference_results_{inference_timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # Añadir el FileHandler al logger raíz
        log_file_path_inference = os.path.join(results_dir, 'inference.log')  # Log de inferencia
        file_handler = logging.FileHandler(log_file_path_inference, encoding='utf-8')  # Añadir encoding
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        logger.info(f"--- Starting Inference using model from: {model_dir} ---")
        logger.info(f"Healthy data directory: {base_healthy_dir}")
        logger.info(f"Damage data directory: {damage_data_dir}")
        logger.info(f"Results will be saved in: {results_dir}")

        # --- Carga de hiperparámetros y escalador ---
        params_path = os.path.join(model_dir, 'hyperparameters.json')
        scaler_path = os.path.join(model_dir, 'scaler.gz')
        # --- NUEVO: Ruta al log de entrenamiento ---
        training_log_path = os.path.join(model_dir, 'training_log.txt')

        # Validar existencia de archivos necesarios
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Hyperparameters file not found: {params_path}")
        if not os.path.exists(scaler_path):
            # Intentar buscar el archivo desempaquetado si gz falla
            scaler_path_alt = os.path.join(model_dir, 'scaler')
            if os.path.exists(scaler_path_alt):
                scaler_path = scaler_path_alt
                logger.info("Found unpacked scaler file.")
            else:
                raise FileNotFoundError(f"Scaler file not found (tried .gz and unpacked): {scaler_path}")

        model_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        # El log de entrenamiento es opcional para la inferencia, pero necesario para el nuevo gráfico
        if not os.path.exists(training_log_path):
            logger.warning(f"Training log file not found: {training_log_path}. Training history plot will be skipped.")
            training_log_path = None  # Poner a None si no existe

        with open(params_path, 'r', encoding='utf-8') as f:  # Añadir encoding
            params = json.load(f)
        scaler = joblib.load(scaler_path)
        logger.info("Hyperparameters and scaler loaded.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # --- Definir y Plotear Grafo --- MODIFICADO ---
        edge_index = define_bridge_graph().to(device)
        # Dibujar solo la estructura base una vez
        plot_bridge_graph(edge_index.cpu().numpy(), results_dir, title_suffix="(Structure Only)",
                          filename_suffix="_structure")
        num_nodes = 5  # Basado en define_bridge_graph
        sensor_ids = list(range(1, num_nodes + 1))

        # --- Inicialización y carga del modelo ---
        model = SpatioTemporalAutoencoder(
            num_nodes=num_nodes,
            num_features=1,
            window_size=params['window_size'],
            gnn_hidden=params.get('gnn_hidden', 32),
            gnn_out=params.get('gnn_out', 16),
            rnn_hidden=params.get('rnn_hidden', 64)
        ).to(device)

        # CORRECCIÓN: Añadir weights_only=True por seguridad y para eliminar warning
        # Usar map_location=device asegura compatibilidad entre CPU/GPU
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        logger.info("Model loaded and in evaluation mode.")

        # --- Carga y procesamiento de datos SANOS ---
        logger.info("Loading and processing healthy data...")
        all_healthy_data_raw, max_len_healthy = load_data_from_dir(base_healthy_dir, num_nodes)

        if all_healthy_data_raw is None:
            raise ValueError(
                f"Fatal Error: Could not load healthy data from {base_healthy_dir}. Check path and files.")

        original_shape_healthy = all_healthy_data_raw.shape
        data_healthy_reshaped = all_healthy_data_raw.reshape(-1, 1)
        scaled_data_healthy_flat = scaler.transform(data_healthy_reshaped)
        healthy_data_scaled = scaled_data_healthy_flat.reshape(original_shape_healthy)
        logger.info(f"Healthy data rescaled from {data_healthy_reshaped.shape} to {healthy_data_scaled.shape}")

        healthy_data_dict = {
            sid: healthy_data_scaled[:, i] for i, sid in enumerate(sensor_ids)
        }

        healthy_dataset = SpatioTemporalWindowDataset(
            healthy_data_dict,
            window_size=params['window_size'],
            stride=params['stride'],
            sensor_ids=sensor_ids
        )
        if len(healthy_dataset) == 0:
            raise ValueError("Healthy dataset has insufficient data to create even one window.")
        healthy_loader = DataLoader(healthy_dataset, batch_size=params['batch_size'], shuffle=False)

        # --- Inferencia en datos SANOS ---
        logger.info("Performing inference on healthy data...")
        # --- MODIFICADO: Recibir ssim ---
        healthy_originals, healthy_reconstructions, healthy_losses, healthy_losses_per_sensor, healthy_ssim_per_sensor = perform_inference(
            model, healthy_loader, device, edge_index
        )
        if healthy_losses_per_sensor.size == 0:
            raise ValueError("Inference on healthy data failed, no results obtained.")

        healthy_mse_mean_per_sensor = np.mean(healthy_losses_per_sensor, axis=0)
        healthy_losses_global = np.mean(healthy_losses_per_sensor, axis=1)  # Promedio de sensores -> (n_samples,)
        healthy_ssim_global = np.mean(healthy_ssim_per_sensor, axis=1)  # Promedio de sensores -> (n_samples,)

        # --- Carga e inferencia en datos con DAÑO (si se proporcionan) ---
        damage_originals = None
        damage_reconstructions = None
        damage_losses = None
        damage_losses_per_sensor = None
        damage_mse_mean_per_sensor = None
        damage_ssim_per_sensor = None
        damage_losses_global = None
        damage_ssim_global = None

        if damage_data_dir:
            logger.info("Loading and processing damage data...")
            all_damage_data_raw, _ = load_data_from_dir(damage_data_dir, num_nodes, max_len=max_len_healthy)

            if all_damage_data_raw is not None:
                original_shape_damage = all_damage_data_raw.shape
                data_damage_reshaped = all_damage_data_raw.reshape(-1, 1)
                scaled_data_damage_flat = scaler.transform(data_damage_reshaped)
                damage_data_scaled = scaled_data_damage_flat.reshape(original_shape_damage)
                logger.info(f"Damage data rescaled from {data_damage_reshaped.shape} to {damage_data_scaled.shape}")

                damage_data_dict = {
                    sid: damage_data_scaled[:, i] for i, sid in enumerate(sensor_ids)
                }

                damage_dataset = SpatioTemporalWindowDataset(
                    damage_data_dict,
                    window_size=params['window_size'],
                    stride=params['stride'],
                    sensor_ids=sensor_ids
                )
                if len(damage_dataset) > 0:
                    damage_loader = DataLoader(damage_dataset, batch_size=params['batch_size'], shuffle=False)
                    logger.info("Performing inference on damage data...")

                    # --- MODIFICADO: Recibir ssim ---
                    damage_originals, damage_reconstructions, damage_losses, damage_losses_per_sensor, damage_ssim_per_sensor = perform_inference(
                        model, damage_loader, device, edge_index
                    )
                    if damage_losses_per_sensor.size > 0:
                        damage_mse_mean_per_sensor = np.mean(damage_losses_per_sensor, axis=0)
                        damage_losses_global = np.mean(damage_losses_per_sensor, axis=1)  # (n_samples,)
                        damage_ssim_global = np.mean(damage_ssim_per_sensor, axis=1)  # (n_samples,)
                    else:
                        logger.warning("Inference on damage data produced no results.")
                else:
                    logger.warning("Damage dataset has insufficient data. Damage analysis will be skipped.")
            else:
                logger.warning("Could not load damage data. Damage analysis will be skipped.")
        else:
            logger.info("No damage data directory provided. Skipping damage analysis.")

        # --- PLOTEO (CORREGIDO Y AMPLIADO) ---
        logger.info("Generating result plots...")

        # --- NUEVO: Generar gráfico de historial de entrenamiento si el log existe ---
        if training_log_path:
            plot_training_history(training_log_path, results_dir)

        # --- Gráficos de reconstrucción por sensor ---
        logger.info("Generating per-sensor reconstruction plots (2 samples)...")
        plot_sensor_reconstruction_samples(
            healthy_originals, healthy_reconstructions, scaler, num_nodes, results_dir, "healthy"
        )
        # Solo plotear daño si existe
        if damage_originals is not None and damage_reconstructions is not None:
            plot_sensor_reconstruction_samples(
                damage_originals, damage_reconstructions, scaler, num_nodes, results_dir, "damage"
            )
        else:
            logger.info("Skipping damage reconstruction plots as damage data was not processed.")

        # --- Generar gráficos comparativos y estadísticos (si hay datos de daño) ---
        # Mover las llamadas a gráficos que dependen de 'damage_data' dentro de este if
        if damage_mse_mean_per_sensor is not None and damage_losses_per_sensor is not None and \
                damage_losses is not None and damage_ssim_per_sensor is not None and \
                damage_losses_global is not None and damage_ssim_global is not None:

            logger.info(
                "Generating comparison, statistical, graph heatmap and paper-style plots...")  # Mensaje actualizado

            # --- NUEVO: Generar heatmap sobre grafo ---
            plot_graph_heatmap_comparison(
                edge_index.cpu().numpy(),
                healthy_mse_mean_per_sensor,  # Array de MSE promedio por sensor (sano)
                damage_mse_mean_per_sensor,  # Array de MSE promedio por sensor (daño)
                results_dir
            )

            # --- Gráficos existentes ---
            plot_mse_comparison(
                healthy_mse_mean_per_sensor,
                damage_mse_mean_per_sensor,
                results_dir
            )

            plot_damage_localization(
                damage_losses_per_sensor,  # (n_samples, n_nodes)
                healthy_losses_per_sensor,  # (n_samples, n_nodes)
                results_dir
            )

            plot_error_distribution_kde(
                healthy_losses,  # (n_samples,) - MSE global por ventana
                damage_losses,  # (n_samples,) - MSE global por ventana
                results_dir
            )

            plot_error_timeseries(
                healthy_losses,  # (n_samples,) - MSE global por ventana
                damage_losses,  # (n_samples,) - MSE global por ventana
                results_dir
            )

            plot_error_heatmap(
                healthy_losses_per_sensor,  # (n_samples, n_nodes)
                damage_losses_per_sensor,  # (n_samples, n_nodes)
                results_dir
            )

            plot_error_statistics_per_sensor(
                healthy_losses_per_sensor,  # (n_samples, n_nodes)
                damage_losses_per_sensor,  # (n_samples, n_nodes)
                results_dir
            )

            # --- Gráficos estilo Paper Figs. 9 & 10 ---
            plot_evaluation_scatters_and_boxes(
                healthy_losses_global,  # (n_samples,)
                healthy_ssim_global,  # (n_samples,)
                damage_losses_global,  # (n_samples,)
                damage_ssim_global,  # (n_samples,)
                results_dir
            )

            plot_ssim_vs_error_cluster(
                healthy_losses_per_sensor,  # (n_samples, n_nodes)
                healthy_ssim_per_sensor,  # (n_samples, n_nodes)
                damage_losses_per_sensor,  # (n_samples, n_nodes)
                damage_ssim_per_sensor,  # (n_samples, n_nodes)
                num_nodes,
                results_dir
            )

        else:
            logger.info(
                "No complete damage data processed, skipping comparison, localization, statistical, graph heatmap and paper-style plots.")  # Mensaje actualizado

        logger.info("--- Inference process completed ---")

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")  # Más específico
    except Exception as e:
        logger.error(f"Critical error in run_inference_and_plot: {e}", exc_info=True)
    finally:
        # Buena práctica: remover el file handler para que futuras ejecuciones (en un notebook)
        # puedan añadir uno nuevo sin duplicar.
        if file_handler is not None:
            logger.info("Closing log file.")
            file_handler.close()
            logger.removeHandler(file_handler)


# --- EJECUCIÓN ---
if __name__ == '__main__':
    # Asegúrate de que estas rutas sean correctas
    # DEBES ASEGURARTE QUE 'training_log.txt' ESTÉ DENTRO DE ESTA CARPETA
    trained_model_directory = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756"
    damage_data_directory = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"
    base_healthy_data_directory = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # Validar directorios antes de ejecutar
    if not os.path.isdir(trained_model_directory):
        logger.error(f"Model directory not found: {trained_model_directory}")
    elif not os.path.isdir(base_healthy_data_directory):
        logger.error(f"Base healthy data directory not found: {base_healthy_data_directory}")
    elif damage_data_directory and not os.path.isdir(damage_data_directory):  # Solo valida si se proporcionó
        # Advertencia en lugar de error, el script puede correr solo con datos sanos
        logger.warning(
            f"Damage data directory provided but not found: {damage_data_directory}. Running with healthy data only.")
        run_inference_and_plot(
            trained_model_directory,
            base_healthy_data_directory,
            damage_data_dir=None  # Pasar None explícitamente
        )
    else:  # Si todos los directorios necesarios existen (o damage_data_directory es None)
        run_inference_and_plot(
            trained_model_directory,
            base_healthy_data_directory,  # Este es 'base_healthy_dir'
            damage_data_directory  # Este es 'damage_data_dir' (puede ser None si no se encontró antes)
        )

