import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import joblib
import logging
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from datetime import datetime
import glob

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

        local_logger.info(f"Inicializando STAutoencoder: N={num_nodes}, F={num_features}, T={window_size}")
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
    local_logger.info(f"Buscando archivos en: {directory} con patrón '<id>_*'")
    all_sensor_data = {}
    loaded_files_count = 0
    min_length = float('inf')

    # Los sensores están indexados desde 1
    sensor_ids_to_find = list(range(1, num_nodes + 1))

    for i in sensor_ids_to_find:
        search_pattern = os.path.join(directory, f"{i}_*")
        file_list = glob.glob(search_pattern)

        if not file_list:
            local_logger.warning(f"No se encontraron archivos para el sensor {i} con el patrón: {search_pattern}")
            continue

        file_list.sort()
        sensor_df_list = []
        for filepath in file_list:
            try:
                # Asumimos una sola columna de datos
                # CORRECCIÓN: Reemplazar 'delim_whitespace=True' por 'sep=r'\s+''
                df = pd.read_csv(filepath, header=None, sep=r'\s+', usecols=[0], engine='python')
                if df.empty:
                    local_logger.warning(f"Archivo vacío omitido: {filepath}")
                    continue
                sensor_df_list.append(df)
            except Exception as e:
                local_logger.error(f"Error al leer o procesar el archivo {filepath}: {e}")

        if not sensor_df_list:
            local_logger.warning(
                f"No se pudo leer ningún dato válido para el sensor {i} a pesar de encontrar archivos.")
            continue

        full_sensor_df = pd.concat(sensor_df_list, ignore_index=True)
        all_sensor_data[i] = full_sensor_df.iloc[:, 0].values
        loaded_files_count += len(file_list)

        if len(all_sensor_data[i]) < min_length:
            min_length = len(all_sensor_data[i])

    if not all_sensor_data:
        local_logger.error("No se cargaron datos de ningún sensor.")
        return None, 0

    if len(all_sensor_data) < num_nodes:
        local_logger.warning(
            f"Se esperaban datos para {num_nodes} sensores, pero solo se cargaron {len(all_sensor_data)}. Continuando...")
        # Llenar los datos faltantes con ceros o manejar de otra forma
        for i in sensor_ids_to_find:
            if i not in all_sensor_data:
                local_logger.warning(f"Rellenando datos faltantes para sensor {i} con ceros.")
                # Usar min_length (de los sensores encontrados) o un valor por defecto
                fill_length = min_length if min_length != float('inf') else 1
                all_sensor_data[i] = np.zeros(fill_length)

    # Si ningún archivo se cargó, min_length sigue siendo inf
    if min_length == float('inf'):
        if max_len:  # Si se pasó un max_len, usarlo
            min_length = max_len
        else:
            local_logger.error("No se pudo determinar la longitud mínima de los datos.")
            return None, 0

    local_logger.info(
        f"Carga completada. Archivos cargados: {loaded_files_count}, Sensores con datos: {len(all_sensor_data)}.")

    # Truncar todos los arrays a la longitud mínima o max_len
    # Si max_len no se proporciona, se usa min_length (la más corta encontrada)
    if max_len is None:
        max_len = min_length

    local_logger.info(f"Normalizando todos los sensores a una longitud de: {max_len}")

    processed_data = np.zeros((max_len, num_nodes))
    for i in sensor_ids_to_find:
        sensor_data = all_sensor_data.get(i)
        if sensor_data is None:
            local_logger.warning(f"No hay datos para sensor {i} en el diccionario final. Usando ceros.")
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

    local_logger.info(f"Datos procesados con shape final: {processed_data.shape}.")

    return processed_data, max_len


def perform_inference(model, dataloader, device, edge_index):
    """Ejecuta la inferencia y calcula errores."""
    local_logger = logging.getLogger(f"{__name__}.inference")
    model.eval()
    all_inputs_np = []
    all_outputs_np = []
    all_losses_np = []  # Pérdida promedio por ventana
    all_losses_per_sensor_np = []  # Array (n_samples, n_nodes)

    criterion_none = nn.MSELoss(reduction='none')

    total_batches = len(dataloader)
    if total_batches == 0:
        local_logger.error("El DataLoader está vacío. No hay datos para inferencia.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    processed_windows = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Inferencia', leave=False, total=total_batches)
        for batch_idx, (inputs, _) in enumerate(progress_bar):
            if inputs is None or len(inputs) == 0:
                local_logger.warning(f"Lote {batch_idx + 1}/{total_batches} vacío. Omitiendo.")
                continue

            inputs = inputs.to(device)  # Shape: (batch, time, nodes, feats)

            # Asegurarse de que edge_index esté en el dispositivo correcto
            edge_index = edge_index.to(device)

            try:
                outputs = model(inputs, edge_index)  # Shape: (batch, time, nodes, feats)

                loss_elementwise = criterion_none(outputs, inputs)  # Shape: (batch, time, nodes, feats)
                loss_per_window = torch.mean(loss_elementwise, dim=(1, 2, 3))  # Shape: (batch,)
                loss_per_sensor_window = torch.mean(loss_elementwise, dim=(1, 3))  # Shape: (batch, nodes)

                all_losses_np.append(loss_per_window.cpu().numpy())
                all_losses_per_sensor_np.append(loss_per_sensor_window.cpu().numpy())
                all_inputs_np.append(inputs.cpu().numpy())
                all_outputs_np.append(outputs.cpu().numpy())

                processed_windows += len(inputs)
                progress_bar.set_postfix({'Ventanas': processed_windows})

            except Exception as e:
                local_logger.error(f"Error durante la inferencia en lote {batch_idx + 1}/{total_batches}: {e}",
                                   exc_info=True)
                continue  # Saltar al siguiente lote

    # Concatenar resultados al final
    if not all_losses_np:
        local_logger.error("No se procesó ninguna ventana con éxito.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_inputs_np = np.concatenate(all_inputs_np, axis=0)
    all_outputs_np = np.concatenate(all_outputs_np, axis=0)
    all_losses_np = np.concatenate(all_losses_np, axis=0)  # Shape: (n_samples,)
    all_losses_per_sensor_np = np.concatenate(all_losses_per_sensor_np, axis=0)  # Shape: (n_samples, n_nodes)

    local_logger.info(f"Inferencia completada. Procesadas {processed_windows} ventanas.")
    local_logger.info(f"Shape final de pérdidas por ventana: {all_losses_np.shape}")
    local_logger.info(f"Shape final de pérdidas por sensor: {all_losses_per_sensor_np.shape}")

    return all_inputs_np, all_outputs_np, all_losses_np, all_losses_per_sensor_np


# --- FUNCIONES DE PLOTEO ---
local_logger_plot = logging.getLogger(f"{__name__}.plotting")
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    local_logger_plot.warning("Estilo 'seaborn-v0_8-whitegrid' no encontrado, usando 'ggplot'.")
    plt.style.use('ggplot')


def plot_reconstruction_sample(original, reconstructed, window_idx, output_dir, prefix):
    """
    Genera gráfico de reconstrucción para una muestra específica.
    original, reconstructed: shape (time, nodes, feats)
    """
    if original is None or reconstructed is None or original.size == 0 or reconstructed.size == 0:
        local_logger_plot.warning(
            f"Datos originales o reconstruidos faltantes/vacíos para ventana {window_idx} ({prefix}). No se generará gráfico.")
        return

    # Asegurar shapes correctos
    if original.ndim != 3 or reconstructed.ndim != 3 or original.shape != reconstructed.shape:
        local_logger_plot.error(f"Shapes inconsistentes para ploteo: Orig {original.shape}, Rec {reconstructed.shape}")
        return
    if original.shape[2] != 1:  # Asumiendo F=1
        local_logger_plot.error(f"Se esperaba 1 característica, pero se encontraron {original.shape[2]}")
        return

    num_sensors = original.shape[1]
    error_signal = original.squeeze(-1) - reconstructed.squeeze(-1)  # Shape (time, nodes)

    fig, axes = plt.subplots(num_sensors, 2, figsize=(15, 3 * num_sensors), sharex=True, squeeze=False)
    fig.suptitle(f'Reconstrucción de Muestra {prefix.capitalize()} (Ventana Índice: {window_idx})', fontsize=16)

    for i in range(num_sensors):
        ax_sig = axes[i, 0]
        ax_err = axes[i, 1]

        ax_sig.plot(original[:, i, 0], label='Original', color='tab:blue', linewidth=1.5)
        ax_sig.plot(reconstructed[:, i, 0], label='Reconstruida', color='tab:orange', linestyle='--', linewidth=1.5)
        ax_sig.set_title(f'Sensor {i + 1}: Señal Original vs. Reconstruida')
        ax_sig.set_ylabel('Valor Normalizado')
        ax_sig.legend(fontsize='small')
        ax_sig.grid(True, linestyle=':')

        ax_err.plot(error_signal[:, i], label='Error', color='tab:red', linewidth=1.5)
        ax_err.set_title(f'Sensor {i + 1}: Error de Reconstrucción')
        ax_err.set_ylabel('Error')
        ax_err.grid(True, linestyle=':')
        ax_err.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax_err.legend(fontsize='small')

    axes[num_sensors - 1, 0].set_xlabel('Paso de Tiempo en Ventana')
    axes[num_sensors - 1, 1].set_xlabel('Paso de Tiempo en Ventana')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, f"{prefix}_reconstruction_sample_{window_idx}.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de reconstrucción guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error al guardar el gráfico {filename}: {e}")
    plt.close(fig)


def plot_mse_comparison(mse_healthy_per_sensor_avg, mse_damage_per_sensor_avg, output_dir):
    """Genera gráfico comparativo de MSE promedio por sensor (sano vs. daño)."""
    if mse_healthy_per_sensor_avg is None or mse_damage_per_sensor_avg is None or \
            mse_healthy_per_sensor_avg.size == 0 or mse_damage_per_sensor_avg.size == 0 or \
            len(mse_healthy_per_sensor_avg) != len(mse_damage_per_sensor_avg):
        local_logger_plot.error("Datos de MSE promedio por sensor inválidos o inconsistentes para comparación.")
        return

    num_sensors = len(mse_healthy_per_sensor_avg)
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Detección de Daño: Error de Reconstrucción Promedio por Sensor', fontsize=16)

    # Panel Izquierdo: Datos Sanos (Escala Lineal)
    axes[0].bar(sensor_labels, mse_healthy_per_sensor_avg, color='skyblue', edgecolor='black')
    axes[0].set_title('Datos Sanos (Línea Base)')
    axes[0].set_ylabel('Error Cuadrático Medio (MSE)')
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
    y_label_damage = 'Error Cuadrático Medio (MSE) (Escala Log)'

    if valid_damage_mse.size != num_sensors:
        local_logger_plot.warning("Errores MSE de daño no positivos encontrados. Usando escala lineal.")
        use_log_scale = False
        y_label_damage = 'Error Cuadrático Medio (MSE)'

    axes[1].bar(sensor_labels, plot_data_damage, color='orangered', edgecolor='black')
    axes[1].set_title('Datos del Sismo' + (' (Escala Logarítmica)' if use_log_scale else ''))
    axes[1].set_ylabel(y_label_damage)

    min_val_plot = 1e-9  # Valor mínimo para escala log

    if use_log_scale and valid_damage_mse.size > 0:
        axes[1].set_yscale('log')
        # Asegurarse de que el límite inferior sea visible
        axes[1].set_ylim(bottom=min(np.min(valid_damage_mse) * 0.1, min_val_plot))
    elif use_log_scale:  # valid_damage_mse está vacío
        axes[1].set_yscale('log')
        axes[1].set_ylim(bottom=min_val_plot)
    else:
        axes[1].set_ylim(bottom=0)

    axes[1].grid(True, axis='y', linestyle=':')

    max_damage_val = np.max(plot_data_damage) if plot_data_damage.size > 0 else 1
    for i, v in enumerate(plot_data_damage):
        if use_log_scale:
            text_pos = max(v, min_val_plot) * 1.5
        else:
            text_pos = v + 0.03 * max(max_damage_val, 1e-9)

        axes[1].text(i, text_pos, f"{v:.3e}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, "damage_detection_mse_comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de comparación MSE guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error al guardar el gráfico {filename}: {e}")
    plt.close(fig)


def plot_damage_localization(losses_damage_per_sensor, losses_healthy_per_sensor, output_dir):
    """
    Genera gráfico de localización (boxplot y factor de amplificación).
    Recibe las distribuciones completas de pérdidas: (n_samples, n_nodes)
    """
    if losses_damage_per_sensor is None or losses_healthy_per_sensor is None or \
            losses_damage_per_sensor.size == 0 or losses_healthy_per_sensor.size == 0 or \
            losses_damage_per_sensor.shape[1] != losses_healthy_per_sensor.shape[1]:
        local_logger_plot.error("Datos de pérdidas por sensor inválidos o inconsistentes para localización.")
        return

    num_sensors = losses_damage_per_sensor.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Análisis de Localización de Daño por Sensor', fontsize=16)

    # --- Panel Superior: Boxplot Error con Daño (Escala Log) ---
    # Filtrar no positivos antes de calcular log o boxplot
    plot_data_damage = [losses_damage_per_sensor[:, i][losses_damage_per_sensor[:, i] > 0] for i in range(num_sensors)]
    valid_data_indices = [i for i, data in enumerate(plot_data_damage) if len(data) > 0]

    if not valid_data_indices:
        local_logger_plot.error("No hay datos positivos de error de daño para graficar el boxplot.")
        axes[0].set_title('Distribución Error Daño (No hay datos positivos)')
    else:
        # Graficar solo para los sensores con datos válidos
        plot_data_damage_valid = [plot_data_damage[i] for i in valid_data_indices]
        sensor_labels_valid = [sensor_labels[i] for i in valid_data_indices]

        bp = axes[0].boxplot(plot_data_damage_valid, labels=sensor_labels_valid, patch_artist=True, showfliers=False)
        for patch in bp['boxes']: patch.set_facecolor('orangered')
        for median in bp['medians']: median.set(color='black', linewidth=1.5)

        axes[0].set_title('Distribución del Error de Reconstrucción (Datos con Daño)')
        axes[0].set_ylabel('Error Cuadrático Medio (MSE) - Escala Log')
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle=':')

    # --- Panel Inferior: Factor de Amplificación Mediano ---
    median_error_damage = np.median(losses_damage_per_sensor, axis=0)
    median_error_healthy = np.median(losses_healthy_per_sensor, axis=0)
    # Evitar división por cero o por valores muy pequeños
    median_error_healthy_safe = np.maximum(median_error_healthy, 1e-10)

    amplification_factor = median_error_damage / median_error_healthy_safe

    axes[1].bar(sensor_labels, amplification_factor, color='crimson', edgecolor='black')
    axes[1].set_title('Factor de Amplificación del Error Mediano (Daño vs. Sano)')
    axes[1].set_ylabel('Error Mediano Daño / Error Mediano Sano')
    axes[1].grid(True, linestyle=':')
    max_amp = np.max(amplification_factor) if amplification_factor.size > 0 else 1
    for i, v in enumerate(amplification_factor):
        axes[1].text(i, v * 1.05, f"{v:.2f}", ha='center', va='bottom', fontsize=9)  # .2f es mejor para factor

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "damage_localization_analysis.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de localización de daño guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error al guardar el gráfico {filename}: {e}")
    plt.close(fig)


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
        log_file_path = os.path.join(results_dir, 'inference.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        logger.info(f"--- Iniciando Inferencia usando modelo de: {model_dir} ---")
        logger.info(f"Directorio de datos sanos: {base_healthy_dir}")
        logger.info(f"Directorio de datos con daño: {damage_data_dir}")
        logger.info(f"Los resultados se guardarán en: {results_dir}")

        # --- Carga de hiperparámetros y escalador ---
        params_path = os.path.join(model_dir, 'hyperparameters.json')
        scaler_path = os.path.join(model_dir, 'scaler.gz')
        with open(params_path, 'r') as f:
            params = json.load(f)
        scaler = joblib.load(scaler_path)
        logger.info("Hiperparámetros y escalador cargados.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")

        # --- Definir Grafo ---
        edge_index = define_bridge_graph().to(device)
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

        model_path = os.path.join(model_dir, 'best_model.pth')
        # CORRECCIÓN: Añadir weights_only=True por seguridad y para eliminar warning
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        logger.info("Modelo cargado y en modo evaluación.")

        # --- Carga y procesamiento de datos SANOS ---
        logger.info("Cargando y procesando datos sanos...")
        all_healthy_data_raw, max_len_healthy = load_data_from_dir(base_healthy_dir, num_nodes)
        if all_healthy_data_raw is None:
            raise ValueError("No se pudieron cargar los datos sanos base.")

        # --- INICIO DE LA CORRECCIÓN DEL ERROR DE SCALER ---
        # El error 'ValueError: X has 5 features, but StandardScaler is expecting 1'
        # indica que el scaler fue entrenado (fit) en datos con 1 columna
        # (probablemente todos los sensores apilados verticalmente).
        # Debemos replicar esa estructura antes de 'transform'.

        original_shape_healthy = all_healthy_data_raw.shape  # (N, 5)

        # 1. Aplanar a (N*5, 1) para que coincida con el scaler
        data_healthy_reshaped = all_healthy_data_raw.reshape(-1, 1)

        # 2. Aplicar el transform
        scaled_data_healthy_flat = scaler.transform(data_healthy_reshaped)

        # 3. Volver a la forma original (N, 5)
        healthy_data_scaled = scaled_data_healthy_flat.reshape(original_shape_healthy)
        logger.info(f"Datos sanos re-escalados de (N*1) a {healthy_data_scaled.shape}")
        # --- FIN DE LA CORRECCIÓN ---

        # Crear el diccionario esperado por el Dataset
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
            raise ValueError("El dataset sano no tiene suficientes datos para crear ni una sola ventana.")
        healthy_loader = DataLoader(healthy_dataset, batch_size=params['batch_size'], shuffle=False)

        # --- Inferencia en datos SANOS ---
        logger.info("Realizando inferencia en datos sanos...")
        healthy_originals, healthy_reconstructions, healthy_losses, healthy_losses_per_sensor = perform_inference(
            model, healthy_loader, device, edge_index
        )
        if healthy_losses_per_sensor.size == 0:
            raise ValueError("Inferencia en datos sanos falló, no se obtuvieron resultados.")

        # Calcular el MSE promedio por sensor sobre todas las ventanas
        healthy_mse_mean_per_sensor = np.mean(healthy_losses_per_sensor, axis=0)

        # --- Carga e inferencia en datos con DAÑO (si se proporcionan) ---
        damage_originals = None
        damage_reconstructions = None
        damage_losses_per_sensor = None
        damage_mse_mean_per_sensor = None

        if damage_data_dir:
            logger.info("Cargando y procesando datos con daño...")
            # Usar max_len_healthy para truncar los datos de daño a la misma longitud
            all_damage_data_raw, _ = load_data_from_dir(damage_data_dir, num_nodes, max_len=max_len_healthy)

            if all_damage_data_raw is not None:

                # --- INICIO DE LA CORRECCIÓN DEL ERROR DE SCALER (DAÑO) ---
                original_shape_damage = all_damage_data_raw.shape  # (N, 5)
                data_damage_reshaped = all_damage_data_raw.reshape(-1, 1)  # (N*5, 1)
                scaled_data_damage_flat = scaler.transform(data_damage_reshaped)
                damage_data_scaled = scaled_data_damage_flat.reshape(original_shape_damage)  # (N, 5)
                logger.info(f"Datos con daño re-escalados de (N*1) a {damage_data_scaled.shape}")
                # --- FIN DE LA CORRECCIÓN ---

                # Crear el diccionario
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
                    logger.info("Realizando inferencia en datos con daño...")

                    damage_originals, damage_reconstructions, damage_losses, damage_losses_per_sensor = perform_inference(
                        model, damage_loader, device, edge_index
                    )
                    if damage_losses_per_sensor.size > 0:
                        damage_mse_mean_per_sensor = np.mean(damage_losses_per_sensor, axis=0)
                    else:
                        logger.warning("Inferencia en datos con daño no produjo resultados.")
                else:
                    logger.warning("El dataset de daño no tiene suficientes datos. Se omitirá el análisis de daño.")
            else:
                logger.warning("No se pudieron cargar los datos de daño. Se omitirá el análisis de daño.")
        else:
            logger.info("No se proporcionó directorio de datos de daño. Omitiendo análisis de daño.")

        # --- PLOTEO (CORREGIDO) ---
        logger.info("Generando gráficos de resultados...")

        if healthy_originals is not None and len(healthy_originals) > 0:
            plot_reconstruction_sample(
                healthy_originals[0], healthy_reconstructions[0], 0, results_dir, "healthy"
            )
        else:
            logger.warning("No hay datos de reconstrucción sanos para plotear.")

        if damage_originals is not None and len(damage_originals) > 0:
            plot_reconstruction_sample(
                damage_originals[0], damage_reconstructions[0], 0, results_dir, "damage"
            )
        else:
            logger.warning("No hay datos de reconstrucción con daño para plotear.")

        if damage_mse_mean_per_sensor is not None and damage_losses_per_sensor is not None:
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
        else:
            logger.info("No hay datos de daño procesados, omitiendo gráficos de comparación y localización.")

        logger.info("--- Proceso de inferencia completado ---")

    except Exception as e:
        logger.error(f"Error crítico en run_inference_and_plot: {e}", exc_info=True)
    finally:
        # Buena práctica: remover el file handler para que futuras ejecuciones (en un notebook)
        # puedan añadir uno nuevo sin duplicar.
        if file_handler is not None:
            logger.info("Cerrando archivo de log.")
            file_handler.close()
            logger.removeHandler(file_handler)


# --- EJECUCIÓN ---
if __name__ == '__main__':
    # Asegúrate de que estas rutas sean correctas
    trained_model_directory = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756"
    damage_data_directory = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"
    base_healthy_data_directory = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # Validar directorios antes de ejecutar
    if not os.path.isdir(trained_model_directory):
        logger.error(f"Directorio del modelo no encontrado: {trained_model_directory}")
    elif not os.path.isdir(base_healthy_data_directory):
        logger.error(f"Directorio base de datos sanos no encontrado: {base_healthy_data_directory}")
    elif not os.path.isdir(damage_data_directory):
        # Advertencia en lugar de error, el script puede correr solo con datos sanos
        logger.warning(f"Directorio de datos de daño no encontrado: {damage_data_directory}")
        # Ejecutar de todas formas, pero sin análisis de daño
        run_inference_and_plot(
            trained_model_directory,
            base_healthy_data_directory,
            damage_data_dir=None  # Pasar None explícitamente
        )
    else:
        run_inference_and_plot(
            trained_model_directory,
            base_healthy_data_directory,  # Este es 'base_healthy_dir'
            damage_data_directory  # Este es 'damage_data_dir'
        )

