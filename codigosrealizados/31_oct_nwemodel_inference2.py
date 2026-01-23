# -*- coding: utf-8 -*-
"""
evaluate_model.py

Carga un modelo STG-AE (Físico) entrenado y su scaler para:
1. Cargar datos SANOS (validación) y datos de DAÑO REAL.
2. Calcular y graficar la distribución de errores (Histogramas Log y Lineal).
3. Calcular y graficar el error POR SENSOR (Localización de daño).
4. Generar gráficas de la arquitectura (Grafo, Matriz) y pre-procesamiento (Wavelets).
5. Graficar el error de reconstrucción a lo largo del tiempo.

Este script genera el set completo de figuras para la sección de resultados.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import json
import joblib
import logging
import sys
import pywt
import gc
from tqdm import tqdm

# --- Dependencias Requeridas ---
GCNConv = None
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("Error: torch_geometric no está instalado. Ejecuta: pip install torch_geometric")
    GCNConv = None
    sys.exit(1)

nx = None
try:
    import networkx as nx
except ImportError:
    print("Advertencia: networkx no está instalado. No se generará la gráfica del grafo.")
    print("Ejecuta: pip install networkx")
    nx = None

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# =============================================================================
# DEFINICIONES DE CLASES (Requeridas para cargar el modelo)
# =============================================================================


def adjust_signal_length(signal, target_len):
    current_len = len(signal)
    if current_len == target_len:
        return signal
    if current_len > target_len:
        return signal[:target_len]
    return np.concatenate((signal, np.zeros(target_len - current_len)))


def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 1:
        return None
    original_len = len(signal)
    if target_len is None:
        target_len = original_len
    if original_len == 0:
        return None
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        reconstructed_bands = []
        for i in range(level, 0, -1):
            detail_level_index = level - i + 1
            if detail_level_index >= len(coeffs):
                return None
            detail_coeffs_list = [np.zeros_like(c) for c in coeffs]
            detail_coeffs_list[detail_level_index] = coeffs[detail_level_index]
            rec_d = pywt.waverec(detail_coeffs_list, wavelet)
            reconstructed_bands.append(adjust_signal_length(rec_d, target_len))
        approx_coeffs_list = [coeffs[0]] + \
                             [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_coeffs_list, wavelet)
        reconstructed_bands.append(adjust_signal_length(rec_a, target_len))
        original_adjusted = adjust_signal_length(signal, target_len)
        ordered_bands_rev = reconstructed_bands[::-1]
        all_bands = [original_adjusted] + ordered_bands_rev
        features = np.stack(all_bands, axis=-1)
        return features
    except Exception:
        return None


class SpatioTemporalWaveletDataset(Dataset):
    def __init__(self, data_dict_features, window_size, stride=1, num_expected_nodes=5):
        self.window_size = window_size
        self.stride = stride
        self.num_expected_nodes = num_expected_nodes
        local_logger = logging.getLogger(self.__class__.__name__)
        valid_data_dict = {}
        min_len = float('inf')
        expected_num_features = -1
        for sid, data in data_dict_features.items():
            if data is not None and isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] >= window_size:
                current_num_features = data.shape[1]
                if expected_num_features == -1:
                    expected_num_features = current_num_features
                elif current_num_features != expected_num_features:
                    local_logger.error(
                        f"Inconsistencia features. S{sid} tiene {current_num_features}, se esperaban {expected_num_features}. Omitiendo.")
                    continue
                valid_data_dict[sid] = data
                min_len = min(min_len, data.shape[0])
            else:
                local_logger.warning(f"Datos inválidos S{sid}. Omitiendo.")
        if len(valid_data_dict) < self.num_expected_nodes:  # Menos estricto para datos de daño
            missing = set(range(1, self.num_expected_nodes + 1)) - \
                      set(valid_data_dict.keys())
            # No lanzar error, solo advertir si faltan
            local_logger.warning(
                f"Faltan datos para sensores: {missing}. Usando {len(valid_data_dict)} sensores.")
            # Re-indexar nodos si es necesario (para este caso asumimos que los 5 están)
            if len(valid_data_dict) != self.num_expected_nodes:
                raise ValueError(f"Faltan datos para sensores: {missing}.")
        if min_len < window_size and min_len != float('inf'):
            raise ValueError(f"Longitud mínima ({min_len}) insuficiente.")
        elif min_len == float('inf'):
            raise ValueError("No se cargaron datos válidos (min_len es inf).")

        processed_data_list = []
        for sid in range(1, self.num_expected_nodes + 1):
            data_node = valid_data_dict[sid][:min_len]
            processed_data_list.append(data_node)
            self.num_features = data_node.shape[1]
        self.data = np.stack(processed_data_list, axis=1)
        self.num_nodes = self.data.shape[1]
        self.n_samples = (self.data.shape[0] - window_size) // stride + 1
        local_logger.info(
            f"Dataset STG-AE creado. Shape datos: {self.data.shape}. {self.n_samples} ventanas.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        return torch.FloatTensor(self.data[start:end]), torch.FloatTensor(self.data[start:end])


class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, bias=False)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index = edge_index.to(x.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(x.device)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden, gnn_out, rnn_hidden, rnn_layers):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_features = num_features
        self.gnn_hidden_dim = gnn_hidden
        self.gnn_encoder_out_dim = gnn_out
        self.rnn_encoder_hidden_dim = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn_decoder_output_dim = self.gnn_hidden_dim * num_nodes
        self.gnn_encoder = GNNLayer(
            num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)
        self.rnn_encoder = nn.GRU(self.gnn_encoder_out_dim * num_nodes,
                                  self.rnn_encoder_hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.rnn_decoder = nn.GRU(
            self.rnn_encoder_hidden_dim, self.rnn_decoder_output_dim, batch_first=True, num_layers=self.rnn_layers)
        self.gnn_decoder = GNNLayer(
            self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)
        self.latent_project_up = nn.Linear(
            self.rnn_encoder_hidden_dim, self.rnn_encoder_hidden_dim)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        try:
            batch_size, T_actual, N_actual, F_actual = x.shape
            x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)
            gnn_encoded = self.gnn_encoder(
                x_reshaped, edge_index, edge_weight)
            gnn_encoded_view = gnn_encoded.reshape(
                batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
            rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)
            _, h_n = self.rnn_encoder(rnn_input)
            latent_vector_z = self.relu(self.latent_project_up(h_n[-1]))
            rnn_decoder_input = latent_vector_z.unsqueeze(
                1).repeat(1, T_actual, 1)
            rnn_decoded, _ = self.rnn_decoder(rnn_decoder_input)
            gnn_input_decoder = rnn_decoded.reshape(
                batch_size * T_actual, N_actual, self.gnn_hidden_dim)
            reconstructed_frames = self.gnn_decoder(
                gnn_input_decoder, edge_index, edge_weight)
            reconstructed_x = reconstructed_frames.reshape(
                batch_size, T_actual, N_actual, F_actual)
            return reconstructed_x
        except Exception as e:
            logger.error(
                f"Error en forward pass de STG-AE: {e}", exc_info=True)
            raise e


# =============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================


def load_raw_data(data_dir, num_expected_nodes=5):
    """Carga y concatena todos los archivos .txt de un directorio."""
    all_files = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_dir}")
        return None, 0

    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    for f_path in all_files:
        try:
            # Modificado para ser más robusto: buscar 'S1', 'S2', etc. o '1_', '2_'
            filename = os.path.basename(f_path)
            sid_str = filename.split('_')[0]
            # Extraer solo dígitos
            sid_digits = ''.join(filter(str.isdigit, sid_str))
            if not sid_digits:
                logger.warning(f"No se pudo extraer ID numérico de '{filename}'. Omitiendo.")
                continue

            sid = int(sid_digits)

            if sid in sensor_data_raw:
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[
                    1], engine='python', on_bad_lines='warn').values
                if data is not None and data.size > 0:
                    sensor_data_raw[sid].append(data)
            else:
                logger.warning(f"ID '{sid}' de '{filename}' fuera de rango [1-{num_expected_nodes}]. Omitiendo.")

        except Exception as e:
            logger.warning(
                f"No se pudo procesar el archivo {f_path}: {e}. Omitiendo.")
            pass

    sensor_data_concat = {}
    min_len_raw = float('inf')
    sensors_with_data = []
    for sid, data_list in sensor_data_raw.items():
        if data_list:
            data_list_valid = [d.reshape(-1, 1) for d in data_list if d is not None and d.size > 0 and d.ndim == 1] + \
                              [d for d in data_list if d is not None and d.size >
                               0 and d.ndim == 2 and d.shape[1] == 1]
            if not data_list_valid:
                logger.warning(f"Sensor {sid} no tiene datos válidos tras filtrar.")
                continue
            try:
                concatenated_data = np.concatenate(data_list_valid, axis=0)
                if concatenated_data.size > 0:
                    sensor_data_concat[sid] = concatenated_data.squeeze()
                    min_len_raw = min(min_len_raw, len(concatenated_data))
                    sensors_with_data.append(sid)
            except Exception as e:
                logger.error(f"Error concatenando S{sid}: {e}")
                pass

    if len(sensors_with_data) != num_expected_nodes:
        logger.error(
            f"Faltan datos de sensor en {data_dir}. Encontrados: {sensors_with_data}")
        return None, 0

    if min_len_raw == float('inf'):
        logger.error(
            f"min_len_raw sigue siendo infinito, no se cargaron datos.")
        return None, 0

    return sensor_data_concat, min_len_raw


def process_data_with_scaler(sensor_data_concat, min_len, hp, scaler):
    """Aplica DWT y el scaler cargado a los datos crudos."""
    num_expected_nodes = hp['num_nodes']
    wavelet_name = hp['wavelet_name']
    wavelet_level = hp['wavelet_level']

    sensor_data_features = {}
    for sid in range(1, num_expected_nodes + 1):
        if sid not in sensor_data_concat:
            logger.error(
                f"Faltan datos concatenados para S{sid} en process_data_with_scaler")
            return None
        signal_1d = sensor_data_concat[sid]
        features_2d = apply_dwt_features(
            signal_1d, wavelet=wavelet_name, level=wavelet_level, target_len=min_len)
        if features_2d is None:
            logger.error(f"Error aplicando DWT a S{sid}")
            return None
        sensor_data_features[sid] = features_2d

    sensor_data_scaled_features = {}
    for sid in range(1, num_expected_nodes + 1):
        sensor_data_scaled_features[sid] = scaler.transform(
            sensor_data_features[sid])

    return sensor_data_scaled_features


# =============================================================================
# FUNCIONES DE EVALUACIÓN
# =============================================================================


def create_physics_informed_graph(num_nodes=5):
    """
    Crea el grafo ponderado (idéntico al de entrenamiento).
    """
    coords = {
        0: np.array([13.88, -4.0, -1.0]),  # Sensor 1
        1: np.array([13.88, 4.0, -1.0]),  # Sensor 2
        2: np.array([27.76, -4.0, -1.0]),  # Sensor 3
        3: np.array([27.76, 4.0, -1.0]),  # Sensor 4
        4: np.array([41.64, 0.0, -1.0])  # Sensor 5
    }
    edge_index_list = []
    edge_weight_list = []
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])
            dist = np.linalg.norm(coords[i] - coords[j])
            weight = 1.0 / (dist + 1e-6)
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight

    edge_index = torch.tensor(
        edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    return {'edge_index': edge_index, 'edge_weight': edge_weight, 'coords': coords, 'adj_matrix': adj_matrix}


def get_reconstruction_errors_per_sensor(model, dataloader, device, edge_index, edge_weight):
    """
    Calcula el error MSE para cada muestra Y CADA SENSOR.
    Retorna un array de shape (N_samples, N_sensores)
    """
    model.eval()
    # Usar reduction='none' para obtener error por elemento
    criterion = nn.MSELoss(reduction='none')
    errors_per_sensor_list = []

    desc = "Calculando error por sensor (SANO)"
    if hasattr(dataloader.dataset, 'is_damage_dataset'):  # Pequeño hack para cambiar el label
        desc = "Calculando error por sensor (DAÑO REAL)"

    all_errors_per_sensor = []
    with torch.no_grad():
        for (inputs, _) in tqdm(dataloader, desc=desc):
            inputs = inputs.to(device)
            outputs = model(inputs, edge_index, edge_weight)

            # outputs/inputs shape: (B, T, N, F)
            # error shape: (B, T, N, F)
            error = criterion(outputs, inputs)

            # Queremos el error medio por sensor.
            # Promediamos sobre Batch, Tiempo y Features (dims 0, 1, 3)
            # El shape resultante es (N,)
            error_per_sensor = torch.mean(error, dim=(0, 1, 3))

            # Como B=1, podemos apilar directamente
            all_errors_per_sensor.append(error_per_sensor.cpu().numpy())

    # Retorna (N_samples, N_sensores)
    return np.array(all_errors_per_sensor)


# =============================================================================
# FUNCIONES DE GRAFICACIÓN (NUEVAS Y MEJORADAS)
# =============================================================================

def plot_error_histograms(errors_dict, output_dir):
    """
    Genera y guarda DOS histogramas comparativos: Lineal y Log.
    """
    healthy_errors = errors_dict.get('Sano')
    if healthy_errors is None:
        logger.warning("No hay errores 'Sano' para plotear histograma.")
        return

    # --- Gráfico 1: Escala Lineal (Zoom a lo 'Sano') ---
    plt.figure(figsize=(12, 7))

    threshold = np.percentile(healthy_errors, 99)

    # Plotear errores sanos
    plt.hist(healthy_errors, bins=100, density=True, alpha=0.7,
             label=f'Sano (N={len(healthy_errors)})', color='C0')

    colors = ['#FF7F0E', '#2CA02C']  # Naranja, Verde

    # Plotear errores de daño real (si existen)
    damage_errors = errors_dict.get('Daño Real')
    if damage_errors is not None:
        plt.hist(damage_errors, bins=100, density=True, alpha=0.7,
                 label=f'Daño Real (N={len(damage_errors)})', color=colors[0])

    plt.axvline(threshold, color='r', linestyle='--',
                label=f'Umbral (Percentil 99 = {threshold:.4f})')

    plt.title('Distribución de Errores (Escala Lineal - Zoom "Sano")')
    plt.xlabel('Error de Reconstrucción (MSE)')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, linestyle=':')
    # Zoom a la parte importante
    plt.xlim(0, threshold * 5)
    # Definir límite Y para que se vea bien el histograma sano
    hist_sano, bins_sano = np.histogram(healthy_errors, bins=100, density=True)
    plt.ylim(0, np.max(hist_sano) * 1.1)

    plot_path_lin = os.path.join(
        output_dir, 'ZZ_RESULT_1_Histograma_Errores_Lineal.png')
    plt.savefig(plot_path_lin, dpi=300)
    logger.info(f"Histograma Lineal guardado en: {plot_path_lin}")
    plt.close()

    # --- Gráfico 2: Escala Logarítmica (Vista Completa) ---
    plt.figure(figsize=(12, 7))

    # Plotear errores sanos
    plt.hist(healthy_errors, bins=100, density=True, alpha=0.7,
             label=f'Sano (N={len(healthy_errors)})', color='C0')

    # Plotear errores de daño real (si existen)
    if damage_errors is not None:
        plt.hist(damage_errors, bins=100, density=True, alpha=0.7,
                 label=f'Daño Real (N={len(damage_errors)})', color=colors[0])

    plt.axvline(threshold, color='r', linestyle='--',
                label=f'Umbral (Percentil 99 = {threshold:.4f})')

    plt.title('Distribución de Errores (Escala Logarítmica - Vista Completa)')
    plt.xlabel('Error de Reconstrucción (MSE)')
    plt.ylabel('Densidad (Log)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.yscale('log')  # Escala Log

    plot_path_log = os.path.join(
        output_dir, 'ZZ_RESULT_2_Histograma_Errores_Log.png')
    plt.savefig(plot_path_log, dpi=300)
    logger.info(f"Histograma Logarítmico guardado en: {plot_path_log}")
    plt.close()


def plot_error_localization(errors_per_sensor_dict, output_dir):
    """
    Genera un boxplot comparando el error por sensor entre Sano y Daño.
    """
    healthy_errors = errors_per_sensor_dict.get('Sano')
    damage_errors = errors_per_sensor_dict.get('Daño Real')

    if healthy_errors is None or damage_errors is None:
        logger.warning(
            "No se pueden graficar errores por sensor sin datos 'Sano' y 'Daño Real'.")
        return

    num_nodes = healthy_errors.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_nodes)]

    plt.figure(figsize=(15, 8))

    # Posiciones para los boxplots
    pos_sano = np.arange(num_nodes) * 2
    pos_dano = np.arange(num_nodes) * 2 + 0.8

    # Boxplot Sano
    bp_sano = plt.boxplot(healthy_errors, positions=pos_sano, patch_artist=True,
                          widths=0.6, boxprops=dict(facecolor='C0', alpha=0.7),
                          whiskerprops=dict(color='C0'),
                          capprops=dict(color='C0'),
                          medianprops=dict(color='black'),
                          showfliers=False)  # Ocultar outliers de 'Sano' para claridad

    # Boxplot Daño
    bp_dano = plt.boxplot(damage_errors, positions=pos_dano, patch_artist=True,
                          widths=0.6, boxprops=dict(facecolor='C3', alpha=0.7),
                          whiskerprops=dict(color='C3'),
                          capprops=dict(color='C3'),
                          medianprops=dict(color='black'),
                          showfliers=True)  # Mostrar outliers de 'Daño'

    plt.title('Localización de Anomalía: Error de Reconstrucción por Sensor')
    plt.ylabel('Error de Reconstrucción (MSE) - Escala Log')
    plt.xticks(np.arange(num_nodes) * 2 + 0.4, sensor_labels)
    plt.yscale('log')
    plt.grid(True, linestyle=':', axis='y')

    plt.legend([bp_sano["boxes"][0], bp_dano["boxes"][0]],
               ['Sano', 'Daño Real'], loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, 'ZZ_RESULT_3_Localizacion_Error_por_Sensor.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Gráfico de localización de error guardado en: {plot_path}")
    plt.close()


def plot_error_over_time(errors_dict, hp, output_dir):
    """
    NUEVO: Grafica el error de reconstrucción a lo largo del tiempo.
    """
    healthy_errors = errors_dict.get('Sano')
    damage_errors = errors_dict.get('Daño Real')

    if healthy_errors is None or damage_errors is None:
        logger.warning(
            "No se pueden graficar errores en el tiempo sin datos 'Sano' y 'Daño Real'.")
        return

    threshold = np.percentile(healthy_errors, 99)

    # Crear figura con 2 subplots (uno para sano, uno para daño)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=False)  # No compartir X

    # --- Gráfico 1: Error en datos SANOS ---
    axs[0].plot(healthy_errors, label='Error (Sano)', color='C0', alpha=0.7)
    axs[0].axhline(threshold, color='r', linestyle='--',
                   label=f'Umbral ({threshold:.4f})')
    axs[0].set_title(f'Error de Reconstrucción en el Tiempo (Datos Sanos, N={len(healthy_errors)})')
    axs[0].set_ylabel('Error (MSE)')
    axs[0].set_xlabel(f'Índice de Ventana (Stride={hp["stride"]})')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')
    # Escala Log para ver mejor la línea base
    axs[0].set_yscale('log')

    # --- Gráfico 2: Error en datos de DAÑO REAL ---
    axs[1].plot(damage_errors, label='Error (Daño Real)', color='C3', alpha=0.7)
    axs[1].axhline(threshold, color='r', linestyle='--',
                   label=f'Umbral ({threshold:.4f})')
    axs[1].set_title(f'Error de Reconstrucción en el Tiempo (Datos Daño Real, N={len(damage_errors)})')
    axs[1].set_xlabel(f'Índice de Ventana (Stride={hp["stride"]})')
    axs[1].set_ylabel('Error (MSE)')
    axs[1].legend()
    axs[1].grid(True, linestyle=':')
    axs[1].set_yscale('log')  # Escala Log para comparar

    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, 'ZZ_RESULT_4_Error_en_el_Tiempo.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Gráfico de error en el tiempo guardado en: {plot_path}")
    plt.close()


def plot_physical_graph(graph_def, output_dir):
    """
    Dibuja el grafo físico 2D (vista en planta) usando networkx.
    """
    if nx is None:
        logger.warning(
            "networkx no está instalado. Omitiendo gráfico del grafo.")
        return

    coords = graph_def['coords']
    edge_index = graph_def['edge_index'].cpu().numpy()
    edge_weight = graph_def['edge_weight'].cpu().numpy()

    # Crear diccionario de posiciones (X, Y) para la vista en planta
    pos = {i: (coords[i][0], coords[i][1]) for i in coords}
    labels = {i: f'S{i + 1}' for i in coords}

    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G.add_edge(u, v, weight=edge_weight[i])

    # Normalizar pesos para visualización
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights)
    min_w = min(weights)
    norm_weights = [(w - min_w) / (max_w - min_w + 1e-6) *
                    5 + 1 for w in weights]  # Ancho de 1 a 6

    plt.figure(figsize=(12, 7))
    nx.draw_networkx_nodes(
        G, pos, node_color='C0', node_size=1000, alpha=0.8)
    nx.draw_networkx_edges(
        G, pos, width=norm_weights, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=12, font_weight='bold')

    plt.title('Visualización del Grafo Físico Ponderado (PINN-GNN)')
    plt.xlabel('Coordenada X (m) - Longitudinal')
    plt.ylabel('Coordenada Y (m) - Transversal')
    plt.grid(True, linestyle=':')
    plt.axis('equal')

    plot_path = os.path.join(output_dir, 'ZZ_INFO_1_Grafo_Fisico.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Gráfico del grafo físico guardado en: {plot_path}")
    plt.close()


def plot_wavelet_transform(raw_signal, wavelet_features, hp, output_dir):
    """
    Grafica la señal original vs sus componentes Wavelet reconstruidas.
    """
    wavelet_name = hp['wavelet_name']
    wavelet_level = hp['wavelet_level']
    num_features = hp['num_features']  # Debería ser level + 2

    # Tomar una muestra de 2000 puntos
    sample_len = 2000
    if len(raw_signal) < sample_len:
        sample_len = len(raw_signal)

    if sample_len == 0:
        logger.warning("No hay datos de señal cruda para graficar wavelets.")
        return

    t = np.arange(sample_len)

    # Nombres de las features: Original, Aprox, D_level, ..., D1
    feature_names = ['Original (Escalada)'] + [f'Aprox (A{wavelet_level})'] + \
                    [f'Detalle (D{i})' for i in range(wavelet_level, 0, -1)]

    if len(feature_names) != num_features:
        logger.warning(
            f"Inconsistencia en features wavelet: Se esperaban {len(feature_names)} pero se encontraron {num_features}")
        return

    fig, axs = plt.subplots(num_features, 1, figsize=(
        15, 12), sharex=True)

    for i in range(num_features):
        axs[i].plot(t, wavelet_features[:sample_len, i],
                    label=feature_names[i])
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle=':')

    axs[0].set_title(
        f"Transformación Wavelet ({wavelet_name}, Nivel {wavelet_level}) - Sensor 1 (Muestra)")
    axs[-1].set_xlabel("Muestras de Tiempo")

    plt.tight_layout()
    plot_path = os.path.join(
        output_dir, 'ZZ_INFO_2_Transformacion_Wavelet.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Gráfico de transformación Wavelet guardado en: {plot_path}")
    plt.close()


def plot_adjacency_heatmap(adj_matrix, output_dir):
    """
    NUEVO: Grafica la matriz de adyacencia (pesos del grafo) como un heatmap.
    """
    num_nodes = adj_matrix.shape[0]
    labels = [f'S{i + 1}' for i in range(num_nodes)]

    plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, cmap='viridis', norm=mcolors.LogNorm())
    plt.colorbar(label='Peso de Conexión (1/distancia)')
    plt.xticks(ticks=np.arange(num_nodes), labels=labels)
    plt.yticks(ticks=np.arange(num_nodes), labels=labels)
    plt.title('Heatmap de la Matriz de Adyacencia Física')

    # Añadir valores de texto
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                plt.text(j, i, f"{adj_matrix[i, j]:.2f}", ha='center', va='center', color='white', fontsize=8)

    plot_path = os.path.join(
        output_dir, 'ZZ_INFO_3_Matriz_Adyacencia.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Gráfico de heatmap de adyacencia guardado en: {plot_path}")
    plt.close()


# =============================================================================
# FUNCIÓN PRINCIPAL DE EVALUACIÓN
# =============================================================================


def run_evaluation(run_dir, healthy_data_dir, damage_data_dir):
    """
    Función principal de evaluación.
    """
    logger.info(f"--- Iniciando Evaluación del Modelo ---")
    logger.info(f"Cargando artefactos desde: {run_dir}")
    logger.info(f"Usando datos SANOS de: {healthy_data_dir}")
    logger.info(f"Usando datos de DAÑO de: {damage_data_dir}")

    # --- 1. Cargar Artefactos ---
    try:
        hp_path = os.path.join(run_dir, 'hyperparameters_stgae_physics.json')
        scaler_path = os.path.join(run_dir, 'scaler_stgae_physics.gz')
        model_path = os.path.join(run_dir, 'best_model_stgae_physics.pth')

        with open(hp_path, 'r') as f:
            hp = json.load(f)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        logger.error(f"Error: No se encontró un archivo requerido: {e}")
        return
    except Exception as e:
        logger.error(f"Error cargando artefactos: {e}", exc_info=True)
        return

    logger.info("Artefactos (HP, Scaler) cargados.")
    num_nodes = hp['num_nodes']

    # --- 2. Cargar, Procesar y Crear Datasets ---

    # --- Datos SANOS (para dataset de validación) ---
    logger.info("Cargando y procesando datos SANOS...")
    healthy_raw_dict, healthy_min_len = load_raw_data(
        healthy_data_dir, num_nodes)
    if healthy_raw_dict is None:
        return

    healthy_scaled_dict = process_data_with_scaler(
        healthy_raw_dict, healthy_min_len, hp, scaler)
    if healthy_scaled_dict is None:
        return

    full_dataset = SpatioTemporalWaveletDataset(
        healthy_scaled_dict, hp['window_size'], hp['stride'], num_nodes)

    val_split = 0.15
    val_len = int(val_split * len(full_dataset))
    train_len = len(full_dataset) - val_len
    _, val_dataset = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # Para evaluación, usamos batch_size=1
    healthy_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info(
        f"Dataset de Validación (Sano) listo. {len(val_dataset)} ventanas.")

    # --- Datos de DAÑO REAL ---
    logger.info("Cargando y procesando datos de DAÑO REAL...")
    damage_raw_dict, damage_min_len = load_raw_data(
        damage_data_dir, num_nodes)
    if damage_raw_dict is None:
        return

    damage_scaled_dict = process_data_with_scaler(
        damage_raw_dict, damage_min_len, hp, scaler)
    if damage_scaled_dict is None:
        return

    damage_dataset = SpatioTemporalWaveletDataset(
        damage_scaled_dict, hp['window_size'], hp['stride'], num_nodes)
    # Pequeño hack para identificar este dataset en el logger
    damage_dataset.is_damage_dataset = True

    damage_loader = DataLoader(damage_dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info(
        f"Dataset de DAÑO REAL listo. {len(damage_dataset)} ventanas.")

    # --- 3. Cargar Modelo y Grafo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    model = SpatioTemporalAutoencoder(hp['num_nodes'], hp['num_features'], hp['window_size'], hp.get(
        'gnn_hidden', 128), hp.get('gnn_out', 64), hp.get('rnn_hidden', 256), hp.get('rnn_layers', 2)).to(device)

    try:
        # Cargar con weights_only=True por seguridad
        model.load_state_dict(torch.load(
            model_path, map_location=device, weights_only=True))
        logger.info(f"Modelo cargado desde: {model_path}")
    except Exception as e:
        logger.error(f"Error cargando state_dict: {e}", exc_info=True)
        return

    graph_def = create_physics_informed_graph(num_nodes=hp['num_nodes'])
    edge_index = graph_def['edge_index'].to(device)
    edge_weight = graph_def['edge_weight'].to(device)
    logger.info("Grafo Físico cargado.")

    # --- 4. Generar Gráficas de Información ---
    plot_physical_graph(graph_def, run_dir)
    # Usar datos escalados sanos del S1 para el plot wavelet
    if 1 in healthy_raw_dict and 1 in healthy_scaled_dict:
        plot_wavelet_transform(
            healthy_raw_dict[1], healthy_scaled_dict[1], hp, run_dir)
    else:
        logger.warning("No se encontraron datos para S1, omitiendo gráfico wavelet.")

    # NUEVO: Heatmap de Adyacencia
    plot_adjacency_heatmap(graph_def['adj_matrix'], run_dir)

    # --- 5. Calcular Errores (Sano vs Daño) ---
    errors_per_sensor_dict = {}

    # Calcular errores SANOS por sensor
    healthy_errors_per_sensor = get_reconstruction_errors_per_sensor(
        model, healthy_loader, device, edge_index, edge_weight)
    errors_per_sensor_dict['Sano'] = healthy_errors_per_sensor
    # También calcular el error total (promedio de sensores)
    healthy_errors_total = np.mean(healthy_errors_per_sensor, axis=1)

    logger.info(
        f"Errores 'Sanos' calculados. Media (total): {np.mean(healthy_errors_total):.6f}")

    # Calcular errores de DAÑO REAL por sensor
    damage_errors_per_sensor = get_reconstruction_errors_per_sensor(
        model, damage_loader, device, edge_index, edge_weight)
    errors_per_sensor_dict['Daño Real'] = damage_errors_per_sensor
    # También calcular el error total (promedio de sensores)
    damage_errors_total = np.mean(damage_errors_per_sensor, axis=1)

    logger.info(
        f"Errores 'Daño Real' calculados. Media (total): {np.mean(damage_errors_total):.6f}")

    # --- 6. Graficar Resultados ---

    # Histograma
    errors_total_dict = {
        'Sano': healthy_errors_total,
        'Daño Real': damage_errors_total
    }
    plot_error_histograms(errors_total_dict, run_dir)

    # Localización
    plot_error_localization(errors_per_sensor_dict, run_dir)

    # NUEVO: Error en el tiempo
    plot_error_over_time(errors_total_dict, hp, run_dir)

    logger.info("--- Evaluación Finalizada ---")
    logger.info(
        f"Todas las gráficas de resultados han sido guardadas en: {run_dir}")


# --- BLOQUE DE EJECUCIÓN ---
if __name__ == '__main__':
    # --- MODIFICADO PARA EJECUCIÓN DIRECTA EN PYCHARM ---

    # 1. Ruta a la carpeta de resultados de tu mejor modelo (el que acaba de terminar)
    RUN_DIRECTORY = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347"

    # 2. Ruta a la carpeta de datos crudos (SANOS)
    HEALTHY_DATA_DIRECTORY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # 3. Ruta a la carpeta de datos crudos (DAÑADOS)
    DAMAGE_DATA_DIRECTORY = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"

    # --- Validación de rutas ---
    if not os.path.isdir(RUN_DIRECTORY):
        print(f"Error: Directorio de resultados no encontrado: {RUN_DIRECTORY}")
        sys.exit(1)
    if not os.path.isdir(HEALTHY_DATA_DIRECTORY):
        print(f"Error: Directorio de datos SANOS no encontrado: {HEALTHY_DATA_DIRECTORY}")
        sys.exit(1)
    if not os.path.isdir(DAMAGE_DATA_DIRECTORY):
        print(f"Error: Directorio de datos DAÑADOS no encontrado: {DAMAGE_DATA_DIRECTORY}")
        sys.exit(1)

    try:
        # Llamar a la función de evaluación con las rutas hardcodeadas
        run_evaluation(
            run_dir=RUN_DIRECTORY,
            healthy_data_dir=HEALTHY_DATA_DIRECTORY,
            damage_data_dir=DAMAGE_DATA_DIRECTORY
        )
    except Exception as e:
        logger.critical(
            f"Error fatal durante la evaluación: {e}", exc_info=True)
        sys.exit(1)

