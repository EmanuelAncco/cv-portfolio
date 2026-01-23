# -*- coding: utf-8 -*-
"""
run_inference_wavelet_corregido.py

Script completo para inferencia y análisis de un modelo STG-AE (Wavelet-GNN).
Carga un modelo entrenado, su scaler y HPs, y ejecuta la inferencia en
datos "sanos" (baseline) y datos "con daño".

Genera una suite completa de gráficos analíticos comparando ambos escenarios,
incluyendo MSE, SSIM, distribuciones de error, heatmaps, visualizaciones
del grafo del puente y visualización de las features wavelet.

Correcciones:
- Arreglado 'import gc'
- Arreglado ploteo en escala logarítmica para 'plot_mse_comparison' y 'plot_training_history'
- Separado 'plot_error_timeseries' en 'plot_error_timeseries_sano' y 'plot_error_timeseries_dano'.
- Añadido 'plot_wavelet_features_ejemplo' para visualizar las features DWT.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
import seaborn as sns
import json
import joblib
import logging
from tqdm import tqdm
from datetime import datetime
import re
import gc  # Garbage Collector (Corregido)

# --- NUEVAS IMPORTACIONES (del script 'inferencia.txt') ---
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("Error: torch_geometric no está instalado. Por favor, instálalo con 'pip install torch_geometric'")
    sys.exit(1)
try:
    # Necesaria para calcular el Structural Similarity Index (SSIM)
    from skimage.metrics import structural_similarity
except ImportError:
    print("Error: scikit-image no está instalado. Por favor, instálalo con 'pip install scikit-image'")
    sys.exit(1)
try:
    # Para dibujar el grafo
    import networkx as nx
except ImportError:
    print("Error: networkx no está instalado. Por favor, instálalo con 'pip install networkx'")
    sys.exit(1)
# --- Importación de Wavelets ---
try:
    import pywt
except ImportError:
    print("Error: PyWavelets no está instalado. Por favor, instálalo con 'pip install PyWavelets'")
    sys.exit(1)

# --- Configuración del Logging (Consola y Archivo) ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()  # Logger Raíz
logger.setLevel(logging.INFO)
# Limpiar handlers existentes para evitar duplicados
for handler in logger.handlers[:]: logger.removeHandler(handler)
# Handler de Consola
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# El FileHandler se añadirá en run_inference_and_plot

# --- DEFINICIONES DE CLASES Y FUNCIONES (del script de entrenamiento Wavelet) ---

def define_bridge_graph(num_nodes=5):
    """Define la estructura del grafo del puente."""
    if num_nodes != 5:
        logger.warning(
            f"define_bridge_graph está codificado para 5 nodos, pero se pidieron {num_nodes}. Usando la topología de 5 nodos.")
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    if edge_index.max() >= num_nodes:
        raise ValueError(
            f"Índice de nodo {edge_index.max()} en edge_index es inválido para {num_nodes} nodos (índices deben ser de 0 a {num_nodes - 1}).")
    return edge_index


class SpatioTemporalWaveletDataset(Dataset):
    """Dataset para cargar ventanas espacio-temporales con características Wavelet."""

    def __init__(self, data_dict_features, window_size, stride=1, num_expected_nodes=5):
        self.window_size = window_size
        self.stride = stride
        self.num_expected_nodes = num_expected_nodes
        local_logger = logging.getLogger(self.__class__.__name__)

        if not data_dict_features:
            local_logger.error("El diccionario de datos de características está vacío.")
            raise ValueError("El diccionario de datos de características está vacío.")

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
                        f"Inconsistencia en número de features. Sensor {sid} tiene {current_num_features}, se esperaban {expected_num_features}. Omitiendo.")
                    continue
                valid_data_dict[sid] = data
                min_len = min(min_len, data.shape[0])
            else:
                shape_info = data.shape if hasattr(data, 'shape') else 'N/A'
                len_info = data.shape[0] if hasattr(data, 'shape') else 'N/A'
                local_logger.warning(
                    f"Datos inválidos/insuficientes para sensor {sid}. Shape: {shape_info}, Longitud: {len_info}, WinSize: {window_size}. Omitiendo.")

        if not valid_data_dict:
            local_logger.error("No hay datos válidos en el diccionario después de filtrar.")
            raise ValueError("No hay datos válidos en el diccionario después de filtrar.")

        if min_len == float('inf') or min_len < window_size:
            len_val = min_len if min_len != float('inf') else 'N/A'
            local_logger.error(f"Longitud mínima ({len_val}) insuficiente para window_size ({window_size}).")
            raise ValueError(f"Longitud mínima ({len_val}) insuficiente para window_size ({window_size}).")

        processed_data_list = []
        actual_node_ids = []
        self.num_features = 0
        for sid in range(1, self.num_expected_nodes + 1):
            if sid in valid_data_dict:
                data_node = valid_data_dict[sid][:min_len]  # Truncar
                processed_data_list.append(data_node)
                actual_node_ids.append(sid)
                if self.num_features == 0:
                    self.num_features = data_node.shape[1]
            else:
                local_logger.error(
                    f"Faltan datos procesados para el sensor esperado {sid}. La arquitectura requiere datos de todos los nodos. Abortando.")
                raise ValueError(f"Faltan datos para el sensor esperado {sid}.")

        if not processed_data_list:
            local_logger.error("La lista de datos procesados está vacía.")
            raise ValueError("La lista de datos procesados está vacía.")

        try:
            self.data = np.stack(processed_data_list, axis=1)  # Shape: (min_len, num_nodes, num_features)
        except ValueError as e:
            shapes_str = ", ".join([str(d.shape) for d in processed_data_list])
            local_logger.error(
                f"Error al apilar datos (np.stack axis=1). Shapes individuales: [{shapes_str}]. Error: {e}")
            raise e

        self.num_nodes = self.data.shape[1]
        local_logger.info(f"Datos apilados con shape final: {self.data.shape}. Sensores usados: {actual_node_ids}")

        if self.num_nodes != self.num_expected_nodes:
            local_logger.error(
                f"Inconsistencia crítica: Nodos procesados ({self.num_nodes}) != Nodos esperados ({self.num_expected_nodes}).")
            raise RuntimeError("Error interno en la creación del dataset: Conteo de nodos inconsistente.")

        self.n_samples = max(0, (self.data.shape[0] - window_size) // stride + 1)
        if self.n_samples == 0:
            local_logger.warning(
                f"Número de muestras es 0. Longitud datos: {self.data.shape[0]}, WinSize: {window_size}, Stride: {stride}.")
        local_logger.info(
            f"Dataset creado con {self.num_nodes} nodos, {self.num_features} features, {self.data.shape[0]} puntos, {self.n_samples} ventanas.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        if start < 0 or end > self.data.shape[0]:
            local_logger = logging.getLogger(self.__class__.__name__)
            local_logger.error(
                f"Índice {idx} genera rango [{start}, {end}) fuera de límites [{0}, {self.data.shape[0]}].")
            raise IndexError(f"Índice {idx} fuera de rango.")
        window = self.data[start:end]  # Shape: (window_size, num_nodes, num_features)
        return torch.FloatTensor(window), torch.FloatTensor(window)


class GNNLayer(nn.Module):
    """Bloque GCN (idéntico al de entrenamiento)."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index):
        edge_index = edge_index.to(x.device)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SpatioTemporalAutoencoder(nn.Module):
    """Arquitectura ST-GAE (idéntica a la de entrenamiento)."""

    def __init__(self, num_nodes, num_features, window_size, gnn_hidden=32, gnn_out=16, rnn_hidden=64, rnn_layers=2):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_features = num_features
        self.gnn_hidden_dim = gnn_hidden
        self.gnn_encoder_out_dim = gnn_out
        self.rnn_encoder_hidden_dim = rnn_hidden
        self.rnn_layers = rnn_layers
        self.rnn_decoder_output_dim = self.gnn_hidden_dim * num_nodes
        local_logger = logging.getLogger(self.__class__.__name__)
        local_logger.info(f"Initializing STAutoencoder: N={num_nodes}, F={num_features}, T={window_size}")
        local_logger.info(f"  GNN Encoder: {num_features} -> {self.gnn_hidden_dim} -> {self.gnn_encoder_out_dim}")
        local_logger.info(
            f"  RNN Encoder: Input={self.gnn_encoder_out_dim * num_nodes}, Hidden={self.rnn_encoder_hidden_dim}, Layers={self.rnn_layers}")
        local_logger.info(
            f"  RNN Decoder: Input={self.rnn_encoder_hidden_dim}, Hidden={self.rnn_decoder_output_dim}, Layers={self.rnn_layers}")
        local_logger.info(f"  GNN Decoder: {self.gnn_hidden_dim} -> {self.gnn_hidden_dim} -> {num_features}")
        self.gnn_encoder = GNNLayer(num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)
        self.rnn_encoder = nn.GRU(input_size=self.gnn_encoder_out_dim * num_nodes,
                                  hidden_size=self.rnn_encoder_hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.rnn_decoder = nn.GRU(input_size=self.rnn_encoder_hidden_dim, hidden_size=self.rnn_decoder_output_dim,
                                  batch_first=True, num_layers=self.rnn_layers)
        self.gnn_decoder = GNNLayer(self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)

    def forward(self, x, edge_index):
        batch_size, T_actual, N_actual, F_actual = x.shape
        if T_actual != self.window_size or N_actual != self.num_nodes or F_actual != self.num_features:
            logger.warning(
                f"Unexpected input shape in forward: {x.shape}. Expected T={self.window_size}, N={self.num_nodes}, F={self.num_features}.")
        x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)
        edge_index = edge_index.to(x.device)
        try:
            gnn_encoded = self.gnn_encoder(x_reshaped, edge_index)
        except Exception as e:
            logger.error(f"Error en GNN Encoder. Input shape: {x_reshaped.shape}. Error: {e}", exc_info=True);
            raise e
        try:
            gnn_encoded_view = gnn_encoded.reshape(batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
            rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)
        except Exception as e:
            logger.error(f"Error en reshape pre-RNN Encoder. GNN Encoded shape: {gnn_encoded.shape}. Error: {e}",
                         exc_info=True);
            raise e
        try:
            _, h_n = self.rnn_encoder(rnn_input)
        except Exception as e:
            logger.error(f"Error en RNN Encoder. Input shape: {rnn_input.shape}. Error: {e}", exc_info=True);
            raise e
        try:
            latent_vector = h_n[-1].unsqueeze(1).repeat(1, T_actual, 1)
            rnn_decoded, _ = self.rnn_decoder(latent_vector)
        except Exception as e:
            logger.error(
                f"Error en RNN Decoder. Latent shape: {latent_vector.shape if 'latent_vector' in locals() else 'N/A'}. Error: {e}",
                exc_info=True);
            raise e
        try:
            gnn_input_decoder = rnn_decoded.reshape(batch_size * T_actual, N_actual, self.gnn_hidden_dim)
        except Exception as e:
            logger.error(f"Error en reshape pre-GNN Decoder. RNN Decoded shape: {rnn_decoded.shape}. Error: {e}",
                         exc_info=True);
            raise e
        try:
            reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index)
        except Exception as e:
            logger.error(f"Error en GNN Decoder. Input shape: {gnn_input_decoder.shape}. Error: {e}", exc_info=True);
            raise e
        try:
            reconstructed_x = reconstructed_frames.reshape(batch_size, T_actual, N_actual, F_actual)
        except Exception as e:
            logger.error(
                f"Error en reshape final. Reconstructed frames shape: {reconstructed_frames.shape}. Target: ({batch_size}, {T_actual}, {N_actual}, {F_actual}). Error: {e}",
                exc_info=True);
            raise e
        return reconstructed_x


def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    """Aplica DWT multinivel y reconstruye bandas (idéntico al de entrenamiento)."""
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 1 or len(signal) == 0:
        logger.warning("apply_dwt_features: Señal de entrada inválida.")
        return None
    original_len = len(signal)
    if target_len is None:
        target_len = original_len
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        reconstructed_bands = []
        # Reconstruir Bandas de Detalle (D1 a D_level)
        for i in range(level, 0, -1):
            detail_level_index = level - i + 1  # Esto es 1 para D1, 2 para D2, ... level para D_level
            if detail_level_index < 0 or detail_level_index >= len(coeffs):
                logger.error(
                    f"Índice de nivel de detalle {detail_level_index} fuera de rango para coeffs (len={len(coeffs)}).")
                return None
            detail_coeffs_list = [np.zeros_like(c) for c in coeffs]
            detail_coeffs_list[detail_level_index] = coeffs[detail_level_index]
            rec_d = pywt.waverec(detail_coeffs_list, wavelet)
            rec_d_adj = adjust_signal_length(rec_d, target_len)
            reconstructed_bands.append(rec_d_adj)

        # Reconstruir Banda de Aproximación (A_level)
        approx_coeffs_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_coeffs_list, wavelet)
        rec_a_adj = adjust_signal_length(rec_a, target_len)

        # Invertir bandas de detalle para tener D1, D2, ..., D_level
        reconstructed_bands.reverse()

        # Ajustar señal original
        original_adjusted = adjust_signal_length(signal, target_len)

        # Orden final: [Original, A_level, D1, D2, ..., D_level]
        all_bands = [original_adjusted, rec_a_adj] + reconstructed_bands

        features = np.stack(all_bands, axis=-1)
        expected_feature_count = 1 + 1 + level
        if features.shape != (target_len, expected_feature_count):
            logger.warning(
                f"Shape inesperado en features wavelet: {features.shape}. Esperado: ({target_len}, {expected_feature_count})")
        return features
    except ValueError as ve:
        logger.error(f"Error de valor aplicando DWT (posiblemente nivel muy alto para longitud {original_len}): {ve}",
                     exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error inesperado aplicando DWT a señal de longitud {original_len}: {e}", exc_info=True)
        return None


def adjust_signal_length(signal, target_len):
    """Ajusta la longitud de una señal 1D (padding o truncado)."""
    current_len = len(signal)
    if current_len == target_len:
        return signal
    elif current_len > target_len:
        return signal[:target_len]
    else:
        padding = np.zeros(target_len - current_len)
        return np.concatenate((signal, padding))


# --- FUNCIÓN DE CARGA DE DATOS (ADAPTADA PARA WAVELETS Y MODIFICADA) ---
def load_and_preprocess_wavelet_data(data_directory, hp, scaler, num_expected_nodes=5, target_len=None):
    """
    Carga, aplica DWT y escala datos desde un directorio.
    Si target_len no se provee, lo calcula (para datos 'healthy').
    Si target_len se provee, lo usa (para datos 'damage').

    MODIFICADO: Devuelve (scaled_data, unscaled_data, target_len)
    """
    wavelet_name = hp.get('wavelet_name', 'db4')
    wavelet_level = hp.get('wavelet_level', 5)

    logger.info(f"Iniciando carga de datos crudos desde: {data_directory}")
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_directory}")
        return None, None, 0

    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    for f_path in tqdm(all_files, desc=f"Cargando archivos crudos ({os.path.basename(data_directory)})"):
        try:
            filename = os.path.basename(f_path)
            sid = int(filename.split('_')[0])
            if sid in sensor_data_raw:
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[1], engine='python',
                                   on_bad_lines='warn').values
                if data is not None and data.size > 0:
                    sensor_data_raw[sid].append(data)
        except Exception as e:
            logger.warning(f"Error cargando '{f_path}': {e}. Omitiendo.")

    logger.info("Concatenando datos crudos...")
    sensor_data_concat = {}
    min_len_raw = float('inf')
    sensors_with_data = []
    for sid, data_list in sensor_data_raw.items():
        if data_list:
            data_list_valid = [d for d in data_list if d is not None and d.size > 0]
            if not data_list_valid: continue
            data_list_2d = [d.reshape(-1, 1) if d.ndim == 1 else d for d in data_list_valid]
            if any(d.shape[1] != 1 for d in data_list_2d): continue
            try:
                concatenated_data = np.concatenate(data_list_2d, axis=0)
                if concatenated_data.size > 0:
                    sensor_data_concat[sid] = concatenated_data.squeeze()  # 1D para DWT
                    min_len_raw = min(min_len_raw, len(concatenated_data))
                    sensors_with_data.append(sid)
            except Exception as e:
                logger.error(f"Error concatenando datos para sensor {sid}: {e}.")
        else:
            logger.warning(f"No se cargaron datos válidos para el sensor {sid}.")

    if len(sensors_with_data) != num_expected_nodes:
        missing_sensors = set(range(1, num_expected_nodes + 1)) - set(sensors_with_data)
        logger.error(f"Faltan datos concatenados para los sensores: {missing_sensors}. No se puede continuar.")
        return None, None, 0

    if target_len is None:
        # Si es la primera carga (healthy), definimos el target_len
        target_len = min_len_raw
        logger.info(f"Longitud mínima (target_len) definida en: {target_len}")
    else:
        # Si es la segunda carga (damage), usamos el target_len provisto
        logger.info(f"Usando target_len provisto: {target_len} (truncando si es necesario)")

    if target_len == float('inf') or target_len < hp.get('window_size', 64):
        logger.error(f"Longitud objetivo ({target_len}) es inválida o insuficiente.")
        return None, None, 0

    logger.info("Aplicando DWT y reconstrucción de bandas...")
    sensor_data_features = {}  # Diccionario para datos SIN escalar
    actual_num_features_generated = 0
    for sid in tqdm(sensors_with_data, desc="Generando Features Wavelet"):
        signal_1d = sensor_data_concat[sid]
        # Aplicar truncado ANTES de DWT si la señal es más larga que target_len
        if len(signal_1d) > target_len:
            signal_1d = signal_1d[:target_len]

        features_2d = apply_dwt_features(signal_1d, wavelet=wavelet_name, level=wavelet_level, target_len=target_len)
        if features_2d is None:
            logger.error(f"Error generando features para sensor {sid}.")
            return None, None, 0
        if actual_num_features_generated == 0:
            actual_num_features_generated = features_2d.shape[1]
        sensor_data_features[sid] = features_2d

    if actual_num_features_generated != hp.get('num_features'):
        logger.warning(
            f"Inconsistencia de features: Generadas={actual_num_features_generated}, Esperadas en HP={hp.get('num_features')}")

    del sensor_data_raw, sensor_data_concat
    gc.collect()

    logger.info("Aplicando scaler cargado...")
    sensor_data_scaled_features = {}
    for sid in sensors_with_data:
        try:
            data_to_scale = sensor_data_features.get(sid)
            if data_to_scale is None: continue
            # Validar que el scaler tenga el n_features correcto
            if scaler.n_features_in_ != data_to_scale.shape[1]:
                logger.error(
                    f"Incompatibilidad del Scaler: Scaler espera {scaler.n_features_in_} features, pero los datos tienen {data_to_scale.shape[1]}.")
                return None, None, 0
            scaled_data = scaler.transform(data_to_scale)
            sensor_data_scaled_features[sid] = scaled_data
        except Exception as e:
            logger.error(f"Error escalando features para sensor {sid}: {e}. Omitiendo sensor.")
            return None, None, 0

    logger.info("Pre-procesamiento de datos completado.")
    # Devolvemos los datos escalados, los sin escalar y el target_len usado
    return sensor_data_scaled_features, sensor_data_features, target_len


# --- FUNCIÓN DE INFERENCIA (ADAPTADA DE 'inferencia.txt') ---
def perform_inference(model, dataloader, device, edge_index, hp):
    """
    Ejecuta la inferencia y calcula errores (MSE) y SSIM.
    Adaptado para el modelo Wavelet (F=7).
    """
    local_logger = logging.getLogger(f"{__name__}.inference")
    model.eval()
    all_inputs_np = []
    all_outputs_np = []
    all_losses_np = []  # Pérdida promedio por ventana (MSE)
    all_losses_per_sensor_np = []  # Array (n_samples, n_nodes) (MSE)
    all_ssim_per_sensor_np = []  # Array (n_samples, n_nodes) (SSIM)

    criterion_none = nn.MSELoss(reduction='none')

    total_batches = len(dataloader)
    if total_batches == 0:
        local_logger.error("DataLoader is empty. No data for inference.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    processed_windows = 0
    num_nodes = dataloader.dataset.num_nodes
    num_features = dataloader.dataset.num_features
    window_size = dataloader.dataset.window_size

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Inference', leave=False, total=total_batches)
        for batch_idx, (inputs, _) in enumerate(progress_bar):
            if inputs is None or len(inputs) == 0:
                local_logger.warning(f"Batch {batch_idx + 1}/{total_batches} empty. Skipping.")
                continue

            # Validar shape
            if inputs.shape[1] != window_size or inputs.shape[2] != num_nodes or inputs.shape[3] != num_features:
                local_logger.warning(
                    f"Batch shape anómalo: {inputs.shape}. Esperado (B, {window_size}, {num_nodes}, {num_features}). Saltando batch.")
                continue

            inputs = inputs.to(device)  # Shape: (batch, time, nodes, feats)
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

                # --- Cálculo de SSIM (basado en la característica 0, la señal original) ---
                inputs_cpu = inputs.cpu().numpy()
                outputs_cpu = outputs.cpu().numpy()
                batch_size = inputs_cpu.shape[0]

                batch_ssim_per_sensor = np.zeros((batch_size, num_nodes))

                for i in range(batch_size):
                    for n in range(num_nodes):
                        # Extraemos solo la CARACTERÍSTICA 0 (señal original)
                        sig_in = inputs_cpu[i, :, n, 0]  # Señal original (ventana, sensor n, feat 0)
                        sig_out = outputs_cpu[i, :, n, 0]  # Señal reconstruida (ventana, sensor n, feat 0)

                        data_range = sig_in.max() - sig_in.min()
                        if data_range < 1e-9: data_range = 1.0  # Evitar división por cero si la señal es plana

                        current_win_size = min(7, window_size)  # SSIM win_size no puede ser > len(signal)
                        if current_win_size < 3:
                            ssim_val = 1.0 if np.allclose(sig_in, sig_out) else 0.0
                        else:
                            if current_win_size % 2 == 0: current_win_size -= 1  # win_size debe ser impar
                            try:
                                ssim_val = structural_similarity(sig_in, sig_out,
                                                                 data_range=data_range,
                                                                 win_size=current_win_size)
                            except ValueError as ve:
                                local_logger.warning(
                                    f"SSIM calculation failed for window {processed_windows + i}, sensor {n + 1}. Setting to 0. Error: {ve}")
                                ssim_val = 0.0
                        batch_ssim_per_sensor[i, n] = ssim_val

                all_ssim_per_sensor_np.append(batch_ssim_per_sensor)
                # --- FIN DE CÁLCULO SSIM ---

                processed_windows += len(inputs)
                progress_bar.set_postfix({'Windows': processed_windows})

            except Exception as e:
                local_logger.error(f"Error during inference on batch {batch_idx + 1}/{total_batches}: {e}",
                                   exc_info=True)
                continue

    if not all_losses_np:
        local_logger.error("No windows were processed successfully.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    all_inputs_np = np.concatenate(all_inputs_np, axis=0)
    all_outputs_np = np.concatenate(all_outputs_np, axis=0)
    all_losses_np = np.concatenate(all_losses_np, axis=0)  # Shape: (n_samples,)
    all_losses_per_sensor_np = np.concatenate(all_losses_per_sensor_np, axis=0)  # Shape: (n_samples, n_nodes)
    all_ssim_per_sensor_np = np.concatenate(all_ssim_per_sensor_np, axis=0)  # Shape: (n_samples, n_nodes)

    local_logger.info(f"Inference completed. Processed {processed_windows} windows.")
    local_logger.info(f"Final shape of inputs/outputs: {all_inputs_np.shape}")
    local_logger.info(f"Final shape of losses per window: {all_losses_np.shape}")
    local_logger.info(f"Final shape of losses per sensor: {all_losses_per_sensor_np.shape}")
    local_logger.info(f"Final shape of SSIM per sensor: {all_ssim_per_sensor_np.shape}")

    return all_inputs_np, all_outputs_np, all_losses_np, all_losses_per_sensor_np, all_ssim_per_sensor_np


# --- FUNCIONES DE PLOTEO (ADAPTADAS DE 'inferencia.txt') ---
local_logger_plot = logging.getLogger(f"{__name__}.plotting")
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    local_logger_plot.warning("Style 'seaborn-v0_8-whitegrid' not found, using 'ggplot'.")
    plt.style.use('ggplot')

palette_scenario = sns.color_palette("Set1", 2)
sensor_colormap = plt.cm.viridis
num_sensors_global = 5
sensor_colors_mapped = sensor_colormap(np.linspace(0.1, 0.9, num_sensors_global))


def plot_sensor_reconstruction_samples(all_originals, all_reconstructions, scaler, num_sensors, output_dir, prefix):
    """
    Genera un gráfico por CADA sensor, mostrando la reconstrucción de
    DOS muestras aleatorias (ventanas) y sus errores.
    all_originals, all_reconstructions: shape (n_samples, time, nodes, 7_features)
    """
    local_logger_plot.info(f"Generando gráficos de reconstrucción por sensor para datos {prefix}...")
    if all_originals is None or all_reconstructions is None or all_originals.size == 0 or all_reconstructions.size == 0:
        local_logger_plot.warning(
            f"Datos originales o reconstruidos faltantes/vacíos ({prefix}). No se generarán gráficos por sensor.")
        return

    num_samples, time_steps, _, num_features = all_originals.shape
    if num_samples < 2:
        local_logger_plot.warning(
            f"Se necesitan al menos 2 muestras para graficar, pero se encontraron {num_samples} ({prefix}). Saltando.")
        return

    color1 = '#377eb8'  # Azul
    color2 = '#4daf4a'  # Verde
    error_color = '#e41a1c'  # Rojo

    for i in range(num_sensors):
        try:
            idx1, idx2 = np.random.choice(num_samples, 2, replace=False)
        except ValueError as e:
            local_logger_plot.error(
                f"Error seleccionando índices aleatorios para sensor {i + 1} ({prefix}): {e}. Saltando sensor.")
            continue

        # --- Preparar datos (Muestra 1) ---
        original_sample1_scaled = all_originals[idx1, :, i, :]  # Shape (time, 7_features)
        reconstructed_sample1_scaled = all_reconstructions[idx1, :, i, :]  # Shape (time, 7_features)
        # --- Preparar datos (Muestra 2) ---
        original_sample2_scaled = all_originals[idx2, :, i, :]  # Shape (time, 7_features)
        reconstructed_sample2_scaled = all_reconstructions[idx2, :, i, :]  # Shape (time, 7_features)

        try:
            # --- INICIO DE LA TRANSFORMACIÓN INVERSA (7 FEATURES) ---
            original_sample1_inv_all_feats = scaler.inverse_transform(original_sample1_scaled)
            reconstructed_sample1_inv_all_feats = scaler.inverse_transform(reconstructed_sample1_scaled)
            original_sample2_inv_all_feats = scaler.inverse_transform(original_sample2_scaled)
            reconstructed_sample2_inv_all_feats = scaler.inverse_transform(reconstructed_sample2_scaled)

            # --- EXTRAER SOLO LA CARACTERÍSTICA 0 (SEÑAL ORIGINAL) PARA PLOTEAR ---
            original_sample1_inv = original_sample1_inv_all_feats[:, 0]  # Shape (time,)
            reconstructed_sample1_inv = reconstructed_sample1_inv_all_feats[:, 0]
            original_sample2_inv = original_sample2_inv_all_feats[:, 0]
            reconstructed_sample2_inv = reconstructed_sample2_inv_all_feats[:, 0]
            # --- FIN DE LA ADAPTACIÓN ---
        except Exception as e:
            local_logger_plot.error(
                f"Error during inverse_transform in plot_sensor_reconstruction_samples: {e}. Graficando datos escalados (feature 0).")
            # Fallback a plotear datos escalados (solo feature 0)
            original_sample1_inv = original_sample1_scaled[:, 0]
            reconstructed_sample1_inv = reconstructed_sample1_scaled[:, 0]
            original_sample2_inv = original_sample2_scaled[:, 0]
            reconstructed_sample2_inv = reconstructed_sample2_scaled[:, 0]

        # Calcular error sobre los datos invertidos (en unidades reales)
        error_signal_1 = original_sample1_inv - reconstructed_sample1_inv
        error_signal_2 = original_sample2_inv - reconstructed_sample2_inv

        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, squeeze=False)
        fig.suptitle(f'Sensor {i + 1}: Reconstrucción de Dos Muestras Aleatorias (Datos {prefix.capitalize()})',
                     fontsize=16)

        # --- Muestra 1 (Fila 0) ---
        ax_sig1 = axes[0, 0]
        ax_err1 = axes[0, 1]
        ax_sig1.plot(original_sample1_inv, label=f'Original (Muestra {idx1})', color=color1, linewidth=1.5, alpha=0.8)
        ax_sig1.plot(reconstructed_sample1_inv, label=f'Reconstruida (Muestra {idx1})', color=color1, linestyle='--',
                     linewidth=1.5, alpha=1.0)
        ax_sig1.set_title(f'Muestra 1 (Ventana: {idx1}) - Original vs. Reconstruida (Feature 0)')
        ax_sig1.set_ylabel('Valor Original (Invertido)')
        ax_sig1.legend(fontsize='small')
        ax_sig1.grid(True, linestyle=':')
        ax_err1.plot(error_signal_1, label='Error', color=error_color, linewidth=1.5)
        ax_err1.set_title(f'Muestra 1 (Ventana: {idx1}) - Error de Reconstrucción')
        ax_err1.set_ylabel('Error (Unidades Originales)')
        ax_err1.grid(True, linestyle=':')
        ax_err1.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax_err1.legend(fontsize='small')

        # --- Muestra 2 (Fila 1) ---
        ax_sig2 = axes[1, 0]
        ax_err2 = axes[1, 1]
        ax_sig2.plot(original_sample2_inv, label=f'Original (Muestra {idx2})', color=color2, linewidth=1.5, alpha=0.8)
        ax_sig2.plot(reconstructed_sample2_inv, label=f'Reconstruida (Muestra {idx2})', color=color2, linestyle='--',
                     linewidth=1.5, alpha=1.0)
        ax_sig2.set_title(f'Muestra 2 (Ventana: {idx2}) - Original vs. Reconstruida (Feature 0)')
        ax_sig2.set_ylabel('Valor Original (Invertido)')
        ax_sig2.legend(fontsize='small')
        ax_sig2.grid(True, linestyle=':')
        ax_err2.plot(error_signal_2, label='Error', color=error_color, linewidth=1.5)
        ax_err2.set_title(f'Muestra 2 (Ventana: {idx2}) - Error de Reconstrucción')
        ax_err2.set_ylabel('Error (Unidades Originales)')
        ax_err2.grid(True, linestyle=':')
        ax_err2.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax_err2.legend(fontsize='small')

        axes[1, 0].set_xlabel('Time Step en Ventana')
        axes[1, 1].set_xlabel('Time Step en Ventana')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        filename = os.path.join(output_dir, f"{prefix}_sensor_{i + 1}_reconstruccion_muestras.png")
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            local_logger_plot.info(f"Gráfico de reconstrucción por sensor guardado en: {filename}")
        except Exception as e:
            local_logger_plot.error(f"Error guardando gráfico {filename}: {e}")
        plt.close(fig)


# --- NUEVA FUNCIÓN (SOLICITADA) ---
def plot_wavelet_features_ejemplo(sensor_data_features_unscaled, output_dir, hp, num_nodes=5):
    """
    Genera un gráfico de ejemplo de las características wavelet (sin escalar)
    para un sensor (ej. Sensor 1).
    """
    local_logger_plot.info("Generando gráfico de ejemplo de features wavelet...")
    if not sensor_data_features_unscaled or 1 not in sensor_data_features_unscaled:
        local_logger_plot.warning("No hay datos de features (sin escalar) para el sensor 1. Saltando gráfico wavelet.")
        return

    data_to_plot = sensor_data_features_unscaled[1]  # Usar Sensor 1 como ejemplo
    level = hp.get('wavelet_level', 5)
    num_features = 1 + 1 + level

    if data_to_plot.shape[1] != num_features:
        local_logger_plot.error(
            f"El número de features ({data_to_plot.shape[1]}) no coincide con el esperado ({num_features}). Saltando gráfico.")
        return

    fig, axes = plt.subplots(num_features, 1, figsize=(18, 3 * num_features), sharex=True)
    fig.suptitle(f"Ejemplo de Features Wavelet (Sensor 1 - Datos 'Healthy' - Sin Escalar)", fontsize=16)

    try:
        # Feature 0: Original
        axes[0].plot(data_to_plot[:, 0], color='black', alpha=0.9)
        axes[0].set_title(f"Feature 0: Señal Original (Ajustada)")
        axes[0].set_ylabel("Amplitud")
        axes[0].grid(True, linestyle=':')

        # Feature 1: Aproximación
        axes[1].plot(data_to_plot[:, 1], color='blue', alpha=0.9)
        axes[1].set_title(f"Feature 1: Aproximación (A{level})")
        axes[1].set_ylabel("Amplitud")
        axes[1].grid(True, linestyle=':')

        # Features 2 en adelante: Detalles
        detail_colors = plt.cm.Reds(np.linspace(0.8, 0.4, level))
        for i in range(level):
            ax_idx = i + 2
            feat_idx = i + 2
            detail_level_label = i + 1  # D1, D2, ...
            axes[ax_idx].plot(data_to_plot[:, feat_idx], color=detail_colors[i], alpha=0.9)
            axes[ax_idx].set_title(f"Feature {feat_idx}: Detalle (D{detail_level_label})")
            axes[ax_idx].set_ylabel("Amplitud")
            axes[ax_idx].grid(True, linestyle=':')

        axes[-1].set_xlabel("Time Step (Longitud Total de la Señal)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        filename = os.path.join(output_dir, "wavelet_features_ejemplo_sensor1.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de features Wavelet guardado en: {filename}")

    except Exception as e:
        local_logger_plot.error(f"Error generando gráfico de features wavelet: {e}")
    finally:
        plt.close(fig)


# --- FUNCIÓN CORREGIDA ---
def plot_mse_comparison(mse_healthy_per_sensor_avg, mse_damage_per_sensor_avg, output_dir):
    """Genera gráfico comparativo de MSE promedio por sensor (sano vs. daño). (CORREGIDA)"""
    if mse_healthy_per_sensor_avg is None or mse_damage_per_sensor_avg is None or \
            mse_healthy_per_sensor_avg.size == 0 or mse_damage_per_sensor_avg.size == 0 or \
            len(mse_healthy_per_sensor_avg) != len(mse_damage_per_sensor_avg):
        local_logger_plot.error("Datos MSE promedio por sensor inválidos o inconsistentes para comparación.")
        return
    num_sensors = len(mse_healthy_per_sensor_avg)
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Detección de Daño: Error Promedio de Reconstrucción por Sensor', fontsize=16)

    # --- Plot Sano (Linear) ---
    axes[0].bar(sensor_labels, mse_healthy_per_sensor_avg, color=palette_scenario[0], edgecolor='black')
    axes[0].set_title('Datos Sanos (Baseline)')
    axes[0].set_ylabel('Mean Squared Error (MSE)')
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, axis='y', linestyle=':')
    max_healthy_val = np.max(mse_healthy_per_sensor_avg) if mse_healthy_per_sensor_avg.size > 0 else 1
    for i, v in enumerate(mse_healthy_per_sensor_avg):
        axes[0].text(i, v + 0.03 * max(max_healthy_val, 1e-9), f"{v:.4e}", ha='center', va='bottom', fontsize=9)

    # --- Plot Daño (Auto Log/Linear) - CORREGIDO ---
    axes[1].set_title('Datos con Daño')
    plot_data_damage = mse_damage_per_sensor_avg
    valid_damage_mse = plot_data_damage[plot_data_damage > 0]

    use_log_scale = False
    min_plot_val = 0

    if valid_damage_mse.size > 0:  # Si hay *algún* dato positivo
        min_pos_val = np.min(valid_damage_mse)
        max_pos_val = np.max(valid_damage_mse)
        # Usar log si el rango dinámico es grande
        if (max_pos_val / min_pos_val > 100) and min_pos_val > 1e-12:
            use_log_scale = True
            min_plot_val = min_pos_val * 0.1  # Límite inferior para log

    if use_log_scale:
        local_logger_plot.info("Usando escala logarítmica para el gráfico de comparación MSE (Daño).")
        plot_data_clipped = np.maximum(plot_data_damage, min_plot_val)  # Clipear datos en 0 para graficar en log
        axes[1].bar(sensor_labels, plot_data_clipped, color=palette_scenario[1], edgecolor='black')
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Mean Squared Error (MSE) (Log Scale)')
        axes[1].set_ylim(bottom=min_plot_val)
    else:
        local_logger_plot.info("Usando escala lineal para el gráfico de comparación MSE (Daño).")
        axes[1].bar(sensor_labels, plot_data_damage, color=palette_scenario[1], edgecolor='black')
        axes[1].set_ylabel('Mean Squared Error (MSE)')
        axes[1].set_ylim(bottom=0)

    axes[1].grid(True, axis='y', linestyle=':')

    # Anotaciones (usar valores originales 'v', no los clipeados)
    max_damage_val_for_annot = np.max(plot_data_damage) if plot_data_damage.size > 0 else 1
    for i, v in enumerate(plot_data_damage):
        if use_log_scale:
            # Posicionar texto encima de la barra (valor clipeado) o en el fondo si el valor es 0
            text_pos = max(v, min_plot_val) * 1.5
        else:
            text_pos = v + 0.03 * max(max_damage_val_for_annot, 1e-9)  # Posición en lineal
        axes[1].text(i, text_pos, f"{v:.3e}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, "damage_detection_mse_comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de comparación MSE guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico {filename}: {e}")
    plt.close(fig)


def plot_damage_localization(losses_damage_per_sensor, losses_healthy_per_sensor, output_dir):
    """Genera gráfico de localización (violin+box y factor de amplificación)."""
    if losses_damage_per_sensor is None or losses_healthy_per_sensor is None or \
            losses_damage_per_sensor.size == 0 or losses_healthy_per_sensor.size == 0 or \
            losses_damage_per_sensor.shape[1] != losses_healthy_per_sensor.shape[1]:
        local_logger_plot.error("Datos de pérdida por sensor inválidos o inconsistentes para localización.")
        return
    num_sensors = losses_damage_per_sensor.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Análisis de Localización de Daño por Sensor', fontsize=16)
    data_list = []
    for i in range(num_sensors):
        sensor_name = sensor_labels[i]
        positive_losses = losses_damage_per_sensor[:, i][losses_damage_per_sensor[:, i] > 0]
        if len(positive_losses) > 0:
            for loss_val in positive_losses:
                data_list.append({'Sensor': sensor_name, 'MSE': loss_val})
        else:
            data_list.append({'Sensor': sensor_name, 'MSE': 1e-12})  # Placeholder para que aparezca el sensor
    if not data_list or pd.DataFrame(data_list)['MSE'].max() <= 1e-11:
        local_logger_plot.error("No hay datos de error de daño positivos para graficar violinplot.")
        axes[0].set_title('Distribución de Error (Daño) (Sin datos positivos)')
    else:
        df_damage = pd.DataFrame(data_list)
        sns.violinplot(x='Sensor', y='MSE', data=df_damage, ax=axes[0],
                       palette=sensor_colors_mapped, inner=None,
                       cut=0, scale='width', linewidth=1.5, alpha=0.7)
        sns.boxplot(x='Sensor', y='MSE', data=df_damage, ax=axes[0],
                    showcaps=False, boxprops={'facecolor': 'None', "zorder": 10},
                    showfliers=False, whiskerprops={'linewidth': 2, "zorder": 10, 'color': 'black'},
                    medianprops={'linewidth': 2, "zorder": 10, 'color': 'black'}, width=0.3)
        axes[0].set_title('Distribución de Error de Reconstrucción (Datos con Daño - Violin + Box)')
        axes[0].set_ylabel('Mean Squared Error (MSE) - Log Scale')
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle=':')
        valid_mse = df_damage['MSE'][np.isfinite(df_damage['MSE'])]
        if len(valid_mse) > 0:
            p_low = np.percentile(valid_mse, 0.5);
            p_high = np.percentile(valid_mse, 99.5)
            min_mse_plot = max(p_low * 0.9, 1e-12);
            max_mse_plot = p_high * 1.1
            if min_mse_plot >= max_mse_plot:
                min_mse_plot = max(valid_mse.min() * 0.8, 1e-12);
                max_mse_plot = valid_mse.max() * 1.2
            axes[0].set_ylim(bottom=min_mse_plot, top=max_mse_plot)
    median_error_damage = np.median(losses_damage_per_sensor, axis=0)
    median_error_healthy = np.median(losses_healthy_per_sensor, axis=0)
    median_error_healthy_safe = np.maximum(median_error_healthy, 1e-10)  # Evitar división por cero
    amplification_factor = median_error_damage / median_error_healthy_safe
    bar_colors = [sensor_colors_mapped[i] for i in range(num_sensors)]
    axes[1].bar(sensor_labels, amplification_factor, color=bar_colors, edgecolor='black', alpha=0.85)
    axes[1].set_title('Factor de Amplificación de Error Mediano (Daño vs. Sano)')
    axes[1].set_ylabel('Error Mediano Daño / Error Mediano Sano')
    axes[1].grid(True, axis='y', linestyle=':')
    valid_amp = amplification_factor[np.isfinite(amplification_factor)]
    if len(valid_amp) > 0:
        max_amp = np.max(valid_amp)
        axes[1].set_ylim(bottom=min(0, np.min(valid_amp) * 0.9))
        for i, v in enumerate(amplification_factor):
            if np.isfinite(v):
                axes[1].text(i, v + 0.02 * max(max_amp, 1e-9), f"{v:.2f}", ha='center', va='bottom', fontsize=9.5)
    else:
        axes[1].set_ylim(bottom=0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "damage_localization_analysis_violinbox.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de localización de daño (Violin+Box) guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico {filename}: {e}")
    plt.close(fig)


def plot_error_distribution_kde(losses_healthy, losses_damage, output_dir):
    """Genera un gráfico KDE (Kernel Density Estimation)."""
    if losses_healthy is None or losses_damage is None or losses_healthy.size == 0 or losses_damage.size == 0:
        local_logger_plot.warning("Datos de pérdida vacíos (sano o daño), saltando gráfico KDE.")
        return
    local_logger_plot.info("Generando gráfico de distribución de error (KDE)...")
    plt.figure(figsize=(12, 7))
    epsilon = 1e-12
    log_losses_healthy = np.log10(losses_healthy[losses_healthy > 0] + epsilon)
    log_losses_damage = np.log10(losses_damage[losses_damage > 0] + epsilon)
    if len(log_losses_healthy) > 1:
        sns.kdeplot(log_losses_healthy, label='Sano', color=palette_scenario[0], fill=True, bw_adjust=0.5, alpha=0.6)
    else:
        local_logger_plot.warning("No hay suficientes datos positivos sanos para graficar KDE.")
    if len(log_losses_damage) > 1:
        sns.kdeplot(log_losses_damage, label='Daño', color=palette_scenario[1], fill=True, bw_adjust=0.5, alpha=0.6)
    else:
        local_logger_plot.warning("No hay suficientes datos positivos con daño para graficar KDE.")
    plt.title('Distribución de Error de Reconstrucción (MSE por Ventana)')
    plt.xlabel('Log10(Mean Squared Error)')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, linestyle=':')
    filename = os.path.join(output_dir, "error_distribution_kde.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico KDE guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico KDE {filename}: {e}")
    plt.close()


# --- FUNCIONES SEPARADAS (SOLICITADO) ---
def plot_error_timeseries_sano(losses_healthy, output_dir):
    """Genera un gráfico del error de reconstrucción (MSE) a lo largo del tiempo (ventanas) para datos SANOS."""
    if losses_healthy is None or losses_healthy.size == 0:
        local_logger_plot.warning("No hay datos de error 'sano' para graficar time series.")
        return
    local_logger_plot.info("Generando gráfico de serie temporal de error (Sano)...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle('Error de Reconstrucción (MSE) en el Tiempo (Ventanas) - Datos Sanos', fontsize=16)

    valid_healthy_indices = np.where(losses_healthy > 0)[0]
    if len(valid_healthy_indices) > 0:
        ax.plot(valid_healthy_indices, losses_healthy[valid_healthy_indices], label='Error Sano',
                color=palette_scenario[0], alpha=0.7, linewidth=0.5)
        ax.set_yscale('log')
        ax.set_ylabel('MSE (Log Scale)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':')
    else:
        local_logger_plot.warning("No hay errores positivos 'sanos' para graficar en escala log.")
        ax.plot([], [], label='Error Sano')

    ax.set_title('Datos Sanos (Baseline)')
    ax.set_xlabel('Índice de Ventana (Tiempo)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "error_timeseries_sano.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de serie temporal (Sano) guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico de serie temporal (Sano) {filename}: {e}")
    plt.close(fig)


def plot_error_timeseries_dano(losses_damage, output_dir):
    """Genera un gráfico del error de reconstrucción (MSE) a lo largo del tiempo (ventanas) para datos CON DAÑO."""
    if losses_damage is None or losses_damage.size == 0:
        local_logger_plot.warning("No hay datos de error 'daño' para graficar time series.")
        return
    local_logger_plot.info("Generando gráfico de serie temporal de error (Daño)...")
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig.suptitle('Error de Reconstrucción (MSE) en el Tiempo (Ventanas) - Datos con Daño', fontsize=16)

    valid_damage_indices = np.where(losses_damage > 0)[0]
    if len(valid_damage_indices) > 0:
        ax.plot(valid_damage_indices, losses_damage[valid_damage_indices], label='Error Daño',
                color=palette_scenario[1], alpha=0.7, linewidth=0.5)
        ax.set_yscale('log')
        ax.set_ylabel('MSE (Log Scale)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':')
    else:
        local_logger_plot.warning("No hay errores positivos 'daño' para graficar en escala log.")
        ax.plot([], [], label='Error Daño')

    ax.set_title('Datos con Daño')
    ax.set_xlabel('Índice de Ventana (Tiempo)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "error_timeseries_dano.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de serie temporal (Daño) guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico de serie temporal (Daño) {filename}: {e}")
    plt.close(fig)


def plot_error_heatmap(losses_healthy_per_sensor, losses_damage_per_sensor, output_dir):
    """Genera mapas de calor que muestran el error por sensor a lo largo del tiempo."""
    local_logger_plot.info("Generando mapas de calor de error...")
    data_list = []
    has_healthy = losses_healthy_per_sensor is not None and losses_healthy_per_sensor.size > 0
    has_damage = losses_damage_per_sensor is not None and losses_damage_per_sensor.size > 0
    if has_healthy:
        valid_healthy = losses_healthy_per_sensor[np.isfinite(losses_healthy_per_sensor)]
        if len(valid_healthy[valid_healthy > 0]) > 0:
            data_list.append(valid_healthy[valid_healthy > 0])
    if has_damage:
        valid_damage = losses_damage_per_sensor[np.isfinite(losses_damage_per_sensor)]
        if len(valid_damage[valid_damage > 0]) > 0:
            data_list.append(valid_damage[valid_damage > 0])
    if not data_list:
        local_logger_plot.warning("No hay datos finitos positivos disponibles para graficar heatmap.")
        return
    all_data_positive = np.concatenate(data_list)
    if all_data_positive.size == 0:
        local_logger_plot.warning("No hay datos finitos positivos disponibles para graficar heatmap.")
        return

    vmin = np.percentile(all_data_positive, 0.5)
    vmax = np.percentile(all_data_positive, 99.5)
    vmin = max(vmin, 1e-12)  # Asegurar que vmin sea positivo para LogNorm

    if vmin >= vmax:
        vmin_plot = vmin * 0.9 if vmin > 0 else -1e-9
        vmax_plot = vmax * 1.1 if vmax > 0 else 1e-9
        if vmin_plot == 0 and vmax_plot == 0: vmin_plot = -1; vmax_plot = 1
        norm = plt.Normalize(vmin=vmin_plot, vmax=vmax_plot)
        cbar_label = 'MSE (Escala Lineal)'
        local_logger_plot.warning(f"Heatmap vmin ~= vmax ({vmin:.2e}). Usando escala lineal cerca del valor.")
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cbar_label = 'MSE (Escala Log)'

    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Heatmap de Error (Sensor vs. Tiempo)', fontsize=16)
    num_sensors = losses_healthy_per_sensor.shape[1] if has_healthy else (
        losses_damage_per_sensor.shape[1] if has_damage else 0)
    if num_sensors == 0:
        local_logger_plot.error("No se pudo determinar el número de sensores para el heatmap.")
        return

    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]
    healthy_cmap = "Blues";
    damage_cmap = "Reds"

    if has_healthy:
        # Clipear datos por si acaso (aunque vmin ya es > 0 si se usa LogNorm)
        data_healthy_clipped = np.maximum(losses_healthy_per_sensor.T, vmin) if isinstance(norm,
                                                                                           LogNorm) else losses_healthy_per_sensor.T
        sns.heatmap(data_healthy_clipped, ax=axes[0], cmap=healthy_cmap, norm=norm,
                    cbar_kws={'label': cbar_label}, xticklabels=10000)
        axes[0].set_title('Datos Sanos (Baseline)')
        axes[0].set_yticklabels(sensor_labels, rotation=0)
        axes[0].set_ylabel('Sensor')
    else:
        axes[0].set_title('Datos Sanos (No disponibles)')

    if has_damage:
        data_damage_clipped = np.maximum(losses_damage_per_sensor.T, vmin) if isinstance(norm,
                                                                                         LogNorm) else losses_damage_per_sensor.T
        sns.heatmap(data_damage_clipped, ax=axes[1], cmap=damage_cmap, norm=norm,
                    cbar_kws={'label': cbar_label}, xticklabels=10000)
        axes[1].set_title('Datos con Daño')
        axes[1].set_yticklabels(sensor_labels, rotation=0)
        axes[1].set_ylabel('Sensor')
        axes[1].set_xlabel('Índice de Ventana (Tiempo Aprox.)')
    else:
        axes[1].set_title('Datos con Daño (No disponibles)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "error_heatmap_sensor_vs_time.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Heatmap de error guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando heatmap {filename}: {e}")
    plt.close()


def plot_error_statistics_per_sensor(losses_healthy_per_sensor, losses_damage_per_sensor, output_dir):
    """Genera gráficos de violín+boxplot comparativos."""
    if losses_healthy_per_sensor is None or losses_damage_per_sensor is None or \
            losses_healthy_per_sensor.size == 0 or losses_damage_per_sensor.size == 0 or \
            losses_healthy_per_sensor.shape[1] != losses_damage_per_sensor.shape[1]:
        local_logger_plot.warning("Datos de pérdida por sensor insuficientes para gráfico estadístico.")
        return
    local_logger_plot.info("Generando gráfico estadístico de error por sensor (Violinplots + Boxplots)...")
    num_sensors = losses_healthy_per_sensor.shape[1]
    sensor_labels = [f'Sensor {i + 1}' for i in range(num_sensors)]
    data_list_healthy = [];
    data_list_damage = []
    for i in range(num_sensors):
        sensor_name = sensor_labels[i]
        healthy_positive = losses_healthy_per_sensor[:, i][losses_healthy_per_sensor[:, i] > 0]
        if len(healthy_positive) > 0:
            for loss_val in healthy_positive: data_list_healthy.append({'Sensor': sensor_name, 'MSE': loss_val})
        else:
            data_list_healthy.append({'Sensor': sensor_name, 'MSE': 1e-12})
        damage_positive = losses_damage_per_sensor[:, i][losses_damage_per_sensor[:, i] > 0]
        if len(damage_positive) > 0:
            for loss_val in damage_positive: data_list_damage.append({'Sensor': sensor_name, 'MSE': loss_val})
        else:
            data_list_damage.append({'Sensor': sensor_name, 'MSE': 1e-12})
    df_healthy = pd.DataFrame(data_list_healthy)
    df_damage = pd.DataFrame(data_list_damage)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    fig.suptitle('Distribución Estadística de Error de Reconstrucción por Sensor', fontsize=16)
    if not df_healthy.empty and df_healthy['MSE'].max() > 1e-11:
        sns.violinplot(x='Sensor', y='MSE', data=df_healthy, ax=axes[0],
                       palette=sensor_colors_mapped, inner=None,
                       cut=0, scale='width', linewidth=1.5, alpha=0.7)
        sns.boxplot(x='Sensor', y='MSE', data=df_healthy, ax=axes[0],
                    showcaps=False, boxprops={'facecolor': 'None', "zorder": 10},
                    showfliers=False, whiskerprops={'linewidth': 2, "zorder": 10, 'color': 'black'},
                    medianprops={'linewidth': 2, "zorder": 10, 'color': 'black'}, width=0.3)
        axes[0].set_title('Datos Sanos (Baseline - Violin + Box)')
        axes[0].set_ylabel('Mean Squared Error (MSE) - Log Scale')
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle=':')
        valid_mse_h = df_healthy['MSE'][np.isfinite(df_healthy['MSE'])]
        if len(valid_mse_h) > 0:
            p_low_h = np.percentile(valid_mse_h, 0.5);
            p_high_h = np.percentile(valid_mse_h, 99.5)
            min_mse_plot_h = max(p_low_h * 0.9, 1e-12);
            max_mse_plot_h = p_high_h * 1.1
            if min_mse_plot_h >= max_mse_plot_h:
                min_mse_plot_h = max(valid_mse_h.min() * 0.8, 1e-12);
                max_mse_plot_h = valid_mse_h.max() * 1.2
            axes[0].set_ylim(bottom=min_mse_plot_h, top=max_mse_plot_h)
    else:
        axes[0].set_title('Datos Sanos (Sin datos positivos o muy pequeños)')
    if not df_damage.empty and df_damage['MSE'].max() > 1e-11:
        sns.violinplot(x='Sensor', y='MSE', data=df_damage, ax=axes[1],
                       palette=sensor_colors_mapped, inner=None, cut=0, scale='width', linewidth=1.5, alpha=0.7)
        sns.boxplot(x='Sensor', y='MSE', data=df_damage, ax=axes[1],
                    showcaps=False, boxprops={'facecolor': 'None', "zorder": 10},
                    showfliers=False, whiskerprops={'linewidth': 2, "zorder": 10, 'color': 'black'},
                    medianprops={'linewidth': 2, "zorder": 10, 'color': 'black'}, width=0.3)
        axes[1].set_title('Datos con Daño (Violin + Box)')
        axes[1].set_ylabel('Mean Squared Error (MSE) - Log Scale')
        axes[1].set_yscale('log')
        axes[1].grid(True, linestyle=':')
        valid_mse_d = df_damage['MSE'][np.isfinite(df_damage['MSE'])]
        if len(valid_mse_d) > 0:
            p_low_d = np.percentile(valid_mse_d, 0.5);
            p_high_d = np.percentile(valid_mse_d, 99.5)
            min_mse_plot_d = max(p_low_d * 0.9, 1e-12);
            max_mse_plot_d = p_high_d * 1.1
            if min_mse_plot_d >= max_mse_plot_d:
                min_mse_plot_d = max(valid_mse_d.min() * 0.8, 1e-12);
                max_mse_plot_d = valid_mse_d.max() * 1.2
            axes[1].set_ylim(bottom=min_mse_plot_d, top=max_mse_plot_d)
    else:
        axes[1].set_title('Datos con Daño (Sin datos positivos o muy pequeños)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, "error_statistics_per_sensor_violinbox.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico estadístico violin+boxplot guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico estadístico violin+boxplot: {e}")
    plt.close(fig)


def plot_evaluation_scatters_and_boxes(healthy_losses_global, healthy_ssim_global, damage_losses_global,
                                       damage_ssim_global, output_dir):
    """Replica la estructura de la Figura 9 del paper (Violin+Box)."""
    local_logger_plot.info("Generando gráficos de evaluación global y distribución (estilo Fig. 9)...")
    if healthy_losses_global is None or healthy_ssim_global is None or \
            damage_losses_global is None or damage_ssim_global is None or \
            healthy_losses_global.size == 0 or damage_losses_global.size == 0:
        local_logger_plot.warning("Faltan datos globales MSE o SSIM (Sano o Daño), saltando gráfico Fig. 9.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Resultados de Evaluación General (Sano vs. Daño)', fontsize=16)
    num_healthy = len(healthy_losses_global);
    num_damage = len(damage_losses_global)
    indices_healthy = np.arange(num_healthy)
    indices_damage = np.arange(num_healthy, num_healthy + num_damage)
    axes[0, 0].scatter(indices_healthy, healthy_losses_global, label='Sano', alpha=0.5, s=10,
                       color=palette_scenario[0])
    axes[0, 0].scatter(indices_damage, damage_losses_global, label='Daño', alpha=0.5, s=10, color=palette_scenario[1])
    axes[0, 0].set_title('(a) Error de Reconstrucción (MSE) vs. Muestras')
    axes[0, 0].set_xlabel('El número de muestras')
    axes[0, 0].set_ylabel('Error (MSE)')
    axes[0, 0].legend();
    axes[0, 0].grid(True, linestyle=':')
    valid_losses = np.concatenate([healthy_losses_global, damage_losses_global])
    valid_losses = valid_losses[np.isfinite(valid_losses)]
    if len(valid_losses) > 0:
        positive_losses = valid_losses[valid_losses > 0]
        if positive_losses.size > 0:
            min_error_all = max(np.min(positive_losses), 1e-12);
            max_error_all = np.max(valid_losses)
            if max_error_all / min_error_all > 1000:
                axes[0, 0].set_yscale('log');
                axes[0, 0].set_ylabel('Error (MSE) - Log Scale')
                p_low = np.percentile(positive_losses, 0.5);
                p_high = np.percentile(valid_losses, 99.5)
                min_mse_plot = max(p_low * 0.9, 1e-12);
                max_mse_plot = p_high * 1.1
                if min_mse_plot >= max_mse_plot:
                    min_mse_plot = max(min_error_all * 0.8, 1e-12);
                    max_mse_plot = max_error_all * 1.2
                axes[0, 0].set_ylim(bottom=min_mse_plot, top=max_mse_plot)
    axes[0, 1].scatter(indices_healthy, healthy_ssim_global, label='Sano', alpha=0.5, s=10,
                       color=palette_scenario[0])
    axes[0, 1].scatter(indices_damage, damage_ssim_global, label='Daño', alpha=0.5, s=10, color=palette_scenario[1])
    axes[0, 1].set_title('(b) Similaridad Estructural (SSIM) vs. Muestras')
    axes[0, 1].set_xlabel('El número de muestras');
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].legend();
    axes[0, 1].grid(True, linestyle=':');
    axes[0, 1].set_ylim(-0.05, 1.05)
    healthy_df = pd.DataFrame({'MSE': healthy_losses_global, 'SSIM': healthy_ssim_global, 'Escenario': 'Sano'})
    damage_df = pd.DataFrame({'MSE': damage_losses_global, 'SSIM': damage_ssim_global, 'Escenario': 'Daño'})
    full_df = pd.concat([healthy_df, damage_df], ignore_index=True)
    full_df_pos_mse = full_df[full_df['MSE'] > 0].copy()
    if full_df_pos_mse.empty or full_df_pos_mse['MSE'].max() <= 1e-11:
        axes[1, 0].set_title('(c) Distribución de Error (Sin datos positivos o muy pequeños)')
    else:
        sns.violinplot(x='Escenario', y='MSE', data=full_df_pos_mse, ax=axes[1, 0],
                       palette=palette_scenario, inner=None,
                       cut=0, scale='width', linewidth=1.5, alpha=0.7)
        sns.boxplot(x='Escenario', y='MSE', data=full_df_pos_mse, ax=axes[1, 0],
                    showcaps=False, boxprops={'facecolor': 'None', "zorder": 10},
                    showfliers=False, whiskerprops={'linewidth': 2, "zorder": 10, 'color': 'black'},
                    medianprops={'linewidth': 2, "zorder": 10, 'color': 'black'}, width=0.3)
        axes[1, 0].set_title('(c) Distribución de Error por Escenario (Violin + Box)')
        axes[1, 0].set_xlabel('Escenarios');
        axes[1, 0].set_ylabel('Error (MSE) - Log Scale')
        axes[1, 0].set_yscale('log');
        axes[1, 0].grid(True, linestyle=':')
        valid_mse_c = full_df_pos_mse['MSE'][np.isfinite(full_df_pos_mse['MSE'])]
        if len(valid_mse_c) > 0:
            p_low_c = np.percentile(valid_mse_c, 0.5);
            p_high_c = np.percentile(valid_mse_c, 99.5)
            min_mse_plot_c = max(p_low_c * 0.9, 1e-12);
            max_mse_plot_c = p_high_c * 1.1
            if min_mse_plot_c >= max_mse_plot_c:
                min_mse_plot_c = max(valid_mse_c.min() * 0.8, 1e-12);
                max_mse_plot_c = valid_mse_c.max() * 1.2
            axes[1, 0].set_ylim(bottom=min_mse_plot_c, top=max_mse_plot_c)
    sns.violinplot(x='Escenario', y='SSIM', data=full_df, ax=axes[1, 1],
                   palette=palette_scenario, inner=None,
                   cut=0, scale='width', linewidth=1.5, alpha=0.7)
    sns.boxplot(x='Escenario', y='SSIM', data=full_df, ax=axes[1, 1],
                showcaps=False, boxprops={'facecolor': 'None', "zorder": 10},
                showfliers=False, whiskerprops={'linewidth': 2, "zorder": 10, 'color': 'black'},
                medianprops={'linewidth': 2, "zorder": 10, 'color': 'black'}, width=0.3)
    axes[1, 1].set_title('(d) Distribución de SSIM por Escenario (Violin + Box)')
    axes[1, 1].set_xlabel('Escenarios');
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].grid(True, linestyle=':');
    axes[1, 1].set_ylim(bottom=-0.05, top=1.05)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, "evaluation_scatters_and_distributions_fig9_violinbox.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico estilo Fig. 9 (Violin+Box) guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico estilo Fig. 9 {filename}: {e}")
    plt.close(fig)


def plot_ssim_vs_error_cluster(healthy_losses_per_sensor, healthy_ssim_per_sensor, damage_losses_per_sensor,
                               damage_ssim_per_sensor, num_nodes, output_dir):
    """Replica la estructura de la Figura 10 y 11 del paper (un gráfico por sensor)."""
    local_logger_plot.info("Generando gráficos de cluster SSIM vs. Error (estilo Fig. 10)...")
    if healthy_losses_per_sensor is None or healthy_ssim_per_sensor is None or \
            damage_losses_per_sensor is None or damage_ssim_per_sensor is None or \
            healthy_losses_per_sensor.size == 0 or damage_losses_per_sensor.size == 0:
        local_logger_plot.warning("Faltan datos por sensor MSE o SSIM (Sano o Daño), saltando gráficos Fig. 10.")
        return
    for i in range(num_nodes):
        sensor_id = i + 1
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f'Resultados Progresivos de Detección de Daño del Sensor {sensor_id}', fontsize=16)
        h_loss = healthy_losses_per_sensor[:, i];
        h_ssim = healthy_ssim_per_sensor[:, i]
        d_loss = damage_losses_per_sensor[:, i];
        d_ssim = damage_ssim_per_sensor[:, i]
        axes[0].scatter(h_loss, h_ssim, label='Escenario: Sano', color=palette_scenario[0], alpha=0.3, s=15)
        axes[0].scatter(d_loss, d_ssim, label='Escenario: Daño', color=palette_scenario[1], alpha=0.3, s=15)
        axes[0].set_title('(a) Análisis de Cluster')
        axes[0].set_xlabel('Error (MSE)');
        axes[0].set_ylabel('SSIM')
        axes[0].legend();
        axes[0].grid(True, linestyle=':');
        all_losses_sensor = np.concatenate([h_loss, d_loss])
        all_losses_sensor = all_losses_sensor[np.isfinite(all_losses_sensor)]
        if len(all_losses_sensor) > 0:
            positive_losses_s = all_losses_sensor[all_losses_sensor > 0]
            if positive_losses_s.size > 0:
                min_loss_s = max(np.min(positive_losses_s), 1e-12);
                max_loss_s = np.max(all_losses_sensor)
                if max_loss_s / min_loss_s > 100:
                    axes[0].set_xscale('log');
                    axes[0].set_xlabel('Error (MSE) - Log Scale')
                    p_low_x = np.percentile(positive_losses_s, 0.5);
                    p_high_x = np.percentile(all_losses_sensor, 99.5)
                    min_mse_plot_x = max(p_low_x * 0.9, 1e-12);
                    max_mse_plot_x = p_high_x * 1.1
                    if min_mse_plot_x >= max_mse_plot_x:
                        min_mse_plot_x = max(min_loss_s * 0.8, 1e-12);
                        max_mse_plot_x = max_loss_s * 1.2
                    axes[0].set_xlim(left=min_mse_plot_x, right=max_mse_plot_x)
        axes[0].set_ylim(-0.05, 1.05)
        h_loss_mean = np.mean(h_loss);
        h_ssim_mean = np.mean(h_ssim)
        d_loss_mean = np.mean(d_loss);
        d_ssim_mean = np.mean(d_ssim)
        axes[1].plot([h_loss_mean, d_loss_mean], [h_ssim_mean, d_ssim_mean], marker='o', markersize=10, linestyle='-',
                     linewidth=2, color='darkgrey')
        axes[1].scatter(h_loss_mean, h_ssim_mean, label=f'Promedio Sano (E: {h_loss_mean:.2e}, S: {h_ssim_mean:.2f})',
                        color=palette_scenario[0], s=120, zorder=5, edgecolor='black')
        axes[1].scatter(d_loss_mean, d_ssim_mean, label=f'Promedio Daño (E: {d_loss_mean:.2e}, S: {d_ssim_mean:.2f})',
                        color=palette_scenario[1], s=120, zorder=5, edgecolor='black')
        axes[1].set_title('(b) Tendencia de Valores Promedio')
        axes[1].set_xlabel('Error (MSE)');
        axes[1].set_ylabel('SSIM')
        axes[1].legend();
        axes[1].grid(True, linestyle=':');
        axes[1].set_ylim(-0.05, 1.05)
        if axes[0].get_xscale() == 'log':
            axes[1].set_xscale('log');
            axes[1].set_xlim(axes[0].get_xlim());
            axes[1].set_xlabel('Error (MSE) - Log Scale')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = os.path.join(output_dir, f"ssim_vs_error_cluster_sensor_{sensor_id}.png")
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            local_logger_plot.info(f"Gráfico estilo Fig. 10 para sensor {sensor_id} guardado en: {filename}")
        except Exception as e:
            local_logger_plot.error(f"Error guardando gráfico estilo Fig. 10 {filename}: {e}")
        plt.close(fig)


# --- FUNCIÓN CORREGIDA ---
def plot_training_history(log_file_path, output_dir):
    """
    Parsea el archivo de log y genera un gráfico de las curvas de pérdida
    con anotaciones en puntos clave. (CORREGIDA)
    """
    local_logger_plot.info("Generando gráfico de historial de entrenamiento (estilo Fig. 7) con anotaciones...")
    epochs = [];
    train_losses = [];
    val_losses = []
    epoch_pattern = re.compile(r"Epoch (\d+)/\d+ -> .*Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+)")
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = epoch_pattern.search(line)
                if match:
                    try:
                        epochs.append(int(match.group(1)))
                        train_losses.append(float(match.group(2)))
                        val_losses.append(float(match.group(3)))
                    except ValueError:
                        local_logger_plot.warning(
                            f"No se pudieron parsear los valores de pérdida en la línea: {line.strip()}. Saltando línea.")
                        continue
    except FileNotFoundError:
        local_logger_plot.error(f"Archivo de log de entrenamiento no encontrado en: {log_file_path}. Saltando gráfico.")
        return
    except Exception as e:
        local_logger_plot.error(
            f"Error parseando archivo de log de entrenamiento {log_file_path}: {e}. Saltando gráfico.")
        return
    if not epochs:
        local_logger_plot.warning("No se encontraron datos de épocas en el log. Saltando gráfico.")
        return

    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # Encontrar índice de la mejor (mínima) pérdida de validación
    best_val_loss_idx = np.argmin(val_losses)
    best_epoch = epochs[best_val_loss_idx]
    best_val_loss = val_losses[best_val_loss_idx]

    plt.figure(figsize=(12, 7))  # Ligeramente más grande para anotaciones
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-', color='#e41a1c', markersize=4,
             alpha=0.7)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--', color='#377eb8', markersize=5,
             alpha=0.7)

    plt.title('Pérdida de Entrenamiento y Validación por Época (Historial Completo)')
    plt.xlabel('Época');
    plt.ylabel('Pérdida (MSE)')
    plt.legend();
    plt.grid(True, linestyle=':')

    # --- LÓGICA DE ESCALA LOG CORREGIDA ---
    all_losses = np.concatenate([train_losses, val_losses])
    all_losses = all_losses[np.isfinite(all_losses)]
    if not all_losses.size:
        min_loss = 1e-6;
        max_loss = 1.0
    else:
        positive_losses = all_losses[all_losses > 0]  # <-- CORRECCIÓN
        min_loss = np.min(positive_losses) if positive_losses.size > 0 else 1e-9  # <-- CORRECCIÓN
        max_loss = np.max(all_losses)

    use_log_scale = (max_loss / max(min_loss, 1e-9) > 100) or min_loss < 0.01
    # --- FIN CORRECCIÓN ---

    if use_log_scale:
        plt.yscale('log');
        plt.ylabel('Pérdida (MSE) - Log Scale')
        plot_min_y = max(min_loss * 0.8, 1e-12)  # Ajustar para log
        plot_max_y = max_loss * 1.2
        plt.ylim(bottom=plot_min_y, top=plot_max_y)
    else:
        plt.ylim(bottom=0, top=max_loss * 1.1)
        plot_min_y = 0  # Para cálculo de posición de anotaciones

    # --- INICIO ANOTACIONES ---
    points_to_annotate = []
    # 1. Primer punto
    if len(epochs) > 0:
        points_to_annotate.append({'epoch': epochs[0], 'train': train_losses[0], 'val': val_losses[0]})
    # 2. Último punto
    if len(epochs) > 0:
        points_to_annotate.append({'epoch': epochs[-1], 'train': train_losses[-1], 'val': val_losses[-1]})
    # 3. Mejor punto de validación (si no es el primero o último)
    if best_epoch != epochs[0] and best_epoch != epochs[-1]:
        points_to_annotate.append({'epoch': best_epoch, 'train': train_losses[best_val_loss_idx], 'val': best_val_loss})

    for point in points_to_annotate:
        ep = point['epoch']
        tr_loss = point['train']
        val_loss = point['val']

        # Anotar Train Loss
        plt.annotate(f'{tr_loss:.4f}', (ep, tr_loss), textcoords="offset points", xytext=(0, 10), ha='center',
                     fontsize=8, color='#e41a1c')
        # Anotar Val Loss
        plt.annotate(f'{val_loss:.4f}', (ep, val_loss), textcoords="offset points", xytext=(0, -15), ha='center',
                     fontsize=8, color='#377eb8')
        # Marcar punto de mejor validación
        if ep == best_epoch:
            plt.scatter(ep, val_loss, s=60, facecolors='none', edgecolors='green', linewidth=1.5,
                        label=f'Mejor Val Loss ({best_val_loss:.4f} en Época {best_epoch})')
            # Actualizar leyenda para incluir el punto verde
            handles, labels = plt.gca().get_legend_handles_labels()
            # Evitar duplicados si ya existe
            if not any("Mejor Val Loss" in lab for lab in labels):
                plt.legend(handles=handles, labels=labels)
            else:
                # Si ya está (caso donde mejor es el último), solo actualizar
                new_labels = [lab.split(' (')[0] if "Mejor Val Loss" in lab else lab for lab in labels]
                plt.legend(handles=handles, labels=new_labels)

    # --- FIN ANOTACIONES ---

    filename = os.path.join(output_dir, "training_history_loss_curves_annotated.png")  # Nuevo nombre
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de historial de entrenamiento (anotado) guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico de historial (anotado) {filename}: {e}")
    plt.close()


def plot_bridge_graph_structure(edge_index_numpy, output_dir):
    """
    Dibuja solo la estructura del grafo del puente.
    """
    local_logger_plot.info("Generando gráfico de estructura del puente...")
    G = nx.Graph()
    G.add_nodes_from(range(5))
    edges = set()
    for i in range(edge_index_numpy.shape[1]):
        u, v = sorted(edge_index_numpy[:, i])
        edges.add((u, v))
    G.add_edges_from(list(edges))
    pos = {0: (0, 0), 1: (1, 0.1), 2: (1, -0.1), 3: (2, 0.1), 4: (2, -0.1)}
    labels = {i: str(i + 1) for i in G.nodes()}
    fig, ax = plt.subplots(figsize=(8, 4))
    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels, node_color='skyblue',
            node_size=800, font_size=12, font_color='black', font_weight='bold', edge_color='gray', width=1.5)
    ax.set_title("Estructura del Grafo de Sensores del Puente")
    filename = os.path.join(output_dir, "bridge_graph_structure.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de estructura del puente guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico de estructura del puente {filename}: {e}")
    plt.close(fig)


def plot_graph_heatmap_comparison(edge_index_numpy, healthy_values, damage_values, output_dir):
    """Dibuja dos grafos lado a lado, coloreando los nodos según los valores (MSE)."""
    local_logger_plot.info("Generando gráfico de comparación de heatmap en grafo...")
    if healthy_values is None or damage_values is None or len(healthy_values) != 5 or len(damage_values) != 5:
        local_logger_plot.warning("Valores promedio por sensor inválidos o faltantes para heatmap de grafo. Saltando.")
        return
    healthy_values = np.nan_to_num(healthy_values, nan=0.0, posinf=np.finfo(np.float32).max,
                                   neginf=np.finfo(np.float32).min)
    damage_values = np.nan_to_num(damage_values, nan=0.0, posinf=np.finfo(np.float32).max,
                                  neginf=np.finfo(np.float32).min)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('MSE Promedio en Nodos del Grafo', fontsize=16)
    cmap_healthy = plt.cm.Blues;
    cmap_damage = plt.cm.Reds
    all_values = np.concatenate([healthy_values, damage_values])
    vmin = np.min(all_values);
    vmax = np.max(all_values)
    if vmin == vmax:
        vmin -= 0.1 * abs(vmin) + 1e-6 if vmin != 0 else 1e-6
        vmax += 0.1 * abs(vmax) + 1e-6 if vmax != 0 else 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper_healthy = cm.ScalarMappable(norm=norm, cmap=cmap_healthy)
    node_colors_healthy = [mapper_healthy.to_rgba(val) for val in healthy_values]
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
    axes[0].set_title("Datos Sanos")
    cbar_h = fig.colorbar(mapper_healthy, ax=axes[0], shrink=0.8, aspect=15)
    cbar_h.set_label('MSE Promedio')
    mapper_damage = cm.ScalarMappable(norm=norm, cmap=cmap_damage)
    node_colors_damage = [mapper_damage.to_rgba(val) for val in damage_values]
    nx.draw(G, pos, ax=axes[1], with_labels=True, labels=labels, node_color=node_colors_damage,
            node_size=900, font_size=12, font_color='black', font_weight='bold', edge_color='darkgray', width=2)
    axes[1].set_title("Datos con Daño")
    cbar_d = fig.colorbar(mapper_damage, ax=axes[1], shrink=0.8, aspect=15)
    cbar_d.set_label('MSE Promedio')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(output_dir, "graph_heatmap_comparison.png")
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        local_logger_plot.info(f"Gráfico de comparación de heatmap en grafo guardado en: {filename}")
    except Exception as e:
        local_logger_plot.error(f"Error guardando gráfico de heatmap en grafo {filename}: {e}")
    plt.close(fig)


# --- FUNCIÓN PRINCIPAL DE INFERENCIA (MODIFICADA) ---
def run_inference_and_plot(model_dir, base_healthy_dir, damage_data_dir=None):
    """
    Función principal que orquesta la carga del modelo, la inferencia y el ploteo de resultados.
    """
    file_handler = None
    try:
        # --- Configuración de directorios y logging de archivo ---
        inference_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(model_dir, f"inference_results_wavelet_{inference_timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        log_file_path_inference = os.path.join(results_dir, 'inference_wavelet.log')
        file_handler = logging.FileHandler(log_file_path_inference, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        logger.info(f"--- Iniciando Inferencia Wavelet-GNN usando modelo de: {model_dir} ---")
        logger.info(f"Directorio de datos sanos: {base_healthy_dir}")
        logger.info(f"Directorio de datos con daño: {damage_data_dir}")
        logger.info(f"Resultados se guardarán en: {results_dir}")

        # --- Carga de hiperparámetros y escalador ---
        hp_path = os.path.join(model_dir, 'hyperparameters_wavelet_gnn.json')
        scaler_path = os.path.join(model_dir, 'scaler_wavelet_gnn.gz')
        model_path = os.path.join(model_dir, 'best_model_wavelet_gnn.pth')
        # El log de entrenamiento se busca en dos posibles ubicaciones
        training_log_path = os.path.join(model_dir, 'training_log_wavelet_RESUME.txt')
        if not os.path.exists(training_log_path):
            training_log_path = os.path.join(model_dir, 'training_log_wavelet.txt')  # Fallback al original

        for f_path in [hp_path, scaler_path, model_path]:
            if not os.path.exists(f_path):
                raise FileNotFoundError(f"Archivo requerido no encontrado: {f_path}")
        if not os.path.exists(training_log_path):
            logger.warning(
                f"Archivo de log de entrenamiento no encontrado en: {training_log_path} (o fallback). Se omitirá el gráfico de historial de entrenamiento.")
            training_log_path = None

        with open(hp_path, 'r', encoding='utf-8') as f:
            hp = json.load(f)
        scaler = joblib.load(scaler_path)
        logger.info("Hiperparámetros y scaler cargados.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")

        edge_index = define_bridge_graph().to(device)
        num_nodes = 5

        # --- Inicialización y carga del modelo (con HPs cargados) ---
        model = SpatioTemporalAutoencoder(
            num_nodes=num_nodes,
            num_features=hp['num_features'],
            window_size=hp['window_size'],
            gnn_hidden=hp.get('gnn_hidden'),
            gnn_out=hp.get('gnn_out'),
            rnn_hidden=hp.get('rnn_hidden'),
            rnn_layers=hp.get('rnn_layers')
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Modelo Wavelet-GNN cargado y en modo evaluación.")

        # --- Carga y procesamiento de datos SANOS ---
        logger.info("Cargando y procesando datos sanos (baseline)...")
        # MODIFICADO: Obtener también los datos sin escalar
        healthy_data_scaled_dict, healthy_data_features_unscaled, target_len_healthy = load_and_preprocess_wavelet_data(
            base_healthy_dir, hp, scaler, num_nodes, target_len=None
        )
        if healthy_data_scaled_dict is None:
            raise ValueError(f"Error fatal: No se pudieron cargar/preprocesar los datos sanos de {base_healthy_dir}.")

        healthy_dataset = SpatioTemporalWaveletDataset(
            healthy_data_scaled_dict,
            window_size=hp['window_size'],
            stride=hp['stride'],
            num_expected_nodes=num_nodes
        )
        if len(healthy_dataset) == 0:
            raise ValueError("El dataset 'sano' no tiene suficientes datos para crear ni una ventana.")
        healthy_loader = DataLoader(healthy_dataset, batch_size=hp['batch_size'], shuffle=False)

        # --- Inferencia en datos SANOS ---
        logger.info("Realizando inferencia en datos sanos...")
        healthy_originals, healthy_reconstructions, healthy_losses, healthy_losses_per_sensor, healthy_ssim_per_sensor = perform_inference(
            model, healthy_loader, device, edge_index, hp
        )
        if healthy_losses_per_sensor.size == 0:
            raise ValueError("La inferencia en datos sanos falló, no se obtuvieron resultados.")

        healthy_mse_mean_per_sensor = np.mean(healthy_losses_per_sensor, axis=0)
        healthy_losses_global = np.mean(healthy_losses_per_sensor, axis=1)  # Promedio de sensores para cada ventana
        healthy_ssim_global = np.mean(healthy_ssim_per_sensor, axis=1)  # Promedio de sensores para cada ventana

        # --- Carga e inferencia en datos con DAÑO (si se proporcionan) ---
        damage_originals = None;
        damage_reconstructions = None;
        damage_losses = None
        damage_losses_per_sensor = None;
        damage_mse_mean_per_sensor = None
        damage_ssim_per_sensor = None;
        damage_losses_global = None;
        damage_ssim_global = None

        if damage_data_dir:
            logger.info("Cargando y procesando datos con daño...")
            # MODIFICADO: Usar '_' para los datos sin escalar (no los necesitamos para 'daño')
            damage_data_scaled_dict, _, _ = load_and_preprocess_wavelet_data(
                damage_data_dir, hp, scaler, num_nodes, target_len=target_len_healthy
            )

            if damage_data_scaled_dict is not None:
                damage_dataset = SpatioTemporalWaveletDataset(
                    damage_data_scaled_dict,
                    window_size=hp['window_size'],
                    stride=hp['stride'],
                    num_expected_nodes=num_nodes
                )
                if len(damage_dataset) > 0:
                    damage_loader = DataLoader(damage_dataset, batch_size=hp['batch_size'], shuffle=False)
                    logger.info("Realizando inferencia en datos con daño...")
                    damage_originals, damage_reconstructions, damage_losses, damage_losses_per_sensor, damage_ssim_per_sensor = perform_inference(
                        model, damage_loader, device, edge_index, hp
                    )
                    if damage_losses_per_sensor.size > 0:
                        damage_mse_mean_per_sensor = np.mean(damage_losses_per_sensor, axis=0)
                        damage_losses_global = np.mean(damage_losses_per_sensor, axis=1)
                        damage_ssim_global = np.mean(damage_ssim_per_sensor, axis=1)
                    else:
                        logger.warning("La inferencia en datos con daño no produjo resultados.")
                else:
                    logger.warning("El dataset 'daño' no tiene suficientes datos. Se omitirá el análisis de daño.")
            else:
                logger.warning("No se pudieron cargar/preprocesar los datos con daño. Se omitirá el análisis de daño.")
        else:
            logger.info("No se proporcionó directorio de datos con daño. Se omitirá el análisis de daño.")

        # --- PLOTEO (ADAPTADO) ---
        logger.info("Generando gráficos de resultados...")

        # --- Gráfico de estructura del grafo (UNA SOLA VEZ) ---
        plot_bridge_graph_structure(edge_index.cpu().numpy(), results_dir)

        # --- Gráfico de historial de entrenamiento (del log) ---
        if training_log_path:
            plot_training_history(training_log_path, results_dir)

        # --- (NUEVO) Gráfico de features wavelet (de datos sanos) ---
        if healthy_data_features_unscaled:
            plot_wavelet_features_ejemplo(healthy_data_features_unscaled, results_dir, hp, num_nodes)
        else:
            logger.warning("No se generó el gráfico de features wavelet (no se recibieron datos sin escalar).")

        # --- Gráficos de reconstrucción por sensor (Wavelet-Adaptado) ---
        logger.info("Generando gráficos de reconstrucción por sensor (2 muestras)...")
        plot_sensor_reconstruction_samples(
            healthy_originals, healthy_reconstructions, scaler, num_nodes, results_dir, "sano"
        )
        if damage_originals is not None and damage_reconstructions is not None:
            plot_sensor_reconstruction_samples(
                damage_originals, damage_reconstructions, scaler, num_nodes, results_dir, "dano"
            )
        else:
            logger.info("Omitiendo gráficos de reconstrucción de daño (datos no procesados).")

        # --- (NUEVO) Gráficos de serie temporal de error (separados) ---
        plot_error_timeseries_sano(healthy_losses, results_dir)
        if damage_losses is not None:
            plot_error_timeseries_dano(damage_losses, results_dir)

        # --- Generar gráficos comparativos y estadísticos (si hay datos de daño) ---
        if damage_mse_mean_per_sensor is not None and damage_losses_per_sensor is not None and \
                damage_losses is not None and damage_ssim_per_sensor is not None and \
                damage_losses_global is not None and damage_ssim_global is not None:

            logger.info("Generando gráficos comparativos, estadísticos, de heatmap y estilo paper...")

            # --- Gráfico de Grafo Heatmap ---
            plot_graph_heatmap_comparison(
                edge_index.cpu().numpy(),
                healthy_mse_mean_per_sensor,
                damage_mse_mean_per_sensor,
                results_dir
            )

            plot_mse_comparison(
                healthy_mse_mean_per_sensor,
                damage_mse_mean_per_sensor,
                results_dir
            )
            plot_damage_localization(
                damage_losses_per_sensor,
                healthy_losses_per_sensor,
                results_dir
            )
            plot_error_distribution_kde(
                healthy_losses,
                damage_losses,
                results_dir
            )
            # plot_error_timeseries (ELIMINADO, reemplazado por los separados)

            plot_error_heatmap(
                healthy_losses_per_sensor,
                damage_losses_per_sensor,
                results_dir
            )
            plot_error_statistics_per_sensor(
                healthy_losses_per_sensor,
                damage_losses_per_sensor,
                results_dir
            )
            plot_evaluation_scatters_and_boxes(
                healthy_losses_global,
                healthy_ssim_global,
                damage_losses_global,
                damage_ssim_global,
                results_dir
            )
            plot_ssim_vs_error_cluster(
                healthy_losses_per_sensor,
                healthy_ssim_per_sensor,
                damage_losses_per_sensor,
                damage_ssim_per_sensor,
                num_nodes,
                results_dir
            )
        else:
            logger.info(
                "No se procesaron datos de daño completos. Se omitirán los gráficos comparativos, de localización, estadísticos y estilo paper.")

        logger.info(f"--- Proceso de inferencia completado. Resultados guardados en: {results_dir} ---")

    except FileNotFoundError as fnf_error:
        logger.error(f"Error de archivo no encontrado: {fnf_error}")
    except Exception as e:
        logger.error(f"Error crítico en run_inference_and_plot: {e}", exc_info=True)
    finally:
        if file_handler is not None:
            logger.info("Cerrando archivo de log.")
            file_handler.close()
            logger.removeHandler(file_handler)


# --- EJECUCIÓN ---
if __name__ == '__main__':

    # --- ¡¡¡ CONFIGURAR ESTAS RUTAS !!! ---

    # 1. Apunta esto a la carpeta de resultados que CONTIENE los archivos
    #    .pth, .gz, y .json de tu entrenamiento Wavelet FINALIZADO.
    trained_model_directory = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547"

    # 2. Apunta esto a la carpeta de datos SANOS (limpios)
    base_healthy_data_directory = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # 3. Apunta esto a la carpeta de datos CON DAÑO (puede ser None si no la tienes)
    damage_data_directory = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"

    # --- Fin de Configuración ---

    if not os.path.isdir(trained_model_directory):
        logger.error(f"Directorio del modelo no encontrado: {trained_model_directory}")
    elif not os.path.isdir(base_healthy_data_directory):
        logger.error(f"Directorio de datos sanos no encontrado: {base_healthy_data_directory}")
    elif damage_data_directory and not os.path.isdir(damage_data_directory):
        logger.warning(
            f"Directorio de datos con daño proporcionado pero no encontrado: {damage_data_directory}. Ejecutando solo con datos sanos.")
        run_inference_and_plot(
            trained_model_directory,
            base_healthy_data_directory,
            damage_data_dir=None
        )
    else:
        run_inference_and_plot(
            trained_model_directory,
            base_healthy_data_directory,
            damage_data_directory
        )
