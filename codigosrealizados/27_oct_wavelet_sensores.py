# -*- coding: utf-8 -*-
"""
evaluate_model_by_sensor.py

Carga un modelo STG-AE entrenado, su scaler y sus HPs desde una carpeta
de resultados para calcular el Error de Reconstrucción (MSE) por cada sensor
individualmente sobre el conjunto de validación.

Uso:
1. Asegúrate de que todas las dependencias (torch, torch_geometric, sklearn, etc.)
   estén instaladas.
2. Actualiza la variable `RUN_TO_ANALYZE_PATH` en el bloque `if __name__ == '__main__':`
   para que apunte a tu carpeta de resultados (la que contiene .pth, .gz y .json).
3. Actualiza la variable `DATA_FOLDER_PATH` para que apunte a tus datos crudos.
4. Ejecuta el script.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import json
import joblib
import logging
import sys
import pywt  # <--- Importar PyWavelets
import gc
from tqdm import tqdm

# Asegúrate de tener torch_geometric instalado
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("Error: torch_geometric no está instalado. Por favor, instálalo con 'pip install torch_geometric'")
    sys.exit(1)

# --- Configuración del Logging (solo consola) ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# --- DEFINICIONES DE CLASES (Deben coincidir 1:1 con el script de entrenamiento) ---

# --- ESTRUCTURA DEL GRAFO ---
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


# --- LÓGICA DE DATOS ---
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
                data_node = valid_data_dict[sid][:min_len]
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
            self.data = np.stack(processed_data_list, axis=1)
        except ValueError as e:
            shapes_str = ", ".join([str(d.shape) for d in processed_data_list])
            local_logger.error(
                f"Error al apilar datos (np.stack axis=1). Shapes individuales: [{shapes_str}]. Error: {e}")
            raise e

        self.num_nodes = self.data.shape[1]
        local_logger.info(f"Datos apilados con shape final: {self.data.shape}. Sensores usados: {actual_node_ids}")

        if self.num_nodes != self.num_expected_nodes:
            local_logger.error(
                f"Inconsistencia crítica: Nodos procesados ({self.num_nodes}) != Nodos esperados ({self.num_expected_nodes}) después del apilado.")
            raise RuntimeError("Error interno en la creación del dataset: Conteo de nodos inconsistente.")
        if self.num_features != expected_num_features and expected_num_features != -1:
            local_logger.error(
                f"Inconsistencia crítica: Features procesadas ({self.num_features}) != Features esperadas ({expected_num_features}).")
            raise RuntimeError("Error interno en la creación del dataset: Conteo de features inconsistente.")

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
                f"Índice {idx} genera rango [{start}, {end}) fuera de límites [{0}, {self.data.shape[0]}]. Stride={self.stride}, WinSize={self.window_size}, N_Samples={self.n_samples}")
            raise IndexError(f"Índice {idx} fuera de rango.")
        window = self.data[start:end]
        return torch.FloatTensor(window), torch.FloatTensor(window)


# --- ARQUITECTURA DEL GNN AUTOENCODER ---
class GNNLayer(nn.Module):
    """Bloque GCN."""

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
    """Arquitectura ST-GAE."""

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
                f"Unexpected input shape in forward: {x.shape}. Expected T={self.window_size}, N={self.num_nodes}, F={self.num_features}. Trying to proceed...")
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


# --- FUNCIONES AUXILIARES WAVELET ---
def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    """Aplica DWT multinivel y reconstruye bandas."""
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 1 or len(signal) == 0:
        logger.warning("apply_dwt_features: Señal de entrada inválida.")
        return None
    original_len = len(signal)
    if target_len is None:
        target_len = original_len
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        reconstructed_bands = []
        for i in range(level, 0, -1):
            detail_level_index = level - i + 1
            if detail_level_index < 0 or detail_level_index >= len(coeffs):
                logger.error(
                    f"Índice de nivel de detalle {detail_level_index} fuera de rango para coeffs (len={len(coeffs)}).")
                return None
            detail_coeffs_list = [np.zeros_like(c) for c in coeffs]
            detail_coeffs_list[detail_level_index] = coeffs[detail_level_index]
            rec_d = pywt.waverec(detail_coeffs_list, wavelet)
            rec_d_adj = adjust_signal_length(rec_d, target_len)
            reconstructed_bands.append(rec_d_adj)
        approx_coeffs_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_coeffs_list, wavelet)
        rec_a_adj = adjust_signal_length(rec_a, target_len)
        reconstructed_bands.append(rec_a_adj)
        original_adjusted = adjust_signal_length(signal, target_len)
        ordered_bands_rev = reconstructed_bands[::-1]
        all_bands = [original_adjusted] + ordered_bands_rev
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


# --- LÓGICA DE CARGA DE DATOS (COPIADA DE TRAIN) ---
def load_and_preprocess_data(data_directory, hp, scaler, num_expected_nodes=5):
    """Carga y pre-procesa todos los datos, aplicando el scaler cargado."""

    wavelet_name = hp.get('wavelet_name', 'db4')
    wavelet_level = hp.get('wavelet_level', 5)

    logger.info(f"Iniciando carga de datos crudos desde: {data_directory}")
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_directory}")
        return None

    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    for f_path in tqdm(all_files, desc="Cargando archivos crudos"):
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
                    sensor_data_concat[sid] = concatenated_data.squeeze()
                    min_len_raw = min(min_len_raw, len(concatenated_data))
                    sensors_with_data.append(sid)
            except Exception as e:
                logger.error(f"Error concatenando datos para sensor {sid}: {e}.")
        else:
            logger.warning(f"No se cargaron datos válidos para el sensor {sid}.")

    if len(sensors_with_data) != num_expected_nodes:
        missing_sensors = set(range(1, num_expected_nodes + 1)) - set(sensors_with_data)
        logger.error(f"Faltan datos concatenados para los sensores: {missing_sensors}. No se puede continuar.")
        return None

    target_len = min_len_raw
    logger.info(f"Longitud objetivo para DWT: {target_len}")

    logger.info("Aplicando DWT y reconstrucción de bandas...")
    sensor_data_features = {}
    actual_num_features_generated = 0
    for sid in tqdm(sensors_with_data, desc="Generando Features Wavelet"):
        features_2d = apply_dwt_features(sensor_data_concat[sid], wavelet=wavelet_name, level=wavelet_level,
                                         target_len=target_len)
        if features_2d is None:
            logger.error(f"Error generando features para sensor {sid}.")
            return None
        if actual_num_features_generated == 0:
            actual_num_features_generated = features_2d.shape[1]
        sensor_data_features[sid] = features_2d

    if actual_num_features_generated != hp.get('num_features'):
        logger.warning(
            f"Inconsistencia de features: Generadas={actual_num_features_generated}, Esperadas en HP={hp.get('num_features')}")
        # Continuar, pero el scaler puede fallar si la forma no coincide

    del sensor_data_raw, sensor_data_concat
    gc.collect()

    logger.info("Aplicando scaler cargado...")
    sensor_data_scaled_features = {}
    for sid in sensors_with_data:
        try:
            data_to_scale = sensor_data_features.get(sid)
            if data_to_scale is None: continue
            scaled_data = scaler.transform(data_to_scale)
            sensor_data_scaled_features[sid] = scaled_data
        except Exception as e:
            logger.error(f"Error escalando features para sensor {sid}: {e}. Omitiendo sensor.")
            return None  # Fallo crítico si el escalado falla

    logger.info("Pre-procesamiento de datos completado.")
    return sensor_data_scaled_features


# --- FUNCIÓN PRINCIPAL DE EVALUACIÓN ---

def evaluate_by_sensor(run_folder_path, data_folder_path):
    """Carga un modelo y evalúa su error de reconstrucción (MSE) por sensor."""

    logger.info(f"--- Iniciando Evaluación por Sensor ---")
    logger.info(f"Cargando artefactos desde: {run_folder_path}")

    # --- Definir rutas a artefactos ---
    hp_path = os.path.join(run_folder_path, 'hyperparameters_wavelet_gnn.json')
    scaler_path = os.path.join(run_folder_path, 'scaler_wavelet_gnn.gz')
    model_path = os.path.join(run_folder_path, 'best_model_wavelet_gnn.pth')

    # --- Validar que existan ---
    for f_path in [hp_path, scaler_path, model_path]:
        if not os.path.exists(f_path):
            logger.error(f"Archivo requerido no encontrado: {f_path}")
            return

    # --- Cargar Artefactos ---
    try:
        with open(hp_path, 'r') as f:
            hp = json.load(f)
        logger.info("Hiperparámetros cargados.")

        scaler = joblib.load(scaler_path)
        logger.info("Scaler cargado.")
    except Exception as e:
        logger.error(f"Error cargando HPs o Scaler: {e}", exc_info=True)
        return

    # --- Definir constantes desde HPs ---
    num_expected_nodes = 5  # Asumido por el proyecto
    window_size = hp.get('window_size')
    stride = hp.get('stride')
    batch_size = hp.get('batch_size')
    num_features = hp.get('num_features')

    if not all([window_size, stride, batch_size, num_features]):
        logger.error(f"Hiperparámetros críticos faltantes en .json: {hp}")
        return

    # --- Recargar y Pre-procesar Datos ---
    # Esto es crucial para recrear el dataset de validación idéntico
    sensor_data_scaled_features = load_and_preprocess_data(
        data_folder_path,
        hp,
        scaler,
        num_expected_nodes
    )
    if sensor_data_scaled_features is None:
        logger.error("Fallo en la carga y pre-procesamiento de datos. Abortando.")
        return

    # --- Recrear Dataset y DataLoaders ---
    try:
        full_dataset = SpatioTemporalWaveletDataset(
            sensor_data_scaled_features,
            window_size,
            stride,
            num_expected_nodes=num_expected_nodes
        )

        # Recrear la *misma* división train/val
        val_split = 0.15
        total_windows = len(full_dataset)
        val_len = int(val_split * total_windows)
        train_len = total_windows - val_len

        # Usar la MISMA semilla que en el entrenamiento
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Dataset de validación recreado (seed 42): {len(val_dataset)} ventanas.")

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    except Exception as e:
        logger.error(f"Error recreando Dataset/DataLoader: {e}", exc_info=True)
        return

    # --- Cargar Modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")

    try:
        model = SpatioTemporalAutoencoder(
            num_nodes=num_expected_nodes,
            num_features=num_features,
            window_size=window_size,
            gnn_hidden=hp.get('gnn_hidden'),
            gnn_out=hp.get('gnn_out'),
            rnn_hidden=hp.get('rnn_hidden'),
            rnn_layers=hp.get('rnn_layers')
        ).to(device)

        # Cargar los pesos entrenados
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Pesos del modelo cargados exitosamente desde: {model_path}")

    except Exception as e:
        logger.error(f"Error cargando el modelo (¿coincide la arquitectura con los HPs?): {e}", exc_info=True)
        return

    # --- Lógica de Evaluación por Sensor ---
    model.eval()
    edge_index = define_bridge_graph(num_nodes=num_expected_nodes).to(device)

    # Acumuladores para el error y el conteo de muestras
    # Shape: [num_nodes]
    total_error_per_sensor = torch.zeros(num_expected_nodes).to(device)
    total_samples_per_sensor = 0  # Contará (B * T * F)

    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc="Calculando error por sensor"):
            inputs = inputs.to(device)

            # 1. Obtener reconstrucción
            outputs = model(inputs, edge_index)

            # 2. Calcular error cuadrático (sin promediar)
            # Shape: [B, T, N, F]
            loss_tensor = (outputs - inputs) ** 2

            # 3. Sumar el error a lo largo de B, T, y F, pero mantener N
            # Sumar sobre dim 0 (Batch), 1 (Time), 3 (Features)
            # Resultado: un tensor de shape [N] (o [num_nodes])
            batch_error_sum_per_sensor = torch.sum(loss_tensor, dim=(0, 1, 3))

            # 4. Acumular el error sumado
            total_error_per_sensor += batch_error_sum_per_sensor

            # 5. Acumular el número de muestras (B*T*F)
            # B*T*F es el denominador total para cada sensor
            B, T, N, F = inputs.shape
            total_samples_per_sensor += (B * T * F)

    if total_samples_per_sensor == 0:
        logger.error("No se procesaron muestras de validación.")
        return

    # 6. Calcular el MSE final por sensor
    # total_error_per_sensor tiene la suma de todos los errores cuadráticos por sensor
    # total_samples_per_sensor tiene el número total de puntos de datos (B*T*F)
    avg_mse_per_sensor = total_error_per_sensor / total_samples_per_sensor

    # --- Imprimir Resultados ---
    logger.info(f"--- Resultados de Evaluación (MSE) por Sensor ---")
    logger.info(f"Set de datos: Validación ({len(val_dataset)} ventanas)")
    logger.info(f"Modelo: {model_path}")
    logger.info(f"-------------------------------------------------")

    for i in range(num_expected_nodes):
        logger.info(f"  Sensor {i + 1}:   {avg_mse_per_sensor[i].item():.10f}")

    logger.info(f"-------------------------------------------------")
    logger.info(f"  MSE Promedio (Global): {torch.mean(avg_mse_per_sensor).item():.10f}")
    logger.info(f"--- Evaluación Finalizada ---")


if __name__ == '__main__':

    # --- ¡¡¡ CONFIGURAR ESTAS RUTAS !!! ---

    # 1. Apunta esto a la carpeta de resultados que CONTIENE los archivos
    #    .pth, .gz, y .json de tu entrenamiento finalizado.
    RUN_TO_ANALYZE_PATH = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547"

    # 2. Apunta esto a la carpeta de datos CRudos (la misma que usaste para entrenar)
    DATA_FOLDER_PATH = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # --- Fin de Configuración ---

    # Validar rutas
    if not os.path.isdir(RUN_TO_ANALYZE_PATH):
        logger.error(f"Error: Directorio de resultados no encontrado: {RUN_TO_ANALYZE_PATH}")
        sys.exit(1)

    if not os.path.isdir(DATA_FOLDER_PATH):
        logger.error(f"Error: Directorio de datos crudos no encontrado: {DATA_FOLDER_PATH}")
        sys.exit(1)

    try:
        evaluate_by_sensor(RUN_TO_ANALYZE_PATH, DATA_FOLDER_PATH)
    except Exception as e:
        logger.critical(f"Error fatal durante la evaluación: {e}", exc_info=True)
        sys.exit(1)
