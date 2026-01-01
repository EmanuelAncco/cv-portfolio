# -*- coding: utf-8 -*-
"""
train_stgae_physics_informed.py

Modelo 3: STG-AE Físicamente Informado (PINN-GNN).

Esta es la contribución innovadora.
Combina la arquitectura del Modelo 2 (STG-AE + Wavelets) con un grafo
basado en la geometría 3D real del puente, extraída de tus planos.

El grafo ya no es binario (0/1), sino ponderado (1 / distancia),
forzando al GNN a aprender correlaciones espaciales físicamente coherentes.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import joblib
import logging
import sys
import pywt
import gc
import argparse

# Importar las clases y funciones del Modelo 2
# Asumimos que train_stgae_wavelet.py está en el mismo directorio
# o que sus clases están definidas aquí. Por simplicidad, las re-definiremos.
GCNConv = None  # Definir GCNConv como None fuera del try para el linter
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("Error: torch_geometric no está instalado. Ejecuta: pip install torch_geometric")
    GCNConv = None  # Definir también en el except para el linter
    sys.exit(1)

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# FileHandler se añade en run_experiment

# =============================================================================
# DEFINICIÓN DEL GRAFO FÍSICAMENTE INFORMADO (LA INNOVACIÓN)
# =============================================================================


def create_physics_informed_graph(num_nodes=5):
    """
    Crea el grafo ponderado basado en la geometría 3D del puente.
    Las coordenadas se estiman de 'Figure 5' y 'image_7d4462.png'.

    Retorna:
        dict: {'edge_index': Tensor, 'edge_weight': Tensor}
    """
    if num_nodes != 5:
        raise ValueError(
            f"El grafo físico está definido para 5 nodos, se pidieron {num_nodes}.")

    logger.info("Creando grafo Físicamente Informado (PINN-GNN)...")

    # Estimación de coordenadas (X, Y, Z) en metros.
    # Origen (0,0,0) = Extremo "Miraflores" (E1), centro del tablero, nivel del tablero.

    # Basado en 'Figure 5' (Vista en Planta - Ejes X, Y):
    # Span = 13.88 * 4 = 55.52 m
    # Asumimos que los sensores 1/2 y 3/4 están en vigas principales
    # separadas por ~8m (Y=-4, Y=4).
    #
    # S1: (x=13.88, y=-4.0)
    # S2: (x=13.88, y=4.0)
    # S3: (x=13.88 + 13.88 = 27.76, y=-4.0)
    # S4: (x=13.88 + 13.88 = 27.76, y=4.0)
    # S5: (x=13.88 * 3 = 41.64, y=0.0)

    # Basado en 'image_7d4462.png' (Vista de Elevación - Eje Z):
    # Todos los sensores están *debajo* del tablero.
    # Asumimos Z = -1.0m para todos.

    coords = {
        0: np.array([13.88, -4.0, -1.0]),  # Sensor 1
        1: np.array([13.88, 4.0, -1.0]),  # Sensor 2
        2: np.array([27.76, -4.0, -1.0]),  # Sensor 3
        3: np.array([27.76, 4.0, -1.0]),  # Sensor 4
        4: np.array([41.64, 0.0, -1.0])  # Sensor 5
    }

    logger.info("Coordenadas 3D (X,Y,Z) estimadas para los sensores:")
    for i in range(num_nodes):
        logger.info(f"  Sensor {i + 1}: {coords[i]}")

    edge_index_list = []
    edge_weight_list = []

    # Crear un grafo completo (todos los nodos conectados con todos)
    # El GNN aprenderá qué conexiones son importantes,
    # y los pesos le ayudarán.
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Añadir arista en ambas direcciones
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])

            # Calcular distancia Euclidiana
            dist = np.linalg.norm(coords[i] - coords[j])

            # Calcular peso (Inverso de la distancia)
            # Añadir epsilon para evitar división por cero (aunque no debería pasar)
            weight = 1.0 / (dist + 1e-6)

            # Añadir peso para ambas direcciones
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)

    edge_index = torch.tensor(
        edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    logger.info(
        f"Grafo físico creado. Nodos: {num_nodes}, Aristas: {len(edge_weight_list)}")

    return {'edge_index': edge_index, 'edge_weight': edge_weight}


# =============================================================================
# ESTRUCTURAS DE DATOS Y MODELO (Re-definidas desde Modelo 2)
# =============================================================================

# --- Lógica de Grafo (ahora usa la función física) ---


def define_bridge_graph(num_nodes=5, custom_definition=None):
    if custom_definition:
        logger.info("Cargando definición de grafo personalizada (Física).")
        edge_index = custom_definition.get('edge_index')
        edge_weight = custom_definition.get('edge_weight')
        if edge_index is None:
            raise ValueError("Grafo personalizado no tiene 'edge_index'.")
        if edge_index.max() >= num_nodes:
            raise ValueError(
                f"Índice {edge_index.max()} inválido para {num_nodes} nodos.")
        return edge_index, edge_weight

    logger.warning(
        "Creando grafo por defecto. ¿Estás seguro de que no querías el grafo físico?")
    edge_index_list = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [
        2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]]
    edge_index = torch.tensor(
        edge_index_list, dtype=torch.long).t().contiguous()
    return edge_index, None


# --- Funciones de Datos (DWT, Dataset) ---


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
        if len(valid_data_dict) != self.num_expected_nodes:
            missing = set(range(1, self.num_expected_nodes + 1)) - \
                      set(valid_data_dict.keys())
            raise ValueError(f"Faltan datos para sensores: {missing}.")
        if min_len < window_size:
            raise ValueError(f"Longitud mínima ({min_len}) insuficiente.")
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


# --- Arquitectura del Modelo ---


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


# --- Función de Experimento (idéntica a Modelo 2) ---


def run_experiment_stgae(data_directory, output_dir, hp, resume_path=None, custom_graph_def=None):
    log_file_path = os.path.join(
        output_dir, f"training_log_stgae_PHYSICS{'_RESUME' if resume_path else ''}.txt")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(
        f"Logging de STG-AE (FÍSICO) iniciado. Guardando en: {log_file_path}")
    logger.info(f"Directorio de datos: {data_directory}")
    logger.info(f"Directorio de salida: {output_dir}")
    if custom_graph_def:
        logger.info("Usando definición de GRAFO FÍSICAMENTE INFORMADO.")

    history = {'train_loss': [], 'val_loss': [],
               'lr': []}
    start_epoch = 0
    best_val_loss = float('inf')
    hp_original = {}
    if resume_path:
        logger.info(f"--- REANUDANDO ENTRENAMIENTO DESDE: {resume_path} ---")
        try:
            # Buscar .json físico
            hp_original_path = os.path.join(
                resume_path, 'hyperparameters_stgae_physics.json')
            scaler_path = os.path.join(
                resume_path, 'scaler_stgae_physics.gz')
            model_path = os.path.join(
                resume_path, 'best_model_stgae_physics.pth')
            history_path = os.path.join(
                resume_path, 'loss_history_stgae_physics.json')
            with open(hp_original_path, 'r') as f:
                hp_original = json.load(f)
            with open(history_path, 'r') as f:
                history = json.load(f)
            hp_combined = hp_original.copy()
            hp_combined.update(hp)
            hp = hp_combined
            best_val_loss = hp_original.get('best_val_loss', float('inf'))
            if best_val_loss is None:
                best_val_loss = float('inf')
            start_epoch = len(history.get('train_loss', []))
            logger.info(
                f"Reanudando desde epoch {start_epoch + 1}. Mejor Val Loss anterior: {best_val_loss:.6f}")
        except Exception as e:
            logger.error(
                f"Error cargando artefactos de reanudación: {e}", exc_info=True)
            if file_handler:
                file_handler.close()
                logger.removeHandler(file_handler)
            return
    else:
        logger.info("--- INICIANDO NUEVO ENTRENAMIENTO (STG-AE FÍSICO) ---")
        scaler_path = None
        model_path = None
    logger.info(f"Hiperparámetros (finales): {hp}")

    num_expected_nodes = 5
    wavelet_name = hp.get('wavelet_name', 'db4')
    wavelet_level = hp.get('wavelet_level', 5)
    if resume_path:
        num_expected_features = hp.get('num_features')
    else:
        num_expected_features = 1 + 1 + wavelet_level
    logger.info(
        f"Config Wavelet: {wavelet_name}, Lvl {wavelet_level} -> {num_expected_features} features")

    logger.info("Cargando datos crudos...")
    all_files = [os.path.join(data_directory, f) for f in os.listdir(
        data_directory) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_directory}")
        return
    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    for f_path in tqdm(all_files, desc="Cargando archivos crudos"):
        try:
            sid = int(os.path.basename(f_path).split('_')[0])
            if sid in sensor_data_raw:
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[
                    1], engine='python', on_bad_lines='warn').values
                if data is not None and data.size > 0:
                    sensor_data_raw[sid].append(data)
        except Exception:
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
                continue
            try:
                concatenated_data = np.concatenate(data_list_valid, axis=0)
                if concatenated_data.size > 0:
                    sensor_data_concat[sid] = concatenated_data.squeeze()
                    min_len_raw = min(min_len_raw, len(concatenated_data))
                    sensors_with_data.append(sid)
            except Exception:
                pass
    if len(sensors_with_data) != num_expected_nodes:
        logger.error(
            f"Faltan datos para sensores: {set(range(1, 6)) - set(sensors_with_data)}. Abortando.")
        if file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)
        return
    target_len = min_len_raw
    logger.info(f"Longitud mínima/objetivo: {target_len}")
    logger.info("Aplicando DWT...")
    sensor_data_features = {}
    actual_num_features_generated = 0
    for sid in tqdm(sensors_with_data, desc="Generando Features Wavelet"):
        features_2d = apply_dwt_features(
            sensor_data_concat[sid], wavelet=wavelet_name, level=wavelet_level, target_len=target_len)
        if features_2d is None:
            logger.error(f"Error features S{sid}. Abortando.")
            if file_handler:
                file_handler.close()
                logger.removeHandler(file_handler)
            return
        sensor_data_features[sid] = features_2d
        actual_num_features_generated = features_2d.shape[1]
    if actual_num_features_generated != num_expected_features:
        logger.warning(
            f"Features generadas ({actual_num_features_generated}) != esperadas ({num_expected_features}). Usando {actual_num_features_generated}.")
        num_expected_features = actual_num_features_generated
    hp['num_features'] = num_expected_features
    logger.info(
        f"Features Wavelet generadas. Shape: ({target_len}, {num_expected_features})")
    del sensor_data_raw, sensor_data_concat
    gc.collect()
    if resume_path and scaler_path is not None:
        logger.info(f"Cargando scaler: {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        logger.info("Ajustando nuevo StandardScaler...")
        scaler = StandardScaler()
        all_features_flat = np.concatenate(
            [data for data in sensor_data_features.values()], axis=0)
        scaler.fit(all_features_flat)
        del all_features_flat
        gc.collect()
        logger.info("StandardScaler ajustado.")
    sensor_data_scaled_features = {}
    for sid in sensors_with_data:
        sensor_data_scaled_features[sid] = scaler.transform(
            sensor_data_features[sid])
    logger.info("Escalado completado.")
    del sensor_data_features
    gc.collect()
    try:
        full_dataset = SpatioTemporalWaveletDataset(
            sensor_data_scaled_features, hp['window_size'], hp['stride'], num_expected_nodes)
    except ValueError as e:
        logger.error(f"Error creando dataset: {e}", exc_info=True)
        if file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)
        return
    if len(full_dataset) == 0:
        logger.error("Dataset vacío. Abortando.")
        return
    actual_num_features = full_dataset.num_features
    actual_num_nodes = full_dataset.num_nodes
    hp['num_features'] = actual_num_features
    hp['num_nodes'] = actual_num_nodes
    logger.info(
        f"Dataset listo: {len(full_dataset)} ventanas. Shape: ({hp['window_size']}, {actual_num_nodes}, {actual_num_features})")
    val_split = 0.15
    val_len = int(val_split * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    logger.info(
        f"Dataset dividido: {len(train_dataset)} train, {len(val_dataset)} val.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    num_workers = 4 if os.name == 'posix' else 0
    batch_size = hp['batch_size']
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=(device.type == 'cuda'), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
        logger.info(f"DataLoaders creados. Batch Size: {batch_size}")
    except Exception as e:
        logger.error(f"Error DataLoaders: {e}", exc_info=True)
        if file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)
        return

    # --- Grafo (usando el custom_graph_def) ---
    try:
        edge_index, edge_weight = define_bridge_graph(
            actual_num_nodes, custom_graph_def)
        edge_index = edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
            logger.info(
                "Grafo FÍSICO con pesos (edge_weight) cargado en dispositivo.")
        else:
            logger.error(
                "Error: El grafo físico DEBERÍA tener pesos pero no se generaron.")
            return
    except Exception as e:
        logger.error(f"Error definiendo el grafo físico: {e}", exc_info=True)
        if file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)
        return

    model = SpatioTemporalAutoencoder(actual_num_nodes, actual_num_features, hp['window_size'], hp.get(
        'gnn_hidden', 128), hp.get('gnn_out', 64), hp.get('rnn_hidden', 256), hp.get('rnn_layers', 2)).to(device)
    if resume_path and model_path:
        try:
            logger.info(f"Cargando pesos: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Pesos cargados.")
        except Exception as e:
            logger.error(f"Error cargando state_dict: {e}.", exc_info=True)
            if file_handler:
                file_handler.close()
                logger.removeHandler(file_handler)
            return
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Modelo STG-AE (FÍSICO) creado. Parámetros: {total_params:,}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(
    ), lr=hp['learning_rate'], weight_decay=hp.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=hp.get(
        'scheduler_patience', 5), factor=hp.get('scheduler_factor', 0.5), verbose=False)

    patience_counter = 0
    best_model_path = os.path.join(
        output_dir, 'best_model_stgae_physics.pth')
    total_epochs = start_epoch + hp['epochs']
    logger.info(
        f"--- Iniciando Entrenamiento (FÍSICO)... {hp['epochs']} épocas (desde {start_epoch + 1} hasta {total_epochs}) ---")
    start_time_train = datetime.now()

    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = datetime.now()
        model.train()
        avg_train_loss = 0.0
        batch_count_train = 0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Train]', leave=False, unit="batch")
        for inputs, _ in progress_bar_train:
            inputs = inputs.to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                outputs = model(inputs, edge_index, edge_weight)
                loss = criterion(outputs, inputs)
                if not torch.isfinite(loss):
                    logger.error(
                        f"Loss NaN/Inf en train epoch {epoch + 1}. Deteniendo.")
                    if file_handler:
                        file_handler.close()
                        logger.removeHandler(file_handler)
                    return
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                avg_train_loss += loss.item()
                batch_count_train += 1
                progress_bar_train.set_postfix({'Loss': f'{loss.item():.6f}'})
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(
                        f"CUDA OOM en train epoch {epoch + 1}. Abortando.", exc_info=False)
                    if file_handler:
                        file_handler.close()
                        logger.removeHandler(file_handler)
                    return
                logger.error(
                    f"Error Runtime en train epoch {epoch + 1}: {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(
                    f"Error inesperado en train epoch {epoch + 1}: {e}", exc_info=True)
                continue
        if batch_count_train > 0:
            avg_train_loss /= batch_count_train
        history['train_loss'].append(
            avg_train_loss if batch_count_train > 0 else None)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        model.eval()
        avg_val_loss = 0.0
        batch_count_val = 0
        progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Val]', leave=False, unit="batch")
        with torch.no_grad():
            for inputs, _ in progress_bar_val:
                inputs = inputs.to(device)
                try:
                    outputs = model(inputs, edge_index, edge_weight)
                    loss = criterion(outputs, inputs)
                    if torch.isfinite(loss):
                        avg_val_loss += loss.item()
                        batch_count_val += 1
                except Exception as e:
                    logger.error(
                        f"Error en val epoch {epoch + 1}: {e}", exc_info=True)
                    continue
        if batch_count_val > 0:
            avg_val_loss /= batch_count_val
        else:
            avg_val_loss = float('inf')
        history['val_loss'].append(
            avg_val_loss if np.isfinite(avg_val_loss) else None)
        epoch_duration = datetime.now() - epoch_start_time
        scheduler.step(avg_val_loss)
        logger.info(
            f"Epoch {epoch + 1}/{total_epochs} -> Lr: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Dur: {epoch_duration})")

        if avg_val_loss < best_val_loss and np.isfinite(avg_val_loss):
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"   -> Nuevo mejor modelo (FÍSICO) guardado. Val Loss: {best_val_loss:.6f}")
            try:  # Guardar artefactos
                joblib.dump(scaler, os.path.join(
                    output_dir, 'scaler_stgae_physics.gz'))
                hp['best_val_loss'] = best_val_loss
                hp['total_params'] = total_params
                hp['training_duration_total'] = str(
                    datetime.now() - start_time_train)
                with open(os.path.join(output_dir, 'hyperparameters_stgae_physics.json'), 'w') as f:
                    json.dump(hp, f, indent=4)
                with open(os.path.join(output_dir, 'loss_history_stgae_physics.json'), 'w') as f:
                    json.dump(history, f, indent=4)
            except Exception as e:
                logger.warning(
                    f"Error guardando artefactos intermedios: {e}")
        else:
            patience_counter += 1
        if optimizer.param_groups[0]['lr'] < 1e-7:
            logger.info("--- Parada Temprana: LR bajo. ---")
            break
        if patience_counter >= hp['patience']:
            logger.info(
                f"--- Parada Temprana: Paciencia {hp['patience']} agotada. ---")
            break

    logger.info(f"--- Entrenamiento (FÍSICO) Finalizado ---")
    logger.info(f"Mejor pérdida de validación: {best_val_loss:.6f}")
    try:  # Plot
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        train_loss_plot = [
            l for l in history.get('train_loss', []) if l is not None]
        val_loss_plot = [
            l for l in history.get('val_loss', []) if l is not None]
        epochs_train = [epochs[i]
                        for i, l in enumerate(history.get('train_loss', [])) if l is not None]
        epochs_val = [epochs[i]
                      for i, l in enumerate(history.get('val_loss', [])) if l is not None]
        plt.figure(figsize=(12, 7))
        plt.plot(epochs_train, train_loss_plot,
                 label='Training Loss', marker='.')
        plt.plot(epochs_val, val_loss_plot,
                 label='Validation Loss', marker='.')
        if resume_path:
            plt.axvline(x=start_epoch + 0.5, color='r',
                        linestyle='--', label=f'Resumed at Epoch {start_epoch + 1}')
        plt.title('Training & Validation Loss (STG-AE FÍSICAMENTE INFORMADO)')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, linestyle=':')
        loss_curve_path = os.path.join(
            output_dir, 'loss_curve_stgae_physics.png')
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()
        logger.info(f"Gráfico de pérdidas guardado en: {loss_curve_path}")
    except Exception as e:
        logger.error(f"Error generando gráfico de pérdidas: {e}", exc_info=True)
    if file_handler:
        logger.info("Cerrando archivo de log.")
        file_handler.close()
        logger.removeHandler(file_handler)


# --- BLOQUE DE EJECUCIÓN ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrenar Modelo STG-AE (FÍSICO) para SHM')
    parser.add_argument('--data_dir', type=str, default=r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar",
                        help='Ruta al directorio de datos .txt.')
    parser.add_argument('--output_dir', type=str,
                        default=r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm",
                        help='Ruta base para resultados.')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='(Opcional) Ruta a una carpeta de ejecución anterior (física) para reanudar.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Número de épocas.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: Directorio de datos no encontrado: {args.data_dir}")
        sys.exit(1)
    if not os.path.isdir(args.output_dir):
        print(
            f"Error: Directorio de salida base no encontrado: {args.output_dir}")
        sys.exit(1)
    if args.resume_path and not os.path.isdir(args.resume_path):
        print(
            f"Error: Directorio de reanudación no encontrado: {args.resume_path}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_STGAE-PHYSICS_lr{args.lr}_bs{args.batch_size}_{timestamp}"
    if args.resume_path:
        run_name = f"RESUME-PHYSICS_{os.path.basename(args.resume_path)}_e{args.epochs}_{timestamp}"

    final_output_dir = os.path.join(args.output_dir, run_name)
    try:
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"Resultados (FÍSICOS) se guardarán en: {final_output_dir}")
    except OSError as e:
        print(f"Error creando directorio de salida {final_output_dir}: {e}")
        sys.exit(1)

    # Hiperparámetros (HP) - Idénticos al Modelo 2
    HP = {
        "window_size": 64, "stride": 32, "wavelet_name": "db4", "wavelet_level": 5,
        "gnn_hidden": 128, "gnn_out": 64, "rnn_hidden": 256, "rnn_layers": 2,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "patience": 10, "scheduler_patience": 5, "scheduler_factor": 0.5, "weight_decay": 1e-5
    }

    # --- ¡LA CLAVE! ---
    # 1. Crear el grafo físico
    try:
        physics_graph_definition = create_physics_informed_graph(num_nodes=5)
    except Exception as e:
        print(f"Error fatal creando el grafo físico: {e}")
        sys.exit(1)

    # 2. Ejecutar el experimento, pasando el grafo físico
    try:
        run_experiment_stgae(
            data_directory=args.data_dir,
            output_dir=final_output_dir,
            hp=HP,
            resume_path=args.resume_path,
            custom_graph_def=physics_graph_definition  # <-- Aquí está la innovación
        )
    except Exception as e:
        logger.critical(
            f"Error fatal durante la ejecución del experimento STG-AE (FÍSICO): {e}", exc_info=True)
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close()
                    logger.removeHandler(handler)
                except:
                    pass
        sys.exit(1)

