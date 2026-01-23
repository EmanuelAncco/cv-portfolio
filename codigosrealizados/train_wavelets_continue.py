# -*- coding: utf-8 -*-
"""
train_wavelet_gnn_continue.py

Continúa el entrenamiento de un modelo STG-AE con características Wavelet
cargando el mejor modelo guardado de una ejecución anterior.
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
import pywt # <--- Importar PyWavelets
import gc # Para liberar memoria

from torch_geometric.nn import GCNConv

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Definiciones (Idénticas al script original train_wavelet_gnn.py) ---

def define_bridge_graph(num_nodes=5):
    """Define la estructura del grafo del puente."""
    if num_nodes != 5:
        logger.warning(f"define_bridge_graph hardcoded for 5 nodes, requested {num_nodes}.")
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    if edge_index.max() >= num_nodes:
         raise ValueError(f"Índice de nodo {edge_index.max()} fuera de rango para {num_nodes} nodos.")
    return edge_index

class SpatioTemporalWaveletDataset(Dataset):
    """Dataset para cargar ventanas espacio-temporales con características Wavelet."""
    def __init__(self, data_dict_features, window_size, stride=1, num_expected_nodes=5):
        self.window_size = window_size
        self.stride = stride
        self.num_expected_nodes = num_expected_nodes
        local_logger = logging.getLogger(self.__class__.__name__)
        if not data_dict_features: raise ValueError("Input feature dictionary is empty.")

        valid_data_dict = {}
        min_len = float('inf')
        expected_num_features = -1
        for sid, data in data_dict_features.items():
            if data is not None and isinstance(data, np.ndarray) and data.ndim == 2 and len(data) >= window_size:
                if expected_num_features == -1: expected_num_features = data.shape[1]
                elif data.shape[1] != expected_num_features:
                    local_logger.error(f"Inconsistent features: Sensor {sid} has {data.shape[1]}, expected {expected_num_features}. Skipping.")
                    continue
                valid_data_dict[sid] = data
                min_len = min(min_len, len(data))
            else: local_logger.warning(f"Invalid/insufficient data for sensor {sid}. Skipping.")

        if not valid_data_dict: raise ValueError("No valid data after filtering.")
        if min_len == float('inf') or min_len < window_size: raise ValueError(f"Min length ({min_len}) insufficient for window_size ({window_size}).")

        processed_data_list = []
        actual_node_ids = []
        self.num_features = 0
        for sid in range(1, self.num_expected_nodes + 1):
             if sid in valid_data_dict:
                 data_node = valid_data_dict[sid][:min_len]
                 processed_data_list.append(data_node)
                 actual_node_ids.append(sid)
                 if self.num_features == 0: self.num_features = data_node.shape[1]
             else: raise ValueError(f"Missing expected sensor data for ID {sid}.")

        if not processed_data_list: raise ValueError("Processed data list is empty.")
        try: self.data = np.stack(processed_data_list, axis=1)
        except ValueError as e: local_logger.error(f"Stacking error. Shapes: {[d.shape for d in processed_data_list]}. Error: {e}"); raise e
        self.num_nodes = self.data.shape[1]
        local_logger.info(f"Stacked data shape: {self.data.shape}. Sensors: {actual_node_ids}")
        if self.num_nodes != self.num_expected_nodes: raise RuntimeError("Node count mismatch.")

        self.n_samples = max(0, (len(self.data) - window_size) // stride + 1)
        local_logger.info(f"Dataset created: {self.num_nodes} nodes, {self.num_features} features, {len(self.data)} points, {self.n_samples} windows.")

    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        if start < 0 or end > len(self.data): raise IndexError(f"Index {idx} out of range.")
        return torch.FloatTensor(self.data[start:end]), torch.FloatTensor(self.data[start:end])

class GNNLayer(nn.Module):
    """Bloque GCN."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)
    def forward(self, x, edge_index):
        edge_index = edge_index.to(x.device)
        x = self.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

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

        logger.info(f"Initializing STAutoencoder: N={num_nodes}, F={num_features}, T={window_size}, GNN_h={gnn_hidden}, GNN_o={gnn_out}, RNN_h={rnn_hidden}, RNN_l={rnn_layers}")

        self.gnn_encoder = GNNLayer(num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)
        self.rnn_encoder = nn.GRU(input_size=self.gnn_encoder_out_dim * num_nodes, hidden_size=self.rnn_encoder_hidden_dim, batch_first=True, num_layers=self.rnn_layers)
        self.rnn_decoder = nn.GRU(input_size=self.rnn_encoder_hidden_dim, hidden_size=self.rnn_decoder_output_dim, batch_first=True, num_layers=self.rnn_layers)
        self.gnn_decoder = GNNLayer(self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index):
        batch_size, T_actual, N_actual, F_actual = x.shape
        if T_actual != self.window_size or N_actual != self.num_nodes or F_actual != self.num_features:
            logger.warning(f"Unexpected input shape: {x.shape}. Expected T={self.window_size}, N={self.num_nodes}, F={self.num_features}.")

        x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)
        edge_index = edge_index.to(x.device)
        try: gnn_encoded = self.gnn_encoder(x_reshaped, edge_index)
        except Exception as e: logger.error(f"GNN Encoder Error: {e}", exc_info=True); raise e
        try:
            gnn_encoded_view = gnn_encoded.reshape(batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
            rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)
        except Exception as e: logger.error(f"Reshape pre-RNN Enc Error: {e}", exc_info=True); raise e
        try: _, h_n = self.rnn_encoder(rnn_input)
        except Exception as e: logger.error(f"RNN Encoder Error: {e}", exc_info=True); raise e
        try:
            latent_vector = h_n[-1].unsqueeze(1).repeat(1, T_actual, 1)
            rnn_decoded, _ = self.rnn_decoder(latent_vector)
        except Exception as e: logger.error(f"RNN Decoder Error: {e}", exc_info=True); raise e
        try: gnn_input_decoder = rnn_decoded.reshape(batch_size * T_actual, N_actual, self.gnn_hidden_dim)
        except Exception as e: logger.error(f"Reshape pre-GNN Dec Error: {e}", exc_info=True); raise e
        try: reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index)
        except Exception as e: logger.error(f"GNN Decoder Error: {e}", exc_info=True); raise e
        try: reconstructed_x = reconstructed_frames.reshape(batch_size, T_actual, N_actual, F_actual)
        except Exception as e: logger.error(f"Final Reshape Error: {e}", exc_info=True); raise e
        return reconstructed_x

def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    """Aplica DWT y reconstruye bandas."""
    if signal is None or signal.ndim != 1 or len(signal) == 0: return None
    if target_len is None: target_len = len(signal)
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        reconstructed_bands = []
        for i in range(level, 0, -1):
            detail_coeffs = [np.zeros_like(c) if idx != (level - i + 1) else coeffs[level - i + 1] for idx, c in enumerate(coeffs)]
            detail_coeffs[0] = np.zeros_like(coeffs[0])
            rec_d_adj = adjust_signal_length(pywt.waverec(detail_coeffs, wavelet), target_len)
            reconstructed_bands.append(rec_d_adj)
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a_adj = adjust_signal_length(pywt.waverec(approx_coeffs, wavelet), target_len)
        reconstructed_bands.append(rec_a_adj)
        original_adjusted = adjust_signal_length(signal, target_len)
        ordered_bands = [original_adjusted] + reconstructed_bands[::-1]
        features = np.stack(ordered_bands, axis=-1)
        if features.shape != (target_len, level + 2): logger.warning(f"Unexpected wavelet features shape: {features.shape}")
        return features
    except Exception as e: logger.error(f"DWT Error on signal len {len(signal)}: {e}", exc_info=True); return adjust_signal_length(signal, target_len)[:, np.newaxis]

def adjust_signal_length(signal, target_len):
    """Ajusta longitud de señal 1D."""
    current_len = len(signal)
    if current_len == target_len: return signal
    elif current_len > target_len: return signal[:target_len]
    else: return np.concatenate((signal, np.zeros(target_len - current_len)))

# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO (Modificada para Continuar) ---

def continue_experiment_wavelet_gnn(data_directory, output_dir, hp, previous_run_dir):
    """
    Continúa el entrenamiento del modelo STG-AE Wavelet desde un checkpoint.
    """
    # --- Configuración del logging ---
    log_file_path = os.path.join(output_dir, 'training_log_wavelet_continued.txt') # Nuevo nombre de log
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging de continuación iniciado. Guardando en: {log_file_path}")
    logger.info(f"--- Continuando Experimento STG-AE con Wavelets ---")
    logger.info(f"Directorio de datos: {data_directory}")
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info(f"Cargando estado desde: {previous_run_dir}")
    logger.info(f"Nuevos Hiperparámetros (hp): {hp}")

    # --- Cargar artefactos de la ejecución anterior ---
    scaler_path = os.path.join(previous_run_dir, 'scaler_wavelet_gnn.gz')
    # Cargar HPs anteriores para obtener la configuración del modelo y wavelet
    prev_hp_path = os.path.join(previous_run_dir, 'hyperparameters_wavelet_gnn.json')
    prev_history_path = os.path.join(previous_run_dir, 'loss_history_wavelet_gnn.json') # Para obtener época inicial
    model_checkpoint_path = os.path.join(previous_run_dir, 'best_model_wavelet_gnn.pth') # El modelo a cargar

    try:
        scaler = joblib.load(scaler_path)
        with open(prev_hp_path, 'r', encoding='utf-8') as f:
            prev_hp = json.load(f)
        with open(prev_history_path, 'r', encoding='utf-8') as f:
             prev_history = json.load(f)
        if not os.path.exists(model_checkpoint_path):
             raise FileNotFoundError(f"Checkpoint del modelo no encontrado: {model_checkpoint_path}")
        logger.info("Artefactos de ejecución anterior cargados (Scaler, HPs, Historial).")
    except Exception as e:
        logger.error(f"Error cargando artefactos de {previous_run_dir}: {e}", exc_info=True)
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    # --- Consistencia de Parámetros ---
    num_expected_nodes = 5
    # Usar config wavelet de HPs anteriores
    wavelet_name = prev_hp.get('wavelet_name', 'db4')
    wavelet_level = prev_hp.get('wavelet_level', 5)
    # Usar num_features de HPs anteriores
    actual_num_features = prev_hp.get('num_features', 1 + wavelet_level + 1)
    # Usar dimensiones del modelo de HPs anteriores
    gnn_hidden = prev_hp.get('gnn_hidden', 32)
    gnn_out = prev_hp.get('gnn_out', 16)
    rnn_hidden = prev_hp.get('rnn_hidden', 64)
    rnn_layers = prev_hp.get('rnn_layers', 2)
    window_size = prev_hp.get('window_size', 64) # Usar window_size anterior
    stride = prev_hp.get('stride', 32)           # Usar stride anterior


    logger.info(f"Configuración del modelo cargada: Features={actual_num_features}, GNN_h={gnn_hidden}, GNN_o={gnn_out}, RNN_h={rnn_hidden}, RNN_l={rnn_layers}, Win={window_size}, Stride={stride}")
    logger.info(f"Configuración Wavelet cargada: Name='{wavelet_name}', Level={wavelet_level}")


    # --- Carga y Procesamiento de Datos (Re-hacer para asegurar consistencia) ---
    # (Esta parte es redundante si los datos no cambian, pero asegura robustez)
    logger.info("Recargando y preprocesando datos...")
    # ... (pegar aquí toda la lógica de carga, concatenación, DWT y escalado del script original) ...
    # ... Asegúrate de usar 'scaler' cargado en lugar de 'scaler.fit()' ...
    # --- Carga de Datos Crudos ---
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    files_processed = 0; files_skipped = 0
    for f_path in tqdm(all_files, desc="Cargando archivos"):
        try:
            filename = os.path.basename(f_path); sid = int(filename.split('_')[0])
            if sid in sensor_data_raw:
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[1], engine='python').values
                if data is not None and data.size > 0: sensor_data_raw[sid].append(data); files_processed += 1
                else: files_skipped += 1
            else: files_skipped += 1
        except Exception as e: logger.error(f"Error cargando {f_path}: {e}. Omitiendo."); files_skipped += 1
    logger.info(f"Carga: {files_processed} procesados, {files_skipped} omitidos.")
    sensor_data_concat = {}; min_len_raw = float('inf'); sensors_with_data = []
    for sid, data_list in sensor_data_raw.items():
        if data_list:
            data_list_2d = [d.reshape(-1, 1) for d in data_list if d.size > 0]
            if not data_list_2d: continue
            try:
                concatenated_data = np.concatenate(data_list_2d, axis=0)
                sensor_data_concat[sid] = concatenated_data.squeeze()
                min_len_raw = min(min_len_raw, len(concatenated_data))
                sensors_with_data.append(sid)
            except Exception as e: logger.error(f"Error concatenando sensor {sid}: {e}")
    if len(sensors_with_data) != num_expected_nodes: logger.error("Faltan sensores. Abortando."); return
    target_len = min_len_raw
    logger.info(f"Longitud objetivo: {target_len}")
    # --- Aplicar Wavelet ---
    logger.info("Aplicando DWT...")
    sensor_data_features = {}
    feature_generation_successful = True
    for sid in tqdm(sensors_with_data, desc="Generando Features Wavelet"):
        features_2d = apply_dwt_features(sensor_data_concat[sid], wavelet=wavelet_name, level=wavelet_level, target_len=target_len)
        if features_2d is None or features_2d.shape[1] != actual_num_features: # Usar actual_num_features
             logger.error(f"Error/Inconsistencia features sensor {sid}. Shape:{features_2d.shape if features_2d is not None else 'None'}. Esperado:(*, {actual_num_features})"); feature_generation_successful = False; break
        sensor_data_features[sid] = features_2d
    if not feature_generation_successful: logger.error("Fallo DWT. Abortando."); return
    logger.info(f"Features Wavelet OK. Shape: ({target_len}, {actual_num_features})")
    del sensor_data_raw, sensor_data_concat; gc.collect()
    # --- Escalar (usando scaler cargado) ---
    logger.info("Escalando features...")
    sensor_data_scaled_features = {}
    for sid, data in sensor_data_features.items():
        try: sensor_data_scaled_features[sid] = scaler.transform(data)
        except Exception as e: logger.error(f"Error escalando sensor {sid}: {e}. Abortando."); return
    logger.info("Escalado OK.")
    del sensor_data_features; gc.collect()
    # --- Crear Datasets ---
    logger.info("Creando datasets...")
    try:
        full_dataset = SpatioTemporalWaveletDataset(sensor_data_scaled_features, window_size, stride, num_expected_nodes)
    except ValueError as e: logger.error(f"Error creando dataset: {e}"); return
    if len(full_dataset) == 0: logger.error("Dataset vacío."); return
    # División Train/Val (usar misma semilla para consistencia)
    val_split = 0.15; total_windows = len(full_dataset); val_len = int(val_split * total_windows); train_len = total_windows - val_len
    if train_len <= 0 or val_len <= 0: logger.error("Ventanas insuficientes."); return
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    logger.info(f"Dataset listo: {len(train_dataset)} train, {len(val_dataset)} val.")
    # --- Fin Recarga Datos ---


    # --- Configuración Entrenamiento (Continuación) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if os.name == 'posix' else 0
    logger.info(f"Dispositivo: {device}, Workers: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    edge_index = define_bridge_graph(num_nodes=num_expected_nodes).to(device)

    # Instanciar el modelo con la configuración cargada
    model = SpatioTemporalAutoencoder(
        num_nodes=num_expected_nodes,
        num_features=actual_num_features,
        window_size=window_size, # Usar valor cargado
        gnn_hidden=gnn_hidden,
        gnn_out=gnn_out,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers
    ).to(device)

    # *** CARGAR PESOS DEL CHECKPOINT ***
    try:
        # Usar map_location para compatibilidad CPU/GPU
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))
        logger.info(f"Pesos del modelo cargados desde: {model_checkpoint_path}")
    except Exception as e:
        logger.error(f"Error cargando pesos del modelo: {e}", exc_info=True)
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo STG-AE (Wavelet Features) instanciado y cargado. Features: {actual_num_features}, Parámetros: {total_params:,}")

    criterion = nn.MSELoss()
    # Usar NUEVOS HPs para el optimizador y scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'], weight_decay=hp.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=hp.get('scheduler_patience', 5),
                                                           factor=hp.get('scheduler_factor', 0.5),
                                                           verbose=True)

    # Recuperar el mejor val_loss anterior
    best_val_loss = prev_hp.get('best_val_loss', float('inf'))
    if best_val_loss is None: best_val_loss = float('inf') # Manejar si se guardó como None
    patience_counter = 0 # Reiniciar contador de paciencia

    # Determinar época inicial
    start_epoch = len(prev_history.get('train_loss', [])) # Época siguiente a la última guardada
    # Nuevo número total de épocas
    total_epochs = hp['epochs'] # Tomar de los nuevos HPs

    # Combinar historiales
    history = prev_history # Empezar con el historial anterior
    # Asegurarse que las listas existen
    for key in ['train_loss', 'val_loss', 'lr']:
        if key not in history: history[key] = []


    # Guardar el mejor modelo en el NUEVO directorio de salida
    best_model_path = os.path.join(output_dir, 'best_model_wavelet_gnn_continued.pth')

    logger.info(f"\n--- Iniciando Continuación del Entrenamiento (Desde Época {start_epoch + 1}) ---")
    start_time_train = datetime.now()

    # --- Bucle Epoch (Continuación) ---
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = datetime.now()
        model.train()
        avg_train_loss = 0.0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Train]', leave=False)
        batch_count_train = 0

        for inputs, _ in progress_bar_train:
            if inputs.shape[1] != window_size or inputs.shape[2] != num_expected_nodes or inputs.shape[3] != actual_num_features:
                 logger.warning(f"Batch train shape inesperado: {inputs.shape}. Omitiendo."); continue

            inputs = inputs.to(device)
            optimizer.zero_grad()
            try:
                outputs = model(inputs, edge_index)
                loss = criterion(outputs, inputs)
                if not torch.isfinite(loss): logger.error(f"Loss NaN/Inf train epoch {epoch+1}. Deteniendo."); return
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                current_loss = loss.item()
                avg_train_loss += current_loss
                batch_count_train += 1
                progress_bar_train.set_postfix({'Loss': f'{current_loss:.6f}'})
            except Exception as e: logger.error(f"Error train epoch {epoch+1}: {e}", exc_info=True); continue

        avg_train_loss /= batch_count_train if batch_count_train > 0 else 1
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # --- Validación ---
        model.eval()
        avg_val_loss = 0.0
        progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Val]', leave=False)
        batch_count_val = 0
        with torch.no_grad():
            for inputs, _ in progress_bar_val:
                if inputs.shape[1] != window_size or inputs.shape[2] != num_expected_nodes or inputs.shape[3] != actual_num_features:
                     logger.warning(f"Batch val shape inesperado: {inputs.shape}. Omitiendo."); continue
                inputs = inputs.to(device)
                try:
                    outputs = model(inputs, edge_index)
                    loss = criterion(outputs, inputs)
                    if not torch.isfinite(loss): logger.warning(f"Loss NaN/Inf val epoch {epoch+1}. Omitiendo batch."); continue
                    avg_val_loss += loss.item()
                    batch_count_val += 1
                    progress_bar_val.set_postfix({'Val Loss': f'{loss.item():.6f}'})
                except Exception as e: logger.error(f"Error val epoch {epoch+1}: {e}", exc_info=True); continue

        avg_val_loss /= batch_count_val if batch_count_val > 0 else 1
        history['val_loss'].append(avg_val_loss)

        epoch_duration = datetime.now() - epoch_start_time
        if not np.isfinite(avg_val_loss): logger.error(f"Epoch {epoch+1} -> Val Loss INVALID. Deteniendo."); return

        logger.info(f"Epoch {epoch + 1}/{total_epochs} -> Lr: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Dur: {epoch_duration})")

        # Scheduler y Early Stopping (usando el best_val_loss cargado)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), best_model_path) # Guardar en el nuevo path
                patience_counter = 0
                logger.info(f"   -> Nuevo mejor modelo (Wavelet GNN Cont.) guardado. Val Loss: {best_val_loss:.6f}")
            except Exception as e: logger.error(f"   -> Error guardando mejor modelo: {e}")
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        # Usar NUEVA paciencia de los HPs
        if current_lr < 1e-7 or patience_counter >= hp['patience']:
            if current_lr < 1e-7: logger.info(f"--- Parada Temprana: LR muy bajo ({current_lr:.2e}). ---")
            else: logger.info(f"--- Parada Temprana: No mejora por {hp['patience']} épocas. ---")
            break

    # --- Fin del Entrenamiento ---
    end_time_train = datetime.now()
    total_training_duration = end_time_train - start_time_train
    logger.info(f"--- Continuación Finalizada (Wavelet GNN) ---")
    logger.info(f"Duración de esta sesión: {total_training_duration}")
    logger.info(f"Mejor Val Loss (global): {best_val_loss:.6f}")

    # --- Guardar Artefactos Finales (en el NUEVO directorio) ---
    logger.info("Guardando artefactos finales...")
    # Scaler (ya estaba guardado, opcional guardar de nuevo)
    scaler_path_new = os.path.join(output_dir, 'scaler_wavelet_gnn_continued.gz')
    try: joblib.dump(scaler, scaler_path_new); logger.info(f"Scaler guardado en: {scaler_path_new}")
    except Exception as e: logger.error(f"Error guardando scaler: {e}")

    # Hiperparámetros (combinar anteriores y nuevos)
    hp_path = os.path.join(output_dir, 'hyperparameters_wavelet_gnn_continued.json')
    try:
        with open(hp_path, 'w', encoding='utf-8') as f:
            hp_save = prev_hp.copy() # Empezar con los HPs originales del modelo
            hp_save.update(hp)     # Actualizar con los HPs de esta ejecución (epochs, lr, etc.)
            hp_save['model_type'] = 'STG-AE (Wavelet Features - Continued)'
            hp_save['total_epochs_run'] = epoch + 1 # Épocas totales corridas
            hp_save['best_val_loss'] = best_val_loss if np.isfinite(best_val_loss) else None
            hp_save['continuation_duration'] = str(total_training_duration)
            json.dump(hp_save, f, indent=4)
        logger.info(f"HPs combinados guardados en: {hp_path}")
    except Exception as e: logger.error(f"Error guardando HPs: {e}")

    # Historial (completo)
    history_path = os.path.join(output_dir, 'loss_history_wavelet_gnn_continued.json')
    try:
        history_safe = {k: [v if np.isfinite(v) else None for v in vals] for k, vals in history.items()}
        with open(history_path, 'w', encoding='utf-8') as f: json.dump(history_safe, f, indent=4)
        logger.info(f"Historial completo guardado en: {history_path}")
    except Exception as e: logger.error(f"Error guardando historial: {e}")

    # Curvas de Pérdida (completa)
    try:
        epochs_total = list(range(1, len(history_safe['train_loss']) + 1))
        train_loss_plot = [l for l in history_safe['train_loss'] if l is not None]
        val_loss_plot = [l for l in history_safe['val_loss'] if l is not None]
        epochs_train = [epochs_total[i] for i, l in enumerate(history_safe['train_loss']) if l is not None]
        epochs_val = [epochs_total[i] for i, l in enumerate(history_safe['val_loss']) if l is not None]

        if train_loss_plot and val_loss_plot:
            plt.figure(figsize=(12, 7))
            plt.plot(epochs_train, train_loss_plot, label='Training Loss', marker='.', linestyle='-')
            plt.plot(epochs_val, val_loss_plot, label='Validation Loss', marker='.', linestyle='--')
            plt.title('Training & Validation Loss (STG-AE Wavelet - Continued)')
            plt.xlabel('Epochs'); plt.ylabel('MSE Loss')
            # ... (misma lógica de escala Y que antes) ...
            all_losses_plot = train_loss_plot + val_loss_plot
            min_loss_plot = min(all_losses_plot) if all_losses_plot else 0.01
            max_loss_plot = max(all_losses_plot) if all_losses_plot else 1.0
            if max_loss_plot / max(min_loss_plot, 1e-9) > 100: plt.yscale('log'); plt.ylabel('MSE Loss (Log Scale)'); plt.ylim(bottom=max(min_loss_plot * 0.8, 1e-9))
            else: plt.ylim(bottom=0)
            plt.legend(); plt.grid(True, linestyle=':')
            loss_curve_path = os.path.join(output_dir, 'loss_curve_wavelet_gnn_continued.png')
            plt.savefig(loss_curve_path, dpi=300); plt.close()
            logger.info(f"Gráfico curvas de pérdida (completo) guardado en: {loss_curve_path}")
        else: logger.warning("No hay datos válidos para plotear curvas de pérdida.")
    except Exception as e: logger.error(f"Error generando gráfico curvas de pérdida: {e}")

    # --- Cerrar Handler ---
    if file_handler:
        logger.info("Cerrando archivo de log.")
        file_handler.close()
        logger.removeHandler(file_handler)


# --- BLOQUE DE EJECUCIÓN ---
if __name__ == '__main__':
    # --- *** MODIFICAR ESTAS RUTAS *** ---
    # Directorio que contiene los DATOS ORIGINALES (limpios)
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    # Directorio que contiene los RESULTADOS DE LA EJECUCIÓN ANTERIOR (scaler, hps, best_model.pth)
    previous_run_folder = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\run_wavelet_db45_20251027-120235"
    # Directorio BASE donde se guardarán los resultados de ESTA EJECUCIÓN DE CONTINUACIÓN
    base_output_dir_cont = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet_cont" # Nuevo directorio base

    # --- Hiperparámetros para la CONTINUACIÓN ---
    HP_CONTINUE = {
        "epochs": 100,         # *** Nuevo número TOTAL de épocas ***
        "batch_size": 32,      # Mantener igual
        "learning_rate": 0.0005, # *** Probar con LR más bajo ***
        "patience": 15,        # *** Aumentar paciencia un poco ***
        "scheduler_patience": 5, # Mantener igual
        "scheduler_factor": 0.5,  # Mantener igual
        "weight_decay": 0
    }
    # NOTA: Los parámetros del modelo (gnn_hidden, rnn_hidden, etc.) y wavelet se leen del archivo `hyperparameters_wavelet_gnn.json` anterior.

    # --- Crear Directorio de Salida Único para esta continuación ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Cargar nombre de wavelet y nivel del HP anterior para el nombre de carpeta
    try:
        with open(os.path.join(previous_run_folder, 'hyperparameters_wavelet_gnn.json'), 'r') as f:
            prev_hp_name = json.load(f)
        w_name = prev_hp_name.get('wavelet_name', 'unknown')
        w_level = prev_hp_name.get('wavelet_level', 'unknown')
    except Exception:
        w_name = 'wavelet'
        w_level = ''

    output_directory = os.path.join(base_output_dir_cont, f"run_wavelet_{w_name}{w_level}_cont_{timestamp}")
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Resultados de continuación se guardarán en: {output_directory}")
    except OSError as e: print(f"Error creando directorio {output_directory}: {e}"); sys.exit(1)

    # --- Validar Directorios ---
    if not os.path.isdir(data_folder_path): print(f"Error: Dir datos no encontrado: {data_folder_path}"); sys.exit(1)
    if not os.path.isdir(previous_run_folder): print(f"Error: Dir ejecución previa no encontrado: {previous_run_folder}"); sys.exit(1)
    if not os.path.exists(os.path.join(previous_run_folder, 'best_model_wavelet_gnn.pth')): print(f"Error: best_model_wavelet_gnn.pth no encontrado en {previous_run_folder}"); sys.exit(1)
    if not os.path.exists(os.path.join(previous_run_folder, 'scaler_wavelet_gnn.gz')): print(f"Error: scaler_wavelet_gnn.gz no encontrado en {previous_run_folder}"); sys.exit(1)
    if not os.path.exists(os.path.join(previous_run_folder, 'hyperparameters_wavelet_gnn.json')): print(f"Error: hyperparameters_wavelet_gnn.json no encontrado en {previous_run_folder}"); sys.exit(1)
    if not os.path.exists(os.path.join(previous_run_folder, 'loss_history_wavelet_gnn.json')): print(f"Error: loss_history_wavelet_gnn.json no encontrado en {previous_run_folder}"); sys.exit(1)


    # --- Ejecutar Continuación ---
    try:
        continue_experiment_wavelet_gnn(data_folder_path, output_directory, HP_CONTINUE, previous_run_folder)
    except Exception as e:
        if logger.hasHandlers(): logger.critical(f"Error fatal: {e}", exc_info=True)
        else: print(f"Error fatal: {e}")
        sys.exit(1)
