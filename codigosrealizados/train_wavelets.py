# -*- coding: utf-8 -*-
"""
train_wavelet_gnn.py

Entrena un Autoencoder Gráfico Espacio-Temporal (STG-AE) utilizando
características extraídas mediante la Transformada Wavelet Discreta (DWT)
además de la señal original.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split # No se usa aquí directamente
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import joblib
import logging
import sys
import pywt # <--- Importar PyWavelets

from torch_geometric.nn import GCNConv # Asegúrate de tener torch_geometric instalado

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger() # Logger Raíz
logger.setLevel(logging.INFO)

# Limpiar handlers existentes
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Handler para consola
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# FileHandler se añade en run_experiment

# --- ESTRUCTURA DEL GRAFO (Idéntica al original) ---

def define_bridge_graph(num_nodes=5):
    """Define la estructura del grafo del puente."""
    # Nodos: 0 a num_nodes-1
    if num_nodes != 5:
        # Si se usa un número diferente de nodos, la topología debe redefinirse
        # Aquí mantenemos la original para 5 nodos.
        logger.warning(f"define_bridge_graph está codificado para 5 nodos, pero se pidieron {num_nodes}. Usando la topología de 5 nodos.")
        # Considerar lanzar un error o implementar lógica para otros `num_nodes`.

    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    if edge_index.max() >= num_nodes:
         raise ValueError(f"Índice de nodo {edge_index.max()} fuera de rango para {num_nodes} nodos.")
    return edge_index


# --- LÓGICA DE DATOS (Modificada para Wavelets) ---

class SpatioTemporalWaveletDataset(Dataset):
    """
    Dataset para cargar ventanas espacio-temporales con características Wavelet.
    Espera que data_dict contenga arrays de shape (time, num_features).
    """
    def __init__(self, data_dict_features, window_size, stride=1, num_expected_nodes=5):
        self.window_size = window_size
        self.stride = stride
        self.num_expected_nodes = num_expected_nodes
        local_logger = logging.getLogger(self.__class__.__name__)

        if not data_dict_features:
            local_logger.error("El diccionario de datos de características está vacío.")
            raise ValueError("El diccionario de datos de características está vacío.")

        # Validar datos y encontrar longitud mínima
        valid_data_dict = {}
        min_len = float('inf')
        expected_num_features = -1

        for sid, data in data_dict_features.items():
            if data is not None and isinstance(data, np.ndarray) and data.ndim == 2 and len(data) >= window_size:
                if expected_num_features == -1:
                    expected_num_features = data.shape[1]
                elif data.shape[1] != expected_num_features:
                    local_logger.error(f"Inconsistencia en número de features. Sensor {sid} tiene {data.shape[1]}, se esperaban {expected_num_features}. Omitiendo.")
                    continue # Saltar este sensor

                valid_data_dict[sid] = data
                min_len = min(min_len, len(data))
            else:
                shape_info = data.shape if hasattr(data, 'shape') else 'N/A'
                len_info = len(data) if hasattr(data, '__len__') else 'N/A'
                local_logger.warning(f"Datos inválidos/insuficientes para sensor {sid}. Shape: {shape_info}, Longitud: {len_info}, WinSize: {window_size}. Omitiendo.")

        if not valid_data_dict:
            local_logger.error("No hay datos válidos en el diccionario después de filtrar.")
            raise ValueError("No hay datos válidos en el diccionario después de filtrar.")

        if min_len == float('inf') or min_len < window_size:
             len_val = min_len if min_len != float('inf') else 'N/A'
             local_logger.error(f"Longitud mínima ({len_val}) insuficiente para window_size ({window_size}).")
             raise ValueError(f"Longitud mínima ({len_val}) insuficiente para window_size ({window_size}).")

        # Asegurar que tenemos datos para todos los nodos esperados y obtener num_features real
        processed_data_list = []
        actual_node_ids = []
        self.num_features = 0
        for sid in range(1, self.num_expected_nodes + 1):
             if sid in valid_data_dict:
                 data_node = valid_data_dict[sid][:min_len] # Truncar
                 processed_data_list.append(data_node)
                 actual_node_ids.append(sid)
                 if self.num_features == 0: # Tomar num_features del primer nodo válido
                     self.num_features = data_node.shape[1]
             else:
                  local_logger.error(f"Faltan datos para el sensor esperado {sid}. No se puede continuar.")
                  raise ValueError(f"Faltan datos para el sensor esperado {sid}.")

        if not processed_data_list:
             local_logger.error("La lista de datos procesados está vacía.")
             raise ValueError("La lista de datos procesados está vacía.")

        # Apilar a lo largo de una nueva dimensión (axis=1 para nodos)
        # Shape esperado: (min_len, num_nodes, num_features)
        try:
            self.data = np.stack(processed_data_list, axis=1)
        except ValueError as e:
             local_logger.error(f"Error al apilar datos. Shapes individuales: {[d.shape for d in processed_data_list]}. Error: {e}")
             raise e

        self.num_nodes = self.data.shape[1]
        local_logger.info(f"Datos apilados con shape: {self.data.shape}. Sensores usados: {actual_node_ids}")

        if self.num_nodes != self.num_expected_nodes:
             # Esto no debería ocurrir si las validaciones anteriores son correctas
             local_logger.error(f"Inconsistencia: Nodos procesados ({self.num_nodes}) != Nodos esperados ({self.num_expected_nodes}).")
             raise RuntimeError("Error interno en la creación del dataset.")

        # Calcular número de muestras
        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples <= 0:
            local_logger.warning(f"Número de muestras <= 0 ({self.n_samples}). Longitud datos: {len(self.data)}, WinSize: {window_size}, Stride: {stride}.")
            self.n_samples = 0
        local_logger.info(f"Dataset creado con {self.num_nodes} nodos, {self.num_features} features, {len(self.data)} puntos, {self.n_samples} ventanas.")


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        if start < 0 or end > len(self.data):
             local_logger = logging.getLogger(self.__class__.__name__)
             local_logger.error(f"Índice {idx} genera rango [{start}, {end}) fuera de límites [{0}, {len(self.data)}]. Stride={self.stride}, WinSize={self.window_size}, N_Samples={self.n_samples}")
             raise IndexError(f"Índice {idx} fuera de rango.")

        window = self.data[start:end] # Shape: (window_size, num_nodes, num_features)
        # No necesita reshape adicional aquí si GNN espera (..., N, F) y RNN (B, T, N*F_gnn)
        return torch.FloatTensor(window), torch.FloatTensor(window)


# --- ARQUITECTURA DEL GNN AUTOENCODER (Idéntica, solo cambia num_features) ---

class GNNLayer(nn.Module):
    """Bloque de capas GCN."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01) # Usar LeakyReLU puede ser más robusto

    def forward(self, x, edge_index):
        # x shape: [B*T, N, F_in] o [N, F_in] si se aplica antes
        edge_index = edge_index.to(x.device) # Asegurar dispositivo
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


class SpatioTemporalAutoencoder(nn.Module):
    """
    Arquitectura ST-GAE. Ahora `num_features` puede ser > 1.
    """
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden=32, gnn_out=16, rnn_hidden=64, rnn_layers=2):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_features = num_features # Ahora puede ser > 1
        self.gnn_hidden_dim = gnn_hidden
        self.gnn_encoder_out_dim = gnn_out
        self.rnn_encoder_hidden_dim = rnn_hidden
        self.rnn_layers = rnn_layers

        local_logger = logging.getLogger(self.__class__.__name__)

        # La salida del decoder GRU debe ser capaz de reconstruir la entrada del GNN Decoder
        # GNN Decoder espera `gnn_hidden_dim` features por nodo
        self.rnn_decoder_output_dim = self.gnn_hidden_dim * num_nodes

        local_logger.info(f"Initializing STAutoencoder: N={num_nodes}, F={num_features}, T={window_size}")
        local_logger.info(f"  GNN Encoder: {num_features} -> {self.gnn_hidden_dim} -> {self.gnn_encoder_out_dim}")
        local_logger.info(f"  RNN Encoder: Input={self.gnn_encoder_out_dim * num_nodes}, Hidden={self.rnn_encoder_hidden_dim}, Layers={self.rnn_layers}")
        local_logger.info(f"  RNN Decoder: Input={self.rnn_encoder_hidden_dim}, Hidden={self.rnn_decoder_output_dim}, Layers={self.rnn_layers}")
        local_logger.info(f"  GNN Decoder: {self.gnn_hidden_dim} -> {self.gnn_hidden_dim} -> {num_features}") # Ojo: salida es num_features

        # Capas
        self.gnn_encoder = GNNLayer(num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)

        self.rnn_encoder = nn.GRU(input_size=self.gnn_encoder_out_dim * num_nodes,
                                  hidden_size=self.rnn_encoder_hidden_dim,
                                  batch_first=True, num_layers=self.rnn_layers)

        self.rnn_decoder = nn.GRU(input_size=self.rnn_encoder_hidden_dim,
                                  hidden_size=self.rnn_decoder_output_dim,
                                  batch_first=True, num_layers=self.rnn_layers)

        # GNN Decoder: Entrada es gnn_hidden_dim, Salida es num_features
        self.gnn_decoder = GNNLayer(self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index):
        # x shape: [B, T, N, F]
        batch_size, T_actual, N_actual, F_actual = x.shape

        # Validaciones de shape en forward
        if T_actual != self.window_size or N_actual != self.num_nodes or F_actual != self.num_features:
            logger.warning(f"Shape de entrada inesperado en forward: {x.shape}. Esperado: ({batch_size}, {self.window_size}, {self.num_nodes}, {self.num_features}). Intentando continuar...")

        # 1. Preparar datos para GNN Encoder
        # Reshape de [B, T, N, F] -> [B*T, N, F]
        # Usar dimensiones reales por si acaso
        x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)


        # Mover edge_index al dispositivo de x
        edge_index = edge_index.to(x.device)

        # 2. GNN Encoder (Vectorizado)
        # gnn_encoded shape: [B*T, N, gnn_encoder_out_dim]
        try:
            gnn_encoded = self.gnn_encoder(x_reshaped, edge_index)
        except Exception as e:
            logger.error(f"Error en GNN Encoder. Input shape: {x_reshaped.shape}. Error: {e}", exc_info=True)
            raise e

        # 3. Preparar datos para RNN Encoder
        # Reshape de [B*T, N, gnn_out] -> [B, T, N * gnn_out]
        # Usar T_actual, N_actual por seguridad
        try:
            gnn_encoded_view = gnn_encoded.reshape(batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
            rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)  # Shape: [B, T, N*gnn_out]
        except Exception as e:
            logger.error(f"Error en reshape pre-RNN Encoder. GNN Encoded shape: {gnn_encoded.shape}. Error: {e}", exc_info=True)
            raise e

        # 4. RNN Encoder
        try:
             _, h_n = self.rnn_encoder(rnn_input)  # h_n shape: [num_layers, B, rnn_hidden]
        except Exception as e:
            logger.error(f"Error en RNN Encoder. Input shape: {rnn_input.shape}. Error: {e}", exc_info=True)
            raise e


        # 5. RNN Decoder
        # Usar el último estado oculto
        try:
            # h_n[-1] es shape [B, rnn_hidden]
            latent_vector = h_n[-1].unsqueeze(1).repeat(1, T_actual, 1)  # Shape: [B, T, rnn_hidden]
            rnn_decoded, _ = self.rnn_decoder(latent_vector)  # Shape: [B, T, rnn_decoder_output_dim (N*gnn_hidden)]
        except Exception as e:
            logger.error(f"Error en RNN Decoder. Latent shape: {latent_vector.shape if 'latent_vector' in locals() else 'N/A'}. Error: {e}", exc_info=True)
            raise e


        # 6. Preparar datos para GNN Decoder
        # rnn_decoder_output_dim es self.gnn_hidden_dim * self.num_nodes
        # Reshape de [B, T, N*gnn_hidden] -> [B*T, N, gnn_hidden]
        try:
            gnn_input_decoder = rnn_decoded.reshape(batch_size * T_actual, N_actual, self.gnn_hidden_dim)
        except Exception as e:
            logger.error(f"Error en reshape pre-GNN Decoder. RNN Decoded shape: {rnn_decoded.shape}. Error: {e}", exc_info=True)
            raise e

        # 7. GNN Decoder (Vectorizado)
        # reconstructed_frames shape: [B*T, N, F_out (num_features)]
        try:
            reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index)
        except Exception as e:
            logger.error(f"Error en GNN Decoder. Input shape: {gnn_input_decoder.shape}. Error: {e}", exc_info=True)
            raise e

        # 8. Reshape final
        # Reshape de [B*T, N, F] -> [B, T, N, F]
        try:
            reconstructed_x = reconstructed_frames.reshape(batch_size, T_actual, N_actual, F_actual)
        except Exception as e:
            logger.error(f"Error en reshape final. Reconstructed frames shape: {reconstructed_frames.shape}. Error: {e}", exc_info=True)
            raise e

        return reconstructed_x


# --- FUNCIONES AUXILIARES WAVELET ---

def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    """
    Aplica DWT multinivel a una señal 1D y reconstruye las bandas.
    Devuelve un array 2D (time, num_features) donde las features son
    la señal original y las bandas reconstruidas (A_last, D_last, ..., D1).
    """
    if signal is None or signal.ndim != 1 or len(signal) == 0:
        logger.warning("apply_dwt_features recibió señal inválida. Devolviendo None.")
        return None

    if target_len is None:
        target_len = len(signal)

    try:
        # Descomposición
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Reconstrucción de bandas individuales
        reconstructed_bands = []

        # Reconstruir cada nivel de detalle
        for i in range(level, 0, -1):
            # Crear lista de coeficientes con solo el nivel i activo
            detail_coeffs = [np.zeros_like(c) if idx != (level - i + 1) else coeffs[level - i + 1] for idx, c in enumerate(coeffs)]
            # Poner la aproximación a cero también
            detail_coeffs[0] = np.zeros_like(coeffs[0])
            # Reconstruir
            rec_d = pywt.waverec(detail_coeffs, wavelet)
            # Ajustar longitud (padding o truncado)
            rec_d_adj = adjust_signal_length(rec_d, target_len)
            reconstructed_bands.append(rec_d_adj)

        # Reconstruir la última aproximación
        approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_coeffs, wavelet)
        rec_a_adj = adjust_signal_length(rec_a, target_len)
        reconstructed_bands.append(rec_a_adj)

        # Apilar la señal original y las bandas reconstruidas
        # Orden: Original, D1, D2, ..., D_last, A_last
        original_adjusted = adjust_signal_length(signal, target_len)
        # Las bandas se añadieron de D_last a D1, y luego A_last. Reordenar D1...D_last, A_last
        # reconstructed_bands contiene [D_level, D_(level-1), ..., D1, A_level]
        # Queremos [Original, D1, D2, ..., D_level, A_level]
        ordered_bands = [original_adjusted] + reconstructed_bands[::-1] # Invertir lista de bandas D y A
        features = np.stack(ordered_bands, axis=-1) # Apilar en la última dimensión

        # Verificar forma final
        if features.shape != (target_len, level + 1 + 1): # Original + Levels Detalles + 1 Aproximación
             logger.warning(f"Shape inesperado en features wavelet: {features.shape}. Esperado: ({target_len}, {level + 2})")

        return features

    except Exception as e:
        logger.error(f"Error aplicando DWT a señal de longitud {len(signal)}: {e}", exc_info=True)
        # Devolver solo la señal original ajustada si falla DWT
        return adjust_signal_length(signal, target_len)[:, np.newaxis]

def adjust_signal_length(signal, target_len):
    """Ajusta la longitud de una señal 1D (padding o truncado)."""
    current_len = len(signal)
    if current_len == target_len:
        return signal
    elif current_len > target_len:
        # Truncar (centrado si es posible, aquí simple)
        return signal[:target_len]
    else:
        # Padding (con ceros o repitiendo último valor)
        # Usaremos padding con ceros por simplicidad
        padding = np.zeros(target_len - current_len)
        return np.concatenate((signal, padding))

# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO (Modificada para Wavelets) ---

def run_experiment_wavelet_gnn(data_directory, output_dir, hp):
    """
    Función principal para entrenar el modelo STG-AE con características Wavelet.
    """
    # --- Configuración del logging de archivo ---
    log_file_path = os.path.join(output_dir, 'training_log_wavelet.txt')
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging de entrenamiento Wavelet-GNN iniciado. Guardando en: {log_file_path}")
    logger.info(f"--- Iniciando Experimento STG-AE con Wavelets ---")
    logger.info(f"Directorio de datos: {data_directory}")
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info(f"Hiperparámetros (hp): {hp}")

    # --- Constantes ---
    num_expected_nodes = 5
    wavelet_name = hp.get('wavelet_name', 'db4')
    wavelet_level = hp.get('wavelet_level', 5)
    # Número de features esperado = 1 (original) + wavelet_level (detalles) + 1 (aprox)
    num_expected_features = 1 + wavelet_level + 1

    logger.info(f"Configuración Wavelet: Name='{wavelet_name}', Level={wavelet_level} -> Features={num_expected_features}")


    # --- Carga de Datos Crudos ---
    logger.info("Cargando datos crudos...")
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_directory}")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    files_processed = 0
    files_skipped = 0

    for f_path in tqdm(all_files, desc="Cargando archivos"):
        # (Misma lógica de carga que en train_no_gnn.py)
        try:
            filename = os.path.basename(f_path)
            sid_str = filename.split('_')[0]
            sid = int(sid_str)
            if sid in sensor_data_raw:
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[1], engine='python').values
                if data is not None and data.size > 0:
                    sensor_data_raw[sid].append(data)
                    files_processed += 1
                else: files_skipped += 1
            else: files_skipped += 1
        except Exception as e:
            logger.error(f"Error cargando {f_path}: {e}. Omitiendo.")
            files_skipped += 1

    logger.info(f"Carga inicial: {files_processed} procesados, {files_skipped} omitidos.")

    # Concatenar y encontrar longitud mínima
    sensor_data_concat = {}
    min_len_raw = float('inf')
    sensors_with_data = []
    for sid, data_list in sensor_data_raw.items():
        if data_list:
            # (Misma lógica de concatenación que en train_no_gnn.py)
            data_list_2d = [d.reshape(-1, 1) for d in data_list if d.size > 0]
            if not data_list_2d: continue
            try:
                concatenated_data = np.concatenate(data_list_2d, axis=0)
                sensor_data_concat[sid] = concatenated_data.squeeze() # Necesitamos 1D para DWT
                min_len_raw = min(min_len_raw, len(concatenated_data))
                sensors_with_data.append(sid)
            except Exception as e:
                 logger.error(f"Error concatenando sensor {sid}: {e}")

    if len(sensors_with_data) != num_expected_nodes:
        logger.error(f"Faltan datos para algunos sensores esperados: {set(range(1, 6)) - set(sensors_with_data)}. Abortando.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    target_len = min_len_raw # Usar la longitud mínima como objetivo
    logger.info(f"Datos crudos concatenados. Longitud mínima/objetivo: {target_len}")

    # --- Aplicar Wavelet Features ---
    logger.info("Aplicando DWT y reconstrucción de bandas...")
    sensor_data_features = {}
    feature_generation_successful = True
    for sid in tqdm(sensors_with_data, desc="Generando Features Wavelet"):
        signal_1d = sensor_data_concat[sid]
        features_2d = apply_dwt_features(signal_1d, wavelet=wavelet_name, level=wavelet_level, target_len=target_len)
        if features_2d is None or features_2d.shape[1] != num_expected_features:
             logger.error(f"Error generando features para sensor {sid}. Shape obtenido: {features_2d.shape if features_2d is not None else 'None'}. Esperado: ({target_len}, {num_expected_features}).")
             feature_generation_successful = False
             break # Salir si falla para un sensor
        sensor_data_features[sid] = features_2d # Shape: (target_len, num_expected_features)

    if not feature_generation_successful:
        logger.error("Falló la generación de características Wavelet. Abortando.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    logger.info(f"Features Wavelet generadas. Shape por sensor: ({target_len}, {num_expected_features})")

    # Liberar memoria de datos crudos si es posible
    del sensor_data_raw, sensor_data_concat
    import gc
    gc.collect()


    # --- Escalado (Aplicado a todas las features) ---
    logger.info("Escalando todas las características...")
    scaler = StandardScaler()

    # Ajustar el scaler: necesita datos 2D (samples, features)
    # Concatenamos todos los datos de todos los sensores temporalmente
    all_features_flat = np.concatenate([data for data in sensor_data_features.values()], axis=0) # Shape: (N_sensors * target_len, num_features)
    if all_features_flat.size == 0:
        logger.error("No hay datos para ajustar el scaler.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    scaler.fit(all_features_flat)
    logger.info("StandardScaler ajustado a todas las features.")

    # Transformar los datos de cada sensor
    sensor_data_scaled_features = {}
    for sid, data in sensor_data_features.items():
        try:
            sensor_data_scaled_features[sid] = scaler.transform(data)
        except Exception as e:
            logger.error(f"Error escalando features para sensor {sid}: {e}. Abortando.")
            if file_handler: logger.removeHandler(file_handler); file_handler.close()
            return

    # Liberar memoria
    del sensor_data_features, all_features_flat
    gc.collect()

    # --- Creación de Datasets (Ventanas con Features) ---
    logger.info("Creando datasets de ventanas con features Wavelet...")
    try:
        full_dataset = SpatioTemporalWaveletDataset( # Usar la clase de dataset adaptada
            sensor_data_scaled_features,
            hp['window_size'],
            hp['stride'],
            num_expected_nodes=num_expected_nodes
        )
    except ValueError as e:
        logger.error(f"Error creando el dataset de ventanas Wavelet: {e}")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    if len(full_dataset) == 0:
        logger.error("El dataset de ventanas Wavelet está vacío.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    actual_num_features = full_dataset.num_features
    if actual_num_features != num_expected_features:
        logger.warning(f"El número de features en el dataset ({actual_num_features}) no coincide con el esperado ({num_expected_features}). Se usará {actual_num_features}.")
        # Esto podría indicar un problema en apply_dwt_features o la lógica de apilado.

    # División Train/Val
    val_split = 0.15
    total_windows = len(full_dataset)
    val_len = int(val_split * total_windows)
    train_len = total_windows - val_len

    if train_len <= 0 or val_len <= 0:
         logger.error(f"Ventanas insuficientes para dividir: Train={train_len}, Val={val_len}, Total={total_windows}")
         if file_handler: logger.removeHandler(file_handler); file_handler.close()
         return

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    logger.info(f"Dataset Wavelet dividido: {len(train_dataset)} train, {len(val_dataset)} val.")

    # --- Bucle de Entrenamiento ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de entrenamiento: {device}")
    num_workers = 4 if os.name == 'posix' else 0
    logger.info(f"Workers DataLoader: {num_workers}")

    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    # Definir grafo
    edge_index = define_bridge_graph(num_nodes=num_expected_nodes).to(device)

    # Instanciar el modelo STG-AE (AHORA con `actual_num_features`)
    model = SpatioTemporalAutoencoder(
        num_nodes=num_expected_nodes, # Debe coincidir con define_bridge_graph y el dataset
        num_features=actual_num_features, # <--- Usar el número real de features
        window_size=hp['window_size'],
        gnn_hidden=hp.get('gnn_hidden', 32),
        gnn_out=hp.get('gnn_out', 16),
        rnn_hidden=hp.get('rnn_hidden', 64),
        rnn_layers=hp.get('rnn_layers', 2)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo STG-AE (Wavelet Features) creado. Features: {actual_num_features}, Parámetros: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'], weight_decay=hp.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=hp.get('scheduler_patience', 5),
                                                           factor=hp.get('scheduler_factor', 0.5),
                                                           verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model_wavelet_gnn.pth')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    logger.info("\n--- Iniciando Entrenamiento del STG-AE (Wavelet Features) ---")
    start_time_train = datetime.now()

    # --- Bucle Epoch (similar a train_no_gnn.py, pero llamando a model con edge_index) ---
    for epoch in range(hp['epochs']):
        epoch_start_time = datetime.now()
        model.train()
        avg_train_loss = 0.0
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Train]', leave=False)
        batch_count_train = 0

        for inputs, _ in progress_bar_train:
             # Shape esperado: [B, T, N, F_wavelet]
            if inputs.shape[1] != hp['window_size'] or inputs.shape[2] != num_expected_nodes or inputs.shape[3] != actual_num_features:
                 logger.warning(f"Batch train shape inesperado: {inputs.shape}. Esperado: (B, {hp['window_size']}, {num_expected_nodes}, {actual_num_features}). Omitiendo.")
                 continue

            inputs = inputs.to(device)
            optimizer.zero_grad()
            try:
                outputs = model(inputs, edge_index) # <--- Pasamos edge_index
                loss = criterion(outputs, inputs)
                if not torch.isfinite(loss):
                    logger.error(f"Loss NaN/Inf en train epoch {epoch+1}. Deteniendo.")
                    if file_handler: logger.removeHandler(file_handler); file_handler.close()
                    return

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clipping
                optimizer.step()

                current_loss = loss.item()
                avg_train_loss += current_loss
                batch_count_train += 1
                progress_bar_train.set_postfix({'Loss': f'{current_loss:.6f}'})
            except Exception as e:
                logger.error(f"Error train epoch {epoch+1}: {e}", exc_info=True)
                continue

        avg_train_loss /= batch_count_train if batch_count_train > 0 else 1
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # --- Validación ---
        model.eval()
        avg_val_loss = 0.0
        progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Val]', leave=False)
        batch_count_val = 0
        with torch.no_grad():
            for inputs, _ in progress_bar_val:
                if inputs.shape[1] != hp['window_size'] or inputs.shape[2] != num_expected_nodes or inputs.shape[3] != actual_num_features:
                    logger.warning(f"Batch val shape inesperado: {inputs.shape}. Omitiendo.")
                    continue
                inputs = inputs.to(device)
                try:
                    outputs = model(inputs, edge_index) # <--- Pasamos edge_index
                    loss = criterion(outputs, inputs)
                    if not torch.isfinite(loss):
                        logger.warning(f"Loss NaN/Inf en val epoch {epoch+1}. Omitiendo batch.")
                        continue
                    avg_val_loss += loss.item()
                    batch_count_val += 1
                    progress_bar_val.set_postfix({'Val Loss': f'{loss.item():.6f}'})
                except Exception as e:
                    logger.error(f"Error val epoch {epoch+1}: {e}", exc_info=True)
                    continue

        avg_val_loss /= batch_count_val if batch_count_val > 0 else 1
        history['val_loss'].append(avg_val_loss)

        epoch_duration = datetime.now() - epoch_start_time

        if not np.isfinite(avg_val_loss):
             logger.error(f"Epoch {epoch+1} -> Val Loss INVALID. Deteniendo.")
             if file_handler: logger.removeHandler(file_handler); file_handler.close()
             return

        logger.info(
            f"Epoch {epoch + 1}/{hp['epochs']} -> Lr: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Dur: {epoch_duration})"
        )

        # Scheduler y Early Stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                logger.info(f"   -> Nuevo mejor modelo (Wavelet GNN) guardado. Val Loss: {best_val_loss:.6f}")
            except Exception as e:
                 logger.error(f"   -> Error guardando mejor modelo: {e}")
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-7 or patience_counter >= hp['patience']:
            if current_lr < 1e-7:
                 logger.info(f"--- Parada Temprana: LR muy bajo ({current_lr:.2e}). ---")
            else:
                 logger.info(f"--- Parada Temprana: No mejora por {hp['patience']} épocas. ---")
            break

    # --- Fin del Entrenamiento ---
    end_time_train = datetime.now()
    total_training_duration = end_time_train - start_time_train
    logger.info(f"--- Entrenamiento Finalizado (Wavelet GNN) ---")
    logger.info(f"Duración total: {total_training_duration}")
    logger.info(f"Mejor Val Loss: {best_val_loss:.6f}")

    # --- Guardar Artefactos ---
    logger.info("Guardando artefactos finales...")
    # Scaler
    scaler_path = os.path.join(output_dir, 'scaler_wavelet_gnn.gz')
    try: joblib.dump(scaler, scaler_path)
    except Exception as e: logger.error(f"Error guardando scaler: {e}")
    else: logger.info(f"Scaler guardado en: {scaler_path}")

    # Hiperparámetros
    hp_path = os.path.join(output_dir, 'hyperparameters_wavelet_gnn.json')
    try:
        with open(hp_path, 'w', encoding='utf-8') as f:
            hp_save = hp.copy()
            hp_save['model_type'] = 'STG-AE (Wavelet Features)'
            hp_save['num_features'] = actual_num_features
            hp_save['total_params'] = total_params
            hp_save['best_val_loss'] = best_val_loss if np.isfinite(best_val_loss) else None
            hp_save['training_duration'] = str(total_training_duration)
            json.dump(hp_save, f, indent=4)
    except Exception as e: logger.error(f"Error guardando HPs: {e}")
    else: logger.info(f"HPs guardados en: {hp_path}")

    # Historial
    history_path = os.path.join(output_dir, 'loss_history_wavelet_gnn.json')
    try:
        history_safe = {k: [v if np.isfinite(v) else None for v in vals] for k, vals in history.items()}
        with open(history_path, 'w', encoding='utf-8') as f: json.dump(history_safe, f, indent=4)
    except Exception as e: logger.error(f"Error guardando historial: {e}")
    else: logger.info(f"Historial guardado en: {history_path}")

    # Curvas de Pérdida
    try:
        # (Misma lógica de ploteo que en train_no_gnn.py)
        epochs = list(range(1, len(history_safe['train_loss']) + 1))
        train_loss_plot = [l for l in history_safe['train_loss'] if l is not None]
        val_loss_plot = [l for l in history_safe['val_loss'] if l is not None]
        epochs_train = [epochs[i] for i, l in enumerate(history_safe['train_loss']) if l is not None]
        epochs_val = [epochs[i] for i, l in enumerate(history_safe['val_loss']) if l is not None]

        if train_loss_plot and val_loss_plot:
            plt.figure(figsize=(12, 7))
            plt.plot(epochs_train, train_loss_plot, label='Training Loss', marker='.', linestyle='-')
            plt.plot(epochs_val, val_loss_plot, label='Validation Loss', marker='.', linestyle='--')
            plt.title('Training & Validation Loss (STG-AE Wavelet)')
            plt.xlabel('Epochs'); plt.ylabel('MSE Loss')
            all_losses_plot = train_loss_plot + val_loss_plot
            min_loss_plot = min(all_losses_plot) if all_losses_plot else 0.01
            max_loss_plot = max(all_losses_plot) if all_losses_plot else 1.0
            if max_loss_plot / max(min_loss_plot, 1e-9) > 100: plt.yscale('log'); plt.ylabel('MSE Loss (Log Scale)'); plt.ylim(bottom=max(min_loss_plot * 0.8, 1e-9))
            else: plt.ylim(bottom=0)
            plt.legend(); plt.grid(True, linestyle=':')
            loss_curve_path = os.path.join(output_dir, 'loss_curve_wavelet_gnn.png')
            plt.savefig(loss_curve_path, dpi=300); plt.close()
            logger.info(f"Gráfico curvas de pérdida guardado en: {loss_curve_path}")
        else: logger.warning("No hay datos válidos para plotear curvas de pérdida.")
    except Exception as e: logger.error(f"Error generando gráfico curvas de pérdida: {e}")

    # --- Cerrar Handler ---
    if file_handler:
        logger.info("Cerrando archivo de log.")
        file_handler.close()
        logger.removeHandler(file_handler)

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == '__main__':
    # --- Rutas (Usa las mismas que el original) ---
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    base_output_dir = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet" # Directorio específico

    # --- Hiperparámetros ---
    HP = {
        # Dataset & Wavelet
        "window_size": 64,
        "stride": 32,
        "wavelet_name": "db4", # Daubechies 4
        "wavelet_level": 5,    # Nivel de descomposición
        # Modelo (Ajusta gnn_hidden/out y rnn_hidden si es necesario por el aumento de features)
        "gnn_hidden": 32, # Puede necesitar ajuste
        "gnn_out": 16,    # Puede necesitar ajuste
        "rnn_hidden": 64, # Puede necesitar ajuste
        "rnn_layers": 2,
        # Entrenamiento
        "epochs": 50,
        "batch_size": 32,      # Podría necesitar reducirse si la memoria GPU es limitada con más features
        "learning_rate": 0.001, # Empezar con el original, ajustar si es inestable
        "patience": 10,
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "weight_decay": 0 # Podría añadirse si hay overfitting
    }

    # --- Crear Directorio de Salida ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_directory = os.path.join(base_output_dir, f"run_wavelet_{HP['wavelet_name']}{HP['wavelet_level']}_{timestamp}")
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Resultados se guardarán en: {output_directory}")
    except OSError as e:
        print(f"Error creando directorio {output_directory}: {e}")
        sys.exit(1)

    # --- Validar Directorio de Datos ---
    if not os.path.isdir(data_folder_path):
        print(f"Error: Directorio de datos no encontrado: {data_folder_path}")
        sys.exit(1)

    # --- Ejecutar ---
    try:
        run_experiment_wavelet_gnn(data_folder_path, output_directory, HP)
    except Exception as e:
        if logger.hasHandlers(): logger.critical(f"Error fatal: {e}", exc_info=True)
        else: print(f"Error fatal: {e}")
        sys.exit(1)
