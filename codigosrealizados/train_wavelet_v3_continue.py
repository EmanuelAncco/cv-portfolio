# -*- coding: utf-8 -*-
"""
resume_train_wavelet_gnn.py

REANUDA un entrenamiento existente de un Autoencoder Gráfico Espacio-Temporal (STG-AE)
cargando el modelo, el scaler y el historial de una ejecución anterior.
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
import pywt  # <--- Importar PyWavelets
import gc  # Para liberar memoria

# Asegúrate de tener torch_geometric instalado: pip install torch_geometric
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("Error: torch_geometric no está instalado. Por favor, instálalo con 'pip install torch_geometric'")
    sys.exit(1)

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()  # Logger Raíz
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler)  # Limpiar handlers
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


# FileHandler se añade en run_experiment

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
    # Validación crucial: Asegura que los índices no excedan el número de nodos esperado
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

        # Validar datos y encontrar longitud mínima
        valid_data_dict = {}
        min_len = float('inf')
        expected_num_features = -1

        for sid, data in data_dict_features.items():
            # Chequeos más robustos
            if data is not None and isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] >= window_size:
                current_num_features = data.shape[1]
                if expected_num_features == -1:
                    expected_num_features = current_num_features
                elif current_num_features != expected_num_features:
                    local_logger.error(
                        f"Inconsistencia en número de features. Sensor {sid} tiene {current_num_features}, se esperaban {expected_num_features}. Omitiendo.")
                    continue  # Saltar este sensor

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

        # Asegurar que tenemos datos para todos los nodos esperados y obtener num_features real
        processed_data_list = []
        actual_node_ids = []
        self.num_features = 0
        for sid in range(1, self.num_expected_nodes + 1):
            if sid in valid_data_dict:
                data_node = valid_data_dict[sid][:min_len]  # Truncar a la longitud mínima común
                processed_data_list.append(data_node)
                actual_node_ids.append(sid)
                if self.num_features == 0:  # Tomar num_features del primer nodo válido
                    self.num_features = data_node.shape[1]
            else:
                # Si falta un sensor, abortamos porque la arquitectura espera un número fijo
                local_logger.error(
                    f"Faltan datos procesados para el sensor esperado {sid}. La arquitectura requiere datos de todos los nodos. Abortando.")
                raise ValueError(f"Faltan datos para el sensor esperado {sid}.")

        if not processed_data_list:
            local_logger.error("La lista de datos procesados está vacía (posiblemente por fallo en carga o filtrado).")
            raise ValueError("La lista de datos procesados está vacía.")

        # Apilar a lo largo de una nueva dimensión (axis=1 para nodos)
        # Shape esperado: (min_len, num_nodes, num_features)
        try:
            # Todos los arrays en processed_data_list ya tienen shape (min_len, num_features)
            self.data = np.stack(processed_data_list, axis=1)
        except ValueError as e:
            shapes_str = ", ".join([str(d.shape) for d in processed_data_list])
            local_logger.error(
                f"Error al apilar datos (np.stack axis=1). Shapes individuales: [{shapes_str}]. Error: {e}")
            raise e

        self.num_nodes = self.data.shape[1]
        local_logger.info(f"Datos apilados con shape final: {self.data.shape}. Sensores usados: {actual_node_ids}")

        # Verificación final de consistencia
        if self.num_nodes != self.num_expected_nodes:
            local_logger.error(
                f"Inconsistencia crítica: Nodos procesados ({self.num_nodes}) != Nodos esperados ({self.num_expected_nodes}) después del apilado.")
            raise RuntimeError("Error interno en la creación del dataset: Conteo de nodos inconsistente.")
        if self.num_features != expected_num_features and expected_num_features != -1:
            local_logger.error(
                f"Inconsistencia crítica: Features procesadas ({self.num_features}) != Features esperadas ({expected_num_features}).")
            raise RuntimeError("Error interno en la creación del dataset: Conteo de features inconsistente.")

        # Calcular número de muestras
        self.n_samples = max(0, (self.data.shape[0] - window_size) // stride + 1)  # Usar self.data.shape[0] (min_len)
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
        if start < 0 or end > self.data.shape[0]:  # Usar self.data.shape[0]
            local_logger = logging.getLogger(self.__class__.__name__)
            local_logger.error(
                f"Índice {idx} genera rango [{start}, {end}) fuera de límites [{0}, {self.data.shape[0]}]. Stride={self.stride}, WinSize={self.window_size}, N_Samples={self.n_samples}")
            raise IndexError(f"Índice {idx} fuera de rango.")

        window = self.data[start:end]  # Shape: (window_size, num_nodes, num_features)
        return torch.FloatTensor(window), torch.FloatTensor(window)


# --- ARQUITECTURA DEL GNN AUTOENCODER ---
class GNNLayer(nn.Module):
    """Bloque GCN."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)  # Usar LeakyReLU

    def forward(self, x, edge_index):
        edge_index = edge_index.to(x.device)  # Mover edge_index al dispositivo de x
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
        # self.relu = nn.LeakyReLU(0.01) # El relu está dentro de GNNLayer

    def forward(self, x, edge_index):
        batch_size, T_actual, N_actual, F_actual = x.shape
        # Validación de shape dentro de forward
        if T_actual != self.window_size or N_actual != self.num_nodes or F_actual != self.num_features:
            logger.warning(
                f"Unexpected input shape in forward: {x.shape}. Expected T={self.window_size}, N={self.num_nodes}, F={self.num_features}. Trying to proceed...")
            # Considerar si se debe lanzar un error aquí en lugar de solo advertir

        # 1. Preparar datos para GNN Encoder: [B, T, N, F] -> [B*T, N, F]
        x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)
        edge_index = edge_index.to(x.device)  # Asegurar dispositivo

        # 2. GNN Encoder
        try:
            gnn_encoded = self.gnn_encoder(x_reshaped, edge_index)  # Shape: [B*T, N, gnn_out]
        except Exception as e:
            logger.error(f"Error en GNN Encoder. Input shape: {x_reshaped.shape}. Error: {e}", exc_info=True)
            raise e

        # 3. Preparar datos para RNN Encoder: [B*T, N, gnn_out] -> [B, T, N * gnn_out]
        try:
            gnn_encoded_view = gnn_encoded.reshape(batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
            rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)  # Aplanar N y gnn_out
        except Exception as e:
            logger.error(f"Error en reshape pre-RNN Encoder. GNN Encoded shape: {gnn_encoded.shape}. Error: {e}",
                         exc_info=True)
            raise e

        # 4. RNN Encoder
        try:
            # output (no usado), h_n (estado final)
            _, h_n = self.rnn_encoder(rnn_input)  # h_n shape: [num_layers, B, rnn_hidden]
        except Exception as e:
            logger.error(f"Error en RNN Encoder. Input shape: {rnn_input.shape}. Error: {e}", exc_info=True)
            raise e

        # 5. RNN Decoder
        try:
            # Usar el último estado oculto como entrada para cada paso de tiempo del decoder
            latent_vector = h_n[-1].unsqueeze(1).repeat(1, T_actual, 1)  # Shape: [B, T, rnn_hidden]
            rnn_decoded, _ = self.rnn_decoder(latent_vector)  # Shape: [B, T, N*gnn_hidden]
        except Exception as e:
            logger.error(
                f"Error en RNN Decoder. Latent shape: {latent_vector.shape if 'latent_vector' in locals() else 'N/A'}. Error: {e}",
                exc_info=True)
            raise e

        # 6. Preparar datos para GNN Decoder: [B, T, N*gnn_hidden] -> [B*T, N, gnn_hidden]
        try:
            gnn_input_decoder = rnn_decoded.reshape(batch_size * T_actual, N_actual, self.gnn_hidden_dim)
        except Exception as e:
            logger.error(f"Error en reshape pre-GNN Decoder. RNN Decoded shape: {rnn_decoded.shape}. Error: {e}",
                         exc_info=True)
            raise e

        # 7. GNN Decoder
        try:
            reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index)  # Shape: [B*T, N, num_features]
        except Exception as e:
            logger.error(f"Error en GNN Decoder. Input shape: {gnn_input_decoder.shape}. Error: {e}", exc_info=True)
            raise e

        # 8. Reshape final: [B*T, N, F] -> [B, T, N, F]
        try:
            reconstructed_x = reconstructed_frames.reshape(batch_size, T_actual, N_actual, F_actual)
        except Exception as e:
            logger.error(
                f"Error en reshape final. Reconstructed frames shape: {reconstructed_frames.shape}. Target: ({batch_size}, {T_actual}, {N_actual}, {F_actual}). Error: {e}",
                exc_info=True)
            raise e

        return reconstructed_x


# --- FUNCIONES AUXILIARES WAVELET ---
def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    """Aplica DWT multinivel y reconstruye bandas."""
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 1 or len(signal) == 0:
        logger.warning("apply_dwt_features: Señal de entrada inválida.")
        return None  # Devolver None para indicar fallo

    original_len = len(signal)
    if target_len is None:
        target_len = original_len

    try:
        # Descomposición
        coeffs = pywt.wavedec(signal, wavelet, level=level)

        # Reconstrucción de bandas individuales
        reconstructed_bands = []

        # Reconstruir cada nivel de detalle D_i
        for i in range(level, 0, -1):
            # Crear lista de coeficientes: solo el nivel i-ésimo de detalle activo
            # Índices en coeffs: [A_level, D_level, D_(level-1), ..., D1]
            detail_level_index = level - i + 1
            if detail_level_index < 0 or detail_level_index >= len(coeffs):
                logger.error(
                    f"Índice de nivel de detalle {detail_level_index} fuera de rango para coeffs (len={len(coeffs)}).")
                return None  # Fallo crítico

            detail_coeffs_list = [np.zeros_like(c) for c in coeffs]  # Lista de ceros
            detail_coeffs_list[detail_level_index] = coeffs[detail_level_index]  # Activar solo el detalle i
            # Reconstruir
            rec_d = pywt.waverec(detail_coeffs_list, wavelet)
            # Ajustar longitud (padding o truncado)
            rec_d_adj = adjust_signal_length(rec_d, target_len)
            reconstructed_bands.append(rec_d_adj)  # Se añaden D_level, D_(level-1), ..., D1

        # Reconstruir la última aproximación A_level
        approx_coeffs_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]  # Solo A_level activo
        rec_a = pywt.waverec(approx_coeffs_list, wavelet)
        rec_a_adj = adjust_signal_length(rec_a, target_len)
        # reconstructed_bands ahora contiene [D_level, ..., D1]
        # Añadimos A_level al final de esta lista para mantener el orden A_level, D_level, ..., D1
        reconstructed_bands.append(rec_a_adj)

        # Apilar la señal original y las bandas reconstruidas
        # Orden final deseado: [Original, A_level, D_level, ..., D1]
        original_adjusted = adjust_signal_length(signal, target_len)

        # reconstructed_bands está [D_level, ..., D1, A_level]. Invertimos para [A_level, D1, ..., D_level]
        ordered_bands_rev = reconstructed_bands[::-1]

        # Combinar: Original + [A_level, D1, ..., D_level]
        all_bands = [original_adjusted] + ordered_bands_rev

        features = np.stack(all_bands, axis=-1)  # Apilar en la última dimensión

        # Verificar forma final (Original + 1 Aprox + Level Detalles)
        expected_feature_count = 1 + 1 + level
        if features.shape != (target_len, expected_feature_count):
            logger.warning(
                f"Shape inesperado en features wavelet: {features.shape}. Esperado: ({target_len}, {expected_feature_count})")
            # Considerar devolver None o lanzar error si el shape es crucial

        return features

    except ValueError as ve:  # Errores comunes de PyWavelets
        logger.error(f"Error de valor aplicando DWT (posiblemente nivel muy alto para longitud {original_len}): {ve}",
                     exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error inesperado aplicando DWT a señal de longitud {original_len}: {e}", exc_info=True)
        # Devolver solo la señal original ajustada si falla DWT? O mejor None? Devolvemos None.
        # return adjust_signal_length(signal, target_len)[:, np.newaxis]
        return None


def adjust_signal_length(signal, target_len):
    """Ajusta la longitud de una señal 1D (padding o truncado)."""
    current_len = len(signal)
    if current_len == target_len:
        return signal
    elif current_len > target_len:
        # Truncar (desde el inicio)
        return signal[:target_len]
    else:
        # Padding (con ceros al final)
        padding = np.zeros(target_len - current_len)
        return np.concatenate((signal, padding))


# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO ---
def run_experiment_wavelet_gnn(data_directory, output_dir, hp, resume_run_path=None):
    """
    Función principal para entrenar o REANUDAR el modelo STG-AE
    con características Wavelet.
    """
    # --- Configuración del logging de archivo ---
    log_file_path = os.path.join(output_dir, 'training_log_wavelet_RESUME.txt')  # Nuevo log
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)  # Añadir handler específico de esta ejecución
    logger.info(f"Logging de entrenamiento Wavelet-GNN iniciado. Guardando en: {log_file_path}")
    logger.info(f"Directorio de datos: {data_directory}")
    logger.info(f"Directorio de salida: {output_dir}")

    # --- LÓGICA DE REANUDACIÓN ---
    hp_original = {}
    if resume_run_path:
        logger.info(f"--- REANUDANDO ENTRENAMIENTO DESDE: {resume_run_path} ---")
        try:
            # Cargar HPs originales para arquitectura y datos
            hp_original_path = os.path.join(resume_run_path, 'hyperparameters_wavelet_gnn.json')
            with open(hp_original_path, 'r') as f:
                hp_original = json.load(f)
            logger.info("Hiperparámetros originales cargados.")

            # Sobrescribir HPs de datos/arquitectura con los originales
            # El diccionario 'hp' de entrada ahora solo define HPs de *entrenamiento*
            hp_to_override = ['window_size', 'stride', 'wavelet_name', 'wavelet_level',
                              'gnn_hidden', 'gnn_out', 'rnn_hidden', 'rnn_layers',
                              'num_features']  # num_features también es crucial

            # Combinar HPs: los originales mandan en arquitectura/datos, los nuevos en entrenamiento
            hp_combined = hp_original.copy()
            hp_combined.update(hp)  # Los HPs nuevos (lr, epochs, etc.) sobrescriben
            hp = hp_combined  # Usar este HP combinado de ahora en adelante

            # Cargar historial y estado anterior
            history_path_original = os.path.join(resume_run_path, 'loss_history_wavelet_gnn.json')
            with open(history_path_original, 'r') as f:
                history = json.load(f)

            best_val_loss = hp_original.get('best_val_loss', float('inf'))
            # Asegurar que best_val_loss no sea None (caso guardado en JSON)
            if best_val_loss is None: best_val_loss = float('inf')

            start_epoch = len(history['train_loss'])  # La siguiente época es len()

            # Rutas a artefactos
            scaler_path_original = os.path.join(resume_run_path, 'scaler_wavelet_gnn.gz')
            model_path_original = os.path.join(resume_run_path, 'best_model_wavelet_gnn.pth')

            logger.info(f"HPs originales cargados. Reanudando desde epoch {start_epoch + 1}.")
            logger.info(f"Mejor Val Loss anterior: {best_val_loss:.6f}")
            logger.info(f"Nuevos HPs de entrenamiento: Epochs Adicionales={hp['epochs']}, LR={hp['learning_rate']}")

        except FileNotFoundError as e:
            logger.error(f"Error: No se encontró un archivo requerido en {resume_run_path}: {e}. Abortando.")
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return
        except Exception as e:
            logger.error(f"Error cargando artefactos de reanudación: {e}", exc_info=True)
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return
    else:
        logger.info("--- INICIANDO NUEVO ENTRENAMIENTO (DESDE CERO) ---")
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        start_epoch = 0
        best_val_loss = float('inf')
        scaler_path_original = None
        model_path_original = None
        # hp ya está_correcto

    logger.info(f"Hiperparámetros (finales) para esta ejecución: {hp}")

    # --- Constantes ---
    num_expected_nodes = 5
    wavelet_name = hp.get('wavelet_name', 'db4')
    wavelet_level = hp.get('wavelet_level', 5)

    # num_expected_features: Si reanudamos, debe venir de hp_original
    if resume_run_path:
        num_expected_features = hp.get('num_features')  # Ya debe estar en hp
        if not num_expected_features:
            logger.error("Error: 'num_features' no se encontró en los HPs cargados. Abortando.")
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return
    else:
        # 1 (original) + 1 (aprox) + wavelet_level (detalles)
        num_expected_features = 1 + 1 + wavelet_level

    logger.info(
        f"Configuración Wavelet: Name='{wavelet_name}', Level={wavelet_level} -> Features esperadas={num_expected_features}")

    # --- Carga de Datos Crudos ---
    # (Esta sección no cambia, siempre necesitamos cargar los datos crudos)
    logger.info("Cargando datos crudos...")
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_directory}")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return  # Salir si no hay archivos

    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    files_processed = 0
    files_skipped = 0
    for f_path in tqdm(all_files, desc="Cargando archivos"):
        try:
            filename = os.path.basename(f_path)
            sid_str = filename.split('_')[0]
            sid = int(sid_str)
            if sid in sensor_data_raw:
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[1], engine='python',
                                   on_bad_lines='warn').values
                if data is not None and data.size > 0:
                    sensor_data_raw[sid].append(data)
                    files_processed += 1
                else:
                    logger.warning(f"Archivo vacío o ilegible omitido: {filename}")
                    files_skipped += 1
            else:
                logger.warning(f"ID '{sid}' de '{filename}' fuera de rango [1-{num_expected_nodes}]. Omitiendo.")
                files_skipped += 1
        except (ValueError, IndexError):
            logger.warning(f"No se pudo extraer ID numérico de '{filename}'. Omitiendo.")
            files_skipped += 1
        except pd.errors.EmptyDataError:
            logger.warning(f"Archivo vacío (Pandas): '{filename}'. Omitiendo.")
            files_skipped += 1
        except Exception as e:
            logger.error(f"Error inesperado cargando '{filename}': {e}. Omitiendo.", exc_info=False)
            files_skipped += 1
    logger.info(f"Carga inicial completa. Archivos procesados: {files_processed}, Omitidos: {files_skipped}")

    # Concatenar y encontrar longitud mínima
    sensor_data_concat = {}
    min_len_raw = float('inf')
    sensors_with_data = []
    for sid, data_list in sensor_data_raw.items():
        if data_list:
            data_list_valid = [d for d in data_list if d is not None and d.size > 0]
            if not data_list_valid:
                logger.warning(f"Sensor {sid}: No hay arrays válidos para concatenar.")
                continue
            data_list_2d = [d.reshape(-1, 1) if d.ndim == 1 else d for d in data_list_valid]
            if any(d.shape[1] != 1 for d in data_list_2d):
                logger.error(
                    f"Sensor {sid}: Inconsistencia en número de columnas ({[d.shape[1] for d in data_list_2d]}). Omitiendo sensor.")
                continue
            try:
                concatenated_data = np.concatenate(data_list_2d, axis=0)
                if concatenated_data.size > 0:
                    sensor_data_concat[sid] = concatenated_data.squeeze()  # Necesitamos 1D para DWT
                    min_len_raw = min(min_len_raw, len(concatenated_data))
                    sensors_with_data.append(sid)
                    logger.info(f"Sensor {sid}: {len(data_list_2d)} archivos concatenados -> {concatenated_data.shape}")
                else:
                    logger.warning(f"Sensor {sid}: Concatenación resultó en array vacío.")
            except ValueError as e:
                logger.error(f"Error concatenando datos para sensor {sid}: {e}. Omitiendo sensor.")
            except Exception as e:
                logger.error(f"Error inesperado concatenando sensor {sid}: {e}. Omitiendo sensor.")
        else:
            logger.warning(f"No se cargaron datos válidos para el sensor {sid}.")

    if len(sensors_with_data) != num_expected_nodes:
        missing_sensors = set(range(1, num_expected_nodes + 1)) - set(sensors_with_data)
        logger.error(f"Faltan datos concatenados para los sensores: {missing_sensors}. No se puede continuar.")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return
    logger.info(f"Datos crudos concatenados para sensores: {sensors_with_data}")

    if min_len_raw == float('inf') or min_len_raw < hp['window_size']:
        len_val = min_len_raw if min_len_raw != float('inf') else 'N/A'
        logger.error(
            f"Longitud mínima post-concatenación ({len_val}) es inválida o insuficiente para window_size ({hp['window_size']}).")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return
    target_len = min_len_raw
    logger.info(f"Longitud mínima/objetivo para DWT y truncado: {target_len}")

    # --- Aplicar Wavelet Features ---
    logger.info("Aplicando DWT y reconstrucción de bandas...")
    sensor_data_features = {}
    feature_generation_successful = True
    actual_num_features_generated = 0
    for sid in tqdm(sensors_with_data, desc="Generando Features Wavelet"):
        signal_1d = sensor_data_concat[sid]
        features_2d = apply_dwt_features(signal_1d, wavelet=wavelet_name, level=wavelet_level, target_len=target_len)

        if features_2d is None:
            logger.error(f"Error generando features para sensor {sid}. Ver logs anteriores.")
            feature_generation_successful = False
            break

        current_features = features_2d.shape[1]
        if actual_num_features_generated == 0:
            actual_num_features_generated = current_features
        elif current_features != actual_num_features_generated:
            logger.error(
                f"Inconsistencia en número de features generadas. Sensor {sid} tiene {current_features}, se esperaban {actual_num_features_generated}.")
            feature_generation_successful = False
            break

        sensor_data_features[sid] = features_2d

    if not feature_generation_successful:
        logger.error("Falló la generación de características Wavelet para uno o más sensores. Abortando.")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    # Verificar si el número generado coincide con el esperado
    if actual_num_features_generated != num_expected_features:
        logger.warning(
            f"El número de features wavelet generadas ({actual_num_features_generated}) no coincide con el esperado ({num_expected_features}). Se usará el valor generado: {actual_num_features_generated}.")
        num_expected_features = actual_num_features_generated  # Usar el real
        hp['num_features'] = actual_num_features_generated  # Guardar el real

    logger.info(
        f"Features Wavelet generadas exitosamente. Shape por sensor: ({target_len}, {actual_num_features_generated})")

    del sensor_data_raw, sensor_data_concat
    gc.collect()

    # --- Escalado (Aplicado a todas las features) ---
    if resume_run_path and scaler_path_original:
        # --- CARGAR SCALER ---
        try:
            logger.info(f"Cargando scaler existente desde: {scaler_path_original}")
            scaler = joblib.load(scaler_path_original)
            if not hasattr(scaler, 'transform'):
                raise ValueError("El objeto cargado no es un scaler válido (no tiene 'transform').")
            logger.info("Scaler cargado exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando scaler desde {scaler_path_original}: {e}. Abortando.", exc_info=True)
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return
    else:
        # --- AJUSTAR NUEVO SCALER ---
        logger.info("Ajustando nuevo StandardScaler a todas las características...")
        scaler = StandardScaler()
        all_features_flat = np.concatenate([data for data in sensor_data_features.values()], axis=0)
        if all_features_flat.size == 0:
            logger.error("No hay datos válidos (post-DWT) para ajustar el scaler.")
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return

        try:
            scaler.fit(all_features_flat)
            logger.info("StandardScaler ajustado a todas las features.")
        except Exception as e:
            logger.error(f"Error ajustando StandardScaler: {e}", exc_info=True)
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return
        del all_features_flat  # Liberar memoria

    # Transformar los datos de cada sensor (siempre necesario)
    logger.info("Transformando datos con el scaler...")
    sensor_data_scaled_features = {}
    scaling_successful_sensors = []
    for sid in sensors_with_data:
        try:
            data_to_scale = sensor_data_features.get(sid)
            if data_to_scale is None or data_to_scale.shape[1] != actual_num_features_generated:
                logger.error(
                    f"Datos inconsistentes para escalar en sensor {sid}. Shape: {data_to_scale.shape if data_to_scale is not None else 'None'}. Esperado: (*, {actual_num_features_generated})")
                continue

            scaled_data = scaler.transform(data_to_scale)
            sensor_data_scaled_features[sid] = scaled_data
            scaling_successful_sensors.append(sid)
        except Exception as e:
            logger.error(f"Error escalando features para sensor {sid}: {e}. Omitiendo sensor.")

    if len(scaling_successful_sensors) != num_expected_nodes:
        missing_scaled = set(range(1, num_expected_nodes + 1)) - set(scaling_successful_sensors)
        logger.error(f"Falló el escalado para los sensores: {missing_scaled}. No se puede continuar.")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    logger.info("Escalado de features completado.")
    del sensor_data_features
    gc.collect()

    # --- Creación de Datasets (Ventanas con Features) ---
    logger.info("Creando datasets de ventanas con features Wavelet...")
    try:
        full_dataset = SpatioTemporalWaveletDataset(
            sensor_data_scaled_features,
            hp['window_size'],
            hp['stride'],
            num_expected_nodes=num_expected_nodes
        )
    except ValueError as e:
        logger.error(f"Error creando el dataset de ventanas Wavelet: {e}")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return
    except Exception as e:
        logger.error(f"Error inesperado creando dataset: {e}", exc_info=True)
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    if len(full_dataset) == 0:
        logger.error(
            "El dataset de ventanas Wavelet está vacío (n_samples=0). Verifique window_size, stride y longitud de datos.")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    # Usar el número de features del dataset, que ya fue validado
    actual_num_features = full_dataset.num_features
    # Asegurarse de que coincida con el esperado (especialmente al reanudar)
    if actual_num_features != num_expected_features:
        logger.error(
            f"Inconsistencia crítica: Features del dataset ({actual_num_features}) != Features esperadas/cargadas ({num_expected_features}).")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    # División Train/Val
    val_split = 0.15
    total_windows = len(full_dataset)
    val_len = int(val_split * total_windows)
    train_len = total_windows - val_len

    if train_len <= 0 or val_len <= 0:
        logger.error(
            f"No hay suficientes ventanas para dividir en entrenamiento ({train_len}) y validación ({val_len}) después de crear ventanas. Total: {total_windows}")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    try:
        # USAR LA MISMA SEMILLA ES CRUCIAL PARA LA REANUDACIÓN
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len],
                                                  generator=torch.Generator().manual_seed(42))
        logger.info(f"Dataset Wavelet dividido (seed 42): {len(train_dataset)} train, {len(val_dataset)} val.")
    except Exception as e:
        logger.error(f"Error dividiendo el dataset: {e}", exc_info=True)
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    # --- Bucle de Entrenamiento ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de entrenamiento: {device}")
    num_workers = 4 if os.name == 'posix' else 0
    logger.info(f"Workers DataLoader: {num_workers}")

    batch_size = hp['batch_size']
    max_retries = 3
    current_retry = 0
    train_loader, val_loader = None, None

    while current_retry <= max_retries:
        try:
            logger.info(f"Intentando crear DataLoaders con Batch Size: {batch_size}")
            pin_memory_flag = True if device.type == 'cuda' else False
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      pin_memory=pin_memory_flag, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                    pin_memory=pin_memory_flag)
            logger.info("DataLoaders creados exitosamente.")
            break
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and batch_size > 1:
                logger.warning(f"CUDA out of memory con batch size {batch_size}. Reduciendo a {batch_size // 2}.")
                batch_size //= 2
                current_retry += 1
                if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
            else:
                logger.error(f"Error creando DataLoaders (no es OOM o batch_size=1): {e}", exc_info=True)
                if file_handler: file_handler.close(); logger.removeHandler(file_handler)
                return
        except Exception as e:
            logger.error(f"Error inesperado creando DataLoaders: {e}", exc_info=True)
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return

    if train_loader is None or val_loader is None:
        logger.error(f"No se pudieron crear los DataLoaders después de {max_retries} intentos. Abortando.")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    hp['batch_size'] = batch_size  # Guardar el batch size final

    try:
        edge_index = define_bridge_graph(num_nodes=num_expected_nodes).to(device)
    except ValueError as e:
        logger.error(f"Error creando/validando edge_index: {e}")
        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
        return

    # *** Instanciar modelo ***
    # (Los HPs de arquitectura deben ser los correctos, cargados de hp_original si se reanuda)
    model = SpatioTemporalAutoencoder(
        num_nodes=num_expected_nodes,
        num_features=actual_num_features,
        window_size=hp['window_size'],
        gnn_hidden=hp.get('gnn_hidden', 128),
        gnn_out=hp.get('gnn_out', 64),
        rnn_hidden=hp.get('rnn_hidden', 256),
        rnn_layers=hp.get('rnn_layers', 2)
    ).to(device)

    # --- CARGAR PESOS DEL MODELO (si se reanuda) ---
    if resume_run_path and model_path_original:
        try:
            logger.info(f"Cargando pesos del modelo desde: {model_path_original}")
            model.load_state_dict(torch.load(model_path_original, map_location=device))
            logger.info("Pesos del modelo cargados exitosamente.")
        except FileNotFoundError:
            logger.error(f"Error: No se encontró el archivo de modelo en {model_path_original}. Abortando.")
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return
        except Exception as e:
            logger.error(f"Error cargando state_dict del modelo: {e}. Verifique que la arquitectura coincida.",
                         exc_info=True)
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Modelo STG-AE (Wavelet Features - Cap Max) listo. Features: {actual_num_features}, Parámetros: {total_params:,}, Batch Size Final: {batch_size}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=hp.get('scheduler_patience', 5),
                                                           factor=hp.get('scheduler_factor', 0.5),
                                                           verbose=True)

    # best_val_loss, history, y start_epoch ya están definidos
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model_wavelet_gnn.pth')  # Guardar en el nuevo dir

    # Calcular épocas totales
    total_epochs = start_epoch + hp['epochs']  # p.ej. 50 (start) + 50 (nuevas) = 100

    logger.info(
        f"\n--- Iniciando Entrenamiento... Total Epochs a correr: {hp['epochs']} (desde {start_epoch + 1} hasta {total_epochs}) ---")
    start_time_train = datetime.now()

    # --- Bucle Epoch ---
    # El rango ahora empieza en start_epoch (p.ej. 50) y va hasta total_epochs (p.ej. 100)
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = datetime.now()
        model.train()
        avg_train_loss = 0.0
        batch_count_train = 0
        # Mostrar la época real (p.ej. 51/100)
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Train]', leave=False,
                                  unit="batch")

        for inputs, _ in progress_bar_train:
            if inputs.shape[1] != hp['window_size'] or inputs.shape[2] != num_expected_nodes or inputs.shape[
                3] != actual_num_features:
                logger.warning(
                    f"Skip train batch shape {inputs.shape}. Expected (B, {hp['window_size']}, {num_expected_nodes}, {actual_num_features})")
                continue

            inputs = inputs.to(device)
            optimizer.zero_grad(set_to_none=True)

            try:
                outputs = model(inputs, edge_index)
                loss = criterion(outputs, inputs)

                if not torch.isfinite(loss):
                    logger.error(
                        f"Loss NaN/Inf detectada en train epoch {epoch + 1}, batch {batch_count_train + 1}. Deteniendo entrenamiento.")
                    if file_handler: file_handler.close(); logger.removeHandler(file_handler)
                    return  # Detener

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                current_loss = loss.item()
                avg_train_loss += current_loss
                batch_count_train += 1
                progress_bar_train.set_postfix({'Loss': f'{current_loss:.6f}'})

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(
                        f"CUDA out of memory durante train epoch {epoch + 1}, batch {batch_count_train + 1}. Batch Size={batch_size}. Intente reducir batch_size aún más.",
                        exc_info=False)
                    if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
                    if file_handler: file_handler.close(); logger.removeHandler(file_handler)
                    return
                else:
                    logger.error(f"Error Runtime en train epoch {epoch + 1}, batch {batch_count_train + 1}: {e}",
                                 exc_info=True)
                    continue
            except Exception as e:
                logger.error(f"Error inesperado en train epoch {epoch + 1}, batch {batch_count_train + 1}: {e}",
                             exc_info=True)
                continue

        if batch_count_train == 0:
            logger.warning(f"Epoch {epoch + 1}: No se procesaron batches de entrenamiento.")
            avg_train_loss = 0.0
        else:
            avg_train_loss /= batch_count_train
        history['train_loss'].append(avg_train_loss if batch_count_train > 0 else None)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # --- Validación ---
        model.eval()
        avg_val_loss = 0.0
        batch_count_val = 0
        progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{total_epochs} [Val]', leave=False, unit="batch")

        with torch.no_grad():
            for inputs, _ in progress_bar_val:
                if inputs.shape[1] != hp['window_size'] or inputs.shape[2] != num_expected_nodes or inputs.shape[
                    3] != actual_num_features:
                    logger.warning(f"Skip val batch shape {inputs.shape}.")
                    continue

                inputs = inputs.to(device)
                try:
                    outputs = model(inputs, edge_index)
                    loss = criterion(outputs, inputs)

                    if not torch.isfinite(loss):
                        logger.warning(
                            f"Loss NaN/Inf detectada en val epoch {epoch + 1}, batch {batch_count_val + 1}. Omitiendo batch.")
                        continue

                    avg_val_loss += loss.item()
                    batch_count_val += 1
                    progress_bar_val.set_postfix({'Val Loss': f'{loss.item():.6f}'})
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error(
                            f"CUDA out of memory durante val epoch {epoch + 1}. Batch Size={batch_size}. Abortando.",
                            exc_info=False)
                        if device.type == 'cuda': torch.cuda.empty_cache(); gc.collect()
                        if file_handler: file_handler.close(); logger.removeHandler(file_handler)
                        return
                    else:
                        logger.error(f"Error Runtime en val epoch {epoch + 1}: {e}", exc_info=True)
                        continue
                except Exception as e:
                    logger.error(f"Error inesperado en val epoch {epoch + 1}: {e}", exc_info=True)
                    continue

        if batch_count_val == 0:
            logger.warning(f"Epoch {epoch + 1}: No se procesaron batches de validación.")
            avg_val_loss = float('inf')
        else:
            avg_val_loss /= batch_count_val
        history['val_loss'].append(avg_val_loss if batch_count_val > 0 else None)

        epoch_duration = datetime.now() - epoch_start_time

        if not np.isfinite(avg_val_loss):
            logger.error(
                f"Epoch {epoch + 1}/{total_epochs} -> Train Loss: {avg_train_loss:.6f}, Val Loss: INVALID. Deteniendo.")
            if file_handler: file_handler.close(); logger.removeHandler(file_handler)
            return

        logger.info(
            f"Epoch {epoch + 1}/{total_epochs} -> Lr: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Dur: {epoch_duration})"
        )

        # --- Scheduler y Early Stopping ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            try:
                if np.isfinite(best_val_loss):
                    # 1. Guardar modelo
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"   -> Nuevo mejor modelo guardado. Val Loss: {best_val_loss:.6f}")

                    # --- INICIO DE LÓGICA DE GUARDADO INTERMEDIO ---
                    # (Guardamos el estado para que la reanudación sea reanudable)

                    # 2. Guardar Hiperparámetros (actualizando best_val_loss)
                    hp_path = os.path.join(output_dir, 'hyperparameters_wavelet_gnn.json')
                    try:
                        with open(hp_path, 'w', encoding='utf-8') as f:
                            hp_save = hp.copy()
                            model_type_str = f"STG-AE (Resumed from {os.path.basename(resume_run_path)})" if resume_run_path else "STG-AE (Wavelet GNN)"
                            hp_save['model_type'] = model_type_str
                            hp_save['num_features'] = actual_num_features
                            hp_save['total_params'] = total_params
                            hp_save['best_val_loss'] = best_val_loss  # <-- Guardar el valor actualizado
                            hp_save['training_duration_this_run'] = str(
                                datetime.now() - start_time_train)  # Duración hasta ahora
                            hp_save['resumed_from'] = resume_run_path
                            json.dump(hp_save, f, indent=4)
                    except Exception as e:
                        logger.error(f"   -> Error guardando HPs intermedios: {e}")

                    # 3. Guardar historial de pérdidas (completo hasta esta época)
                    history_path = os.path.join(output_dir, 'loss_history_wavelet_gnn.json')
                    try:
                        history_safe = {}
                        for key, values in history.items():
                            history_safe[key] = [(v if (v is not None and np.isfinite(v)) else None) for v in values]
                        with open(history_path, 'w', encoding='utf-8') as f:
                            json.dump(history_safe, f, indent=4)
                    except Exception as e:
                        logger.error(f"   -> Error guardando historial intermedio: {e}")

                    logger.info("   -> Historial de pérdidas y HPs actualizados en disco.")
                    # --- FIN DE LÓGICA DE GUARDADO INTERMEDIO ---

                else:
                    logger.warning("   -> Val Loss es inf/nan, no se guarda el modelo ni el historial.")
            except Exception as e:
                logger.error(f"   -> Error guardando artefactos del mejor modelo: {e}", exc_info=True)
        else:
            patience_counter += 1
            logger.debug(f"   Patience counter: {patience_counter}/{hp['patience']}")

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-7:
            logger.info(f"--- Parada Temprana: Learning Rate ({current_lr:.2e}) demasiado bajo. ---")
            break

        if patience_counter >= hp['patience']:
            logger.info(
                f"--- Parada Temprana: La pérdida de validación no mejoró por {hp['patience']} épocas consecutivas. ---")
            break

    # --- Fin del Bucle de Entrenamiento ---
    end_time_train = datetime.now()
    total_training_duration = end_time_train - start_time_train
    logger.info(f"--- Entrenamiento (Reanudado) Finalizado ---")
    logger.info(f"Duración de esta sesión: {total_training_duration}")
    logger.info(f"Mejor pérdida de validación alcanzada (global): {best_val_loss:.6f}")

    # --- Guardar artefactos finales ---
    logger.info("Guardando artefactos finales...")

    # Guardar Scaler (el que se cargó o creó)
    scaler_path = os.path.join(output_dir, 'scaler_wavelet_gnn.gz')
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler guardado en: {scaler_path}")
    except Exception as e:
        logger.error(f"Error guardando el scaler: {e}")

    # Guardar Hiperparámetros (el combinado)
    hp_path = os.path.join(output_dir, 'hyperparameters_wavelet_gnn.json')
    try:
        with open(hp_path, 'w', encoding='utf-8') as f:
            hp_save = hp.copy()
            hp_save['model_type'] = f"STG-AE (Resumed from {os.path.basename(resume_run_path)})"
            hp_save['num_features'] = actual_num_features
            hp_save['total_params'] = total_params
            hp_save['best_val_loss'] = best_val_loss if np.isfinite(best_val_loss) else None
            hp_save['training_duration_this_run'] = str(total_training_duration)
            hp_save['resumed_from'] = resume_run_path
            json.dump(hp_save, f, indent=4)
        logger.info(f"Hiperparámetros guardados en: {hp_path}")
    except Exception as e:
        logger.error(f"Error guardando hiperparámetros: {e}")

    # Guardar historial de pérdidas (el COMPLETO)
    history_path = os.path.join(output_dir, 'loss_history_wavelet_gnn.json')
    try:
        history_safe = {}
        for key, values in history.items():
            history_safe[key] = [(v if (v is not None and np.isfinite(v)) else None) for v in values]

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_safe, f, indent=4)
        logger.info(f"Historial de pérdidas (completo) guardado en: {history_path}")
    except Exception as e:
        logger.error(f"Error guardando historial de pérdidas: {e}")

    # --- Plotear curvas de pérdida (COMPLETAS) ---
    try:
        # Ahora 'history_safe' contiene todas las épocas (viejas + nuevas)
        epochs = list(range(1, len(history_safe.get('train_loss', [])) + 1))
        train_loss_plot = [l for l in history_safe.get('train_loss', []) if l is not None]
        val_loss_plot = [l for l in history_safe.get('val_loss', []) if l is not None]
        epochs_train = [epochs[i] for i, l in enumerate(history_safe.get('train_loss', [])) if l is not None]
        epochs_val = [epochs[i] for i, l in enumerate(history_safe.get('val_loss', [])) if l is not None]

        if not epochs_train or not epochs_val:
            logger.warning("No hay suficientes datos válidos de pérdida para plotear.")
        else:
            plt.figure(figsize=(12, 7))
            plt.plot(epochs_train, train_loss_plot, label='Training Loss', marker='.', linestyle='-', markersize=4)
            plt.plot(epochs_val, val_loss_plot, label='Validation Loss', marker='.', linestyle='--', markersize=4)
            plt.title('Training & Validation Loss (STG-AE Wavelet - Resumed)')  # Título actualizado
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')

            # Línea vertical para indicar dónde se reanudó
            if start_epoch > 0:
                plt.axvline(x=start_epoch + 0.5, color='r', linestyle='--', label=f'Resumed at Epoch {start_epoch + 1}')

            all_losses_plot = train_loss_plot + val_loss_plot
            valid_losses_plot = [l for l in all_losses_plot if l is not None and np.isfinite(l)]
            if not valid_losses_plot:
                min_loss_plot = 0.001;
                max_loss_plot = 1.0
            else:
                min_loss_plot = min(valid_losses_plot)
                max_loss_plot = max(valid_losses_plot)

            if (max_loss_plot / max(min_loss_plot, 1e-9) > 100) or min_loss_plot < 0.01:
                plt.yscale('log')
                plt.ylabel('MSE Loss (Log Scale)')
                plot_min_y = max(min_loss_plot * 0.8, 1e-9)
                plot_max_y = max_loss_plot * 1.2
                if plot_min_y >= plot_max_y: plot_max_y = plot_min_y * 10
                plt.ylim(bottom=plot_min_y, top=plot_max_y)

            else:
                plt.ylim(bottom=min(0, min_loss_plot * 0.9 if min_loss_plot < 0 else 0), top=max_loss_plot * 1.1)

            plt.legend()
            plt.grid(True, linestyle=':')
            loss_curve_path = os.path.join(output_dir, 'loss_curve_wavelet_gnn_RESUME.png')
            plt.savefig(loss_curve_path, dpi=300)
            plt.close()
            logger.info(f"Gráfico de curvas de pérdida (completo) guardado en: {loss_curve_path}")

    except Exception as e:
        logger.error(f"Error generando gráfico de curvas de pérdida: {e}", exc_info=True)

    # --- Cerrar Handler ---
    if file_handler:
        logger.info("Cerrando archivo de log de entrenamiento.")
        file_handler.close()
        logger.removeHandler(file_handler)

    # --- BLOQUE DE EJECUCIÓN ---


if __name__ == '__main__':
    # --- Rutas ---

    # !!! IMPORTANTE: RUTA A LA EJECUCIÓN ANTERIOR QUE QUIERES CONTINUAR !!!
    # (Basado en tus logs, debería ser esta)
    RESUME_RUN_PATH = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343"
    # Si quieres entrenar desde cero, pon: RESUME_RUN_PATH = None

    # Ruta a los datos (sigue siendo necesaria)
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    # Directorio base de salida (para la *nueva* carpeta)
    base_output_dir = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet"

    # --- Hiperparámetros (PARA LA NUEVA SESIÓN DE ENTRENAMIENTO) ---
    HP = {
        # --- Configuración de Reanudación ---
        "epochs": 50,  # <-- Cuántas ÉPOCAS ADICIONALES entrenar (total 50+50=100)
        "learning_rate": 0.0001,  # <-- BAJAR LR para fine-tuning
        "batch_size": 16,  # <-- Mantener (o ajustar si hubo OOM)
        "patience": 15,  # <-- Más paciencia para el fine-tuning
        "scheduler_patience": 7,
        "scheduler_factor": 0.5,
        "weight_decay": 1e-5

        # NOTA: Los HPs de arquitectura (gnn_hidden, rnn_hidden, window_size, etc.)
        # se cargarán automáticamente desde el JSON de la ejecución anterior.
    }

    # --- Validar Ruta de Reanudación (si se usa) ---
    if RESUME_RUN_PATH and not os.path.isdir(RESUME_RUN_PATH):
        print(f"Error: Directorio de reanudación no encontrado en {RESUME_RUN_PATH}")
        sys.exit(1)

    # --- Crear Directorio de Salida (NUEVO) ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if RESUME_RUN_PATH:
        # Crear un nombre de carpeta basado en la original
        original_run_name = os.path.basename(RESUME_RUN_PATH)
        output_directory = os.path.join(base_output_dir,
                                        f"RESUME_{original_run_name}_e{HP['epochs']}_lr{HP['learning_rate']}_{timestamp}")
    else:
        # Fallback si se decide entrenar de cero
        output_directory = os.path.join(base_output_dir,
                                        f"run_wavelet_NEW_h{HP.get('gnn_hidden', 'NA')}_r{HP.get('rnn_hidden', 'NA')}_{timestamp}")

    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Resultados (reanudados) se guardarán en: {output_directory}")
    except OSError as e:
        print(f"Error creando directorio de salida {output_directory}: {e}")
        sys.exit(1)

        # --- Validar Directorio de Datos ---
    if not os.path.isdir(data_folder_path):
        print(f"Error: Directorio de datos no encontrado en {data_folder_path}")
        sys.exit(1)

    # --- Ejecutar Experimento ---
    try:
        run_experiment_wavelet_gnn(
            data_directory=data_folder_path,
            output_dir=output_directory,
            hp=HP,
            resume_run_path=RESUME_RUN_PATH  # <-- Pasar la ruta de reanudación
        )
    except Exception as e:
        if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            logger.critical(f"Error fatal durante la ejecución del experimento: {e}", exc_info=True)
        else:
            print(f"Error fatal durante la ejecución del experimento (antes de iniciar log): {e}")

        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                try:
                    handler.close(); logger.removeHandler(handler)
                except:
                    pass
        sys.exit(1)

