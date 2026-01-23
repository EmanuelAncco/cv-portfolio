# -*- coding: utf-8 -*-
"""
train_no_gnn.py

Entrena un Autoencoder Espacio-Temporal (ST-AE) estándar (solo GRU),
sin usar capas GNN. Este modelo sirve como línea base ("retador")
para comparar contra el STG-AE (con GNN) en un estudio de ablación.

Basado en el script original 'entrenamiento.txt'.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split # No necesario si no hay división train/test aquí
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import joblib
import logging
import sys

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger() # Logger Raíz
logger.setLevel(logging.INFO)

# Limpiar handlers existentes para evitar duplicados en re-ejecuciones
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Handler para consola
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# El FileHandler se añadirá en run_experiment

# --- LÓGICA DE DATOS (Idéntica al original) ---

class SpatioTemporalWindowDataset(Dataset):
    """Dataset para cargar ventanas espacio-temporales."""
    def __init__(self, data_dict, window_size, stride=1, num_expected_nodes=5):
        self.window_size = window_size
        self.stride = stride
        self.num_expected_nodes = num_expected_nodes # Guardamos el número esperado
        local_logger = logging.getLogger(self.__class__.__name__)

        if not data_dict:
            local_logger.error("El diccionario de datos de entrada está vacío.")
            raise ValueError("El diccionario de datos de entrada está vacío.")

        # Validar y filtrar datos
        valid_data_dict = {}
        min_len = float('inf')
        for sid, data in data_dict.items():
            if data is not None and isinstance(data, np.ndarray) and data.ndim > 0 and len(data) >= window_size:
                 # Asegurar que sea 2D (N, 1) antes de apilar
                if data.ndim == 1:
                    data = data[:, np.newaxis]
                elif data.shape[1] != 1:
                    local_logger.warning(f"Sensor {sid} data has shape {data.shape}, expected (N, 1). Skipping.")
                    continue

                valid_data_dict[sid] = data
                min_len = min(min_len, len(data))
            else:
                 local_logger.warning(f"Datos inválidos o insuficientes para sensor {sid}. Longitud: {len(data) if data is not None else 'None'}, WinSize: {window_size}. Omitiendo.")

        if not valid_data_dict:
            local_logger.error("No hay datos válidos en el diccionario después de filtrar.")
            raise ValueError("No hay datos válidos en el diccionario después de filtrar.")

        if min_len == float('inf') or min_len < window_size:
             min_len_val = min_len if min_len != float('inf') else 'N/A'
             local_logger.error(f"Longitud mínima ({min_len_val}) insuficiente para window_size ({window_size}).")
             raise ValueError(f"Longitud mínima ({min_len_val}) insuficiente para window_size ({window_size}).")

        # Ordenar por ID de sensor y truncar
        # Usamos range(1, num_expected_nodes + 1) para asegurar el orden correcto y manejar faltantes
        processed_data_list = []
        actual_node_ids = []
        for sid in range(1, self.num_expected_nodes + 1):
            if sid in valid_data_dict:
                processed_data_list.append(valid_data_dict[sid][:min_len])
                actual_node_ids.append(sid)
            else:
                # Si falta un sensor, podemos decidir qué hacer:
                # Opción 1: Fallar (como ahora)
                # Opción 2: Rellenar con ceros (requiere ajustar el input_size del modelo)
                local_logger.error(f"Faltan datos para el sensor esperado {sid}. No se puede continuar con la estructura actual.")
                raise ValueError(f"Faltan datos para el sensor esperado {sid}.")

        if not processed_data_list:
             local_logger.error("La lista de datos procesados está vacía.")
             raise ValueError("La lista de datos procesados está vacía.")

        self.data = np.concatenate(processed_data_list, axis=1) # Shape: (min_len, num_actual_nodes)
        self.num_nodes = self.data.shape[1] # Número real de nodos usados
        local_logger.info(f"Datos concatenados con shape: {self.data.shape}. Sensores usados: {actual_node_ids}")

        if self.num_nodes != self.num_expected_nodes:
             local_logger.warning(f"El número de nodos con datos válidos ({self.num_nodes}) no coincide con el esperado ({self.num_expected_nodes}).")
             # Considerar si esto debe ser un error dependiendo de la lógica del modelo

        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples <= 0:
            local_logger.warning(f"Número de muestras <= 0 ({self.n_samples}). Longitud datos: {len(self.data)}, WinSize: {window_size}, Stride: {stride}.")
            self.n_samples = 0 # Asegurar que sea no negativo
        local_logger.info(f"Dataset creado con {self.num_nodes} nodos, {len(self.data)} puntos, {self.n_samples} ventanas.")


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        if start < 0 or end > len(self.data): # Chequeo más robusto
             local_logger = logging.getLogger(self.__class__.__name__)
             local_logger.error(f"Índice {idx} genera rango [{start}, {end}) fuera de límites [{0}, {len(self.data)}]. Stride={self.stride}, WinSize={self.window_size}, N_Samples={self.n_samples}")
             raise IndexError(f"Índice {idx} fuera de rango.")

        window = self.data[start:end] # Shape: (window_size, num_nodes)
        # Añadir dimensión de feature = 1
        window = window[:, :, np.newaxis] # Shape: (window_size, num_nodes, 1)
        return torch.FloatTensor(window), torch.FloatTensor(window)


# --- ARQUITECTURA DEL AUTOENCODER SIN GNN ---

class SpatioTemporalAutoencoderNoGNN(nn.Module):
    """
    Autoencoder ST-AE que usa solo GRU para codificar/decodificar.
    La GNNLayer ha sido eliminada.
    """
    def __init__(self, num_nodes, num_features, window_size, rnn_hidden=64, rnn_layers=2):
        super(SpatioTemporalAutoencoderNoGNN, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.window_size = window_size
        self.rnn_hidden_dim = rnn_hidden
        self.rnn_layers = rnn_layers

        # El input de la GRU ahora es el número de nodos * número de features
        self.rnn_input_size = num_nodes * num_features
        # La salida del decoder debe reconstruir el input original aplanado
        self.rnn_decoder_output_size = self.rnn_input_size

        logger.info(f"Initializing STAutoencoderNoGNN: N={num_nodes}, F={num_features}, T={window_size}")
        logger.info(f"  RNN Encoder: Input={self.rnn_input_size}, Hidden={self.rnn_hidden_dim}, Layers={self.rnn_layers}")
        logger.info(f"  RNN Decoder: Input={self.rnn_hidden_dim}, Hidden={self.rnn_decoder_output_size}, Layers={self.rnn_layers}")


        # Encoder: Recibe la secuencia aplanada espacialmente
        self.rnn_encoder = nn.GRU(input_size=self.rnn_input_size,
                                  hidden_size=self.rnn_hidden_dim,
                                  batch_first=True,
                                  num_layers=self.rnn_layers,
                                  bidirectional=False) # Bidireccional podría ser una opción a explorar

        # Decoder: Recibe el estado oculto y reconstruye la secuencia aplanada
        self.rnn_decoder = nn.GRU(input_size=self.rnn_hidden_dim,
                                  hidden_size=self.rnn_decoder_output_size, # La salida debe coincidir con el input del encoder
                                  batch_first=True,
                                  num_layers=self.rnn_layers,
                                  bidirectional=False)

        # Capa lineal final opcional para ajustar la salida (a veces ayuda)
        # self.output_layer = nn.Linear(self.rnn_decoder_output_size, self.rnn_input_size)


    def forward(self, x):
        # x shape: [B, T, N, F]
        batch_size, T_actual, N_actual, F_actual = x.shape # Obtener dimensiones reales

        # Validaciones de shape en forward
        if T_actual != self.window_size or N_actual != self.num_nodes or F_actual != self.num_features:
            logger.warning(f"Shape de entrada inesperado en forward: {x.shape}. Esperado: ({batch_size}, {self.window_size}, {self.num_nodes}, {self.num_features}). Intentando continuar...")
            # Podríamos intentar ajustar self.window_size aquí, pero es arriesgado
            # self.window_size = T_actual

        # 1. Aplanar Nodos y Features para la GRU
        # Reshape de [B, T, N, F] -> [B, T, N * F]
        # Usar T_actual, N_actual, F_actual para el reshape
        rnn_input = x.reshape(batch_size, T_actual, N_actual * F_actual)


        # 2. RNN Encoder
        # output shape: [B, T, rnn_hidden]
        # h_n shape: [num_layers * num_directions, B, rnn_hidden]
        try:
             _, h_n = self.rnn_encoder(rnn_input)
        except RuntimeError as e:
            logger.error(f"Error en RNN Encoder. Input shape: {rnn_input.shape}. Error: {e}")
            raise e


        # 3. Preparar entrada para el Decoder
        # Usamos el último estado oculto de la última capa como vector latente
        # h_n[-1] toma el estado de la última capa. Shape: [B, rnn_hidden]
        # Lo repetimos T_actual veces para alimentar el decoder paso a paso
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, T_actual, 1) # Shape: [B, T_actual, rnn_hidden]

        # 4. RNN Decoder
        # rnn_decoded shape: [B, T_actual, rnn_decoder_output_size (N*F)]
        try:
            rnn_decoded, _ = self.rnn_decoder(decoder_input)
        except RuntimeError as e:
            logger.error(f"Error en RNN Decoder. Input shape: {decoder_input.shape}. Error: {e}")
            raise e

        # (Opcional) Pasar por capa lineal
        # rnn_decoded = self.output_layer(rnn_decoded)

        # 5. Reshape final para coincidir con la entrada original
        # Reshape de [B, T_actual, N*F] -> [B, T_actual, N, F]
        try:
            reconstructed_x = rnn_decoded.reshape(batch_size, T_actual, N_actual, F_actual)
        except RuntimeError as e:
            logger.error(f"Error en reshape final. Input shape: {rnn_decoded.shape}. Target: ({batch_size}, {T_actual}, {N_actual}, {F_actual}). Error: {e}")
            raise e

        return reconstructed_x


# --- FUNCIÓN PRINCIPAL DE EXPERIMENTO (Modificada) ---

def run_experiment_no_gnn(data_directory, output_dir, hp):
    """
    Función principal para entrenar el modelo ST-AE (sin GNN).
    """
    # --- Configuración del logging de archivo para esta ejecución ---
    log_file_path = os.path.join(output_dir, 'training_log.txt')
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging de entrenamiento iniciado. Guardando en: {log_file_path}")
    logger.info(f"--- Iniciando Experimento ST-AE (Sin GNN) ---")
    logger.info(f"Directorio de datos: {data_directory}")
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info(f"Hiperparámetros (hp): {hp}")


    # --- Carga de Datos ---
    # Asumiendo 5 sensores basados en el script original define_bridge_graph
    # Si este número puede variar, debería ser un parámetro.
    num_expected_nodes = 5
    logger.info(f"Esperando datos de {num_expected_nodes} sensores.")

    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
    if not all_files:
        logger.error(f"No se encontraron archivos .txt en {data_directory}")
        if file_handler: logger.removeHandler(file_handler); file_handler.close() # Asegurar cierre
        return

    # Diccionario para almacenar datos por sensor ID (1 a 5)
    sensor_data_raw = {i: [] for i in range(1, num_expected_nodes + 1)}
    files_processed = 0
    files_skipped = 0

    logger.info(f"Procesando {len(all_files)} archivos encontrados...")
    for f_path in tqdm(all_files, desc="Cargando archivos"):
        try:
            # Extraer ID del sensor del nombre del archivo (asume formato '<id>_...')
            filename = os.path.basename(f_path)
            sid_str = filename.split('_')[0]
            sid = int(sid_str)

            if sid in sensor_data_raw:
                # Leer solo la segunda columna (índice 1)
                data = pd.read_csv(f_path, sep='\s+', header=None, usecols=[1], engine='python').values
                if data is not None and data.size > 0:
                    sensor_data_raw[sid].append(data)
                    files_processed += 1
                else:
                    logger.warning(f"Archivo vacío o ilegible omitido: {filename}")
                    files_skipped += 1
            else:
                logger.warning(f"ID de sensor '{sid}' extraído de '{filename}' no está en el rango esperado [1-{num_expected_nodes}]. Omitiendo archivo.")
                files_skipped += 1
        except (ValueError, IndexError): # Captura errores específicos de parseo
             logger.warning(f"No se pudo extraer ID numérico del archivo {filename}. Omitiendo.")
             files_skipped += 1
        except pd.errors.EmptyDataError:
             logger.warning(f"Archivo vacío (EmptyDataError): {filename}. Omitiendo.")
             files_skipped += 1
        except Exception as e:
            logger.error(f"Error inesperado procesando archivo {f_path}: {e}. Omitiendo.")
            files_skipped += 1


    logger.info(f"Carga inicial completa. Archivos procesados: {files_processed}, Omitidos: {files_skipped}")

    # Concatenar datos de múltiples archivos para el mismo sensor
    sensor_data_concat = {}
    min_len_post_concat = float('inf')
    sensors_with_data = []
    for sid, data_list in sensor_data_raw.items():
        if data_list:
            try:
                # Asegurarse que todos los arrays son 2D (N, 1) antes de concatenar
                data_list_2d = [d.reshape(-1, 1) if d.ndim == 1 else d for d in data_list if d.size > 0] # Filtrar vacíos aquí también
                if not data_list_2d:
                    logger.warning(f"No hay arrays no vacíos para concatenar en sensor {sid}.")
                    continue
                concatenated_data = np.concatenate(data_list_2d, axis=0)
                sensor_data_concat[sid] = concatenated_data
                min_len_post_concat = min(min_len_post_concat, len(concatenated_data))
                sensors_with_data.append(sid)
                logger.info(f"Sensor {sid}: {len(data_list_2d)} archivos concatenados -> {concatenated_data.shape}")
            except ValueError as e:
                 logger.error(f"Error concatenando datos para sensor {sid}: {e}. Omitiendo sensor.")
                 # No añadir a sensors_with_data
            except Exception as e: # Captura general por si acaso
                 logger.error(f"Error inesperado concatenando datos para sensor {sid}: {e}. Omitiendo sensor.")
                 # No añadir a sensors_with_data
        else:
            logger.warning(f"No se cargaron datos válidos para el sensor {sid}.")

    # Verificar si tenemos datos de TODOS los sensores esperados
    if len(sensors_with_data) != num_expected_nodes:
        missing_sensors = set(range(1, num_expected_nodes + 1)) - set(sensors_with_data)
        logger.error(f"Faltan datos concatenados para los sensores: {missing_sensors}. No se puede continuar.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return
    logger.info(f"Datos concatenados para sensores: {sensors_with_data}")

    # Verificar si min_len_post_concat es válido
    if min_len_post_concat == float('inf') or min_len_post_concat < hp['window_size']:
         len_val = min_len_post_concat if min_len_post_concat != float('inf') else 'N/A'
         logger.error(f"Longitud mínima post-concatenación ({len_val}) es inválida o insuficiente para window_size ({hp['window_size']}).")
         if file_handler: logger.removeHandler(file_handler); file_handler.close()
         return


    # --- División y Escalado ---
    # Usaremos una división simple aquí, asumiendo que todos los datos son 'sanos'
    # para el entrenamiento del autoencoder.
    logger.info("Escalando datos...")
    scaler = StandardScaler()

    # Ajustar el scaler usando TODOS los datos concatenados disponibles (solo de los sensores con datos)
    all_concat_data_for_scaling = np.concatenate([sensor_data_concat[sid] for sid in sensors_with_data], axis=0)
    if all_concat_data_for_scaling.size == 0:
         logger.error("No hay datos válidos para ajustar el StandardScaler.")
         if file_handler: logger.removeHandler(file_handler); file_handler.close()
         return
    scaler.fit(all_concat_data_for_scaling)
    logger.info("StandardScaler ajustado.")

    # Escalar los datos concatenados de cada sensor
    sensor_data_scaled = {}
    scaling_successful_sensors = []
    for sid in sensors_with_data: # Iterar solo sobre los que tienen datos
        try:
            scaled_data = scaler.transform(sensor_data_concat[sid])
            sensor_data_scaled[sid] = scaled_data
            scaling_successful_sensors.append(sid)
        except Exception as e:
             logger.error(f"Error escalando datos para sensor {sid}: {e}. Omitiendo sensor.")

    # Verificar si el escalado fue exitoso para todos los sensores necesarios
    if len(scaling_successful_sensors) != num_expected_nodes:
        missing_scaled = set(range(1, num_expected_nodes + 1)) - set(scaling_successful_sensors)
        logger.error(f"Falló el escalado para los sensores: {missing_scaled}. No se puede continuar.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    # --- Creación de Datasets (Ventanas) ---
    logger.info("Creando datasets de ventanas...")
    try:
        # Usar todos los datos escalados para crear un único dataset de ventanas
        full_dataset = SpatioTemporalWindowDataset(
            sensor_data_scaled, # Usar el diccionario de datos escalados
            hp['window_size'],
            hp['stride'],
            num_expected_nodes=num_expected_nodes # Pasar el número esperado
        )
    except ValueError as e:
        logger.error(f"Error creando el dataset de ventanas: {e}")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    if len(full_dataset) == 0:
        logger.error("El dataset de ventanas está vacío. Verifique window_size, stride y longitud de datos.")
        if file_handler: logger.removeHandler(file_handler); file_handler.close()
        return

    # Dividir el dataset de ventanas en entrenamiento y validación
    val_split = 0.15 # Porcentaje para validación
    total_windows = len(full_dataset)
    val_len = int(val_split * total_windows)
    train_len = total_windows - val_len


    if train_len <= 0 or val_len <= 0: # Chequeo más estricto
         logger.error(f"No hay suficientes ventanas para dividir en entrenamiento ({train_len}) y validación ({val_len}). Total: {total_windows}")
         if file_handler: logger.removeHandler(file_handler); file_handler.close()
         return

    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)) # Añadir semilla para reproducibilidad
    logger.info(f"Dataset dividido: {len(train_dataset)} ventanas de entrenamiento, {len(val_dataset)} ventanas de validación.")

    # --- Bucle de Entrenamiento ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de entrenamiento: {device}")

    # Determinar num_workers basado en SO
    num_workers = 4 if os.name == 'posix' else 0 # 4 para Linux/Mac, 0 para Windows
    logger.info(f"Usando {num_workers} workers para DataLoaders.")

    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False, drop_last=True) # drop_last=True puede ayudar
    val_loader = DataLoader(val_dataset, batch_size=hp['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False, drop_last=False)

    # Instanciar el modelo SIN GNN
    # Necesitamos saber el número real de nodos usados en el dataset
    num_nodes_in_dataset = full_dataset.num_nodes
    if num_nodes_in_dataset != num_expected_nodes:
         # Esto ya no debería pasar debido a las validaciones anteriores, pero lo dejamos por seguridad
         logger.error(f"El número de nodos en el dataset ({num_nodes_in_dataset}) no coincide con el esperado ({num_expected_nodes}) después de las validaciones.")
         if file_handler: logger.removeHandler(file_handler); file_handler.close()
         return

    model = SpatioTemporalAutoencoderNoGNN(
        num_nodes=num_nodes_in_dataset,
        num_features=1, # Asumiendo 1 feature (aceleración)
        window_size=hp['window_size'],
        rnn_hidden=hp.get('rnn_hidden', 64), # Usar valor de hp o default
        rnn_layers=hp.get('rnn_layers', 2)   # Usar valor de hp o default
    ).to(device)

    # Contar parámetros para comparación (opcional pero útil)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo ST-AE (No GNN) creado. Parámetros entrenables: {total_params:,}")


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp['learning_rate'], weight_decay=hp.get('weight_decay', 0)) # Añadir weight decay opcional
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', # 'min' es correcto para loss
                                                           patience=hp.get('scheduler_patience', 5),
                                                           factor=hp.get('scheduler_factor', 0.5),
                                                           verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    # Guardar el modelo en el directorio de salida específico de esta ejecución
    best_model_path = os.path.join(output_dir, 'best_model_no_gnn.pth')
    history = {'train_loss': [], 'val_loss': [], 'lr': []} # Guardar también LR

    logger.info("\n--- Iniciando Entrenamiento del ST-AE (Sin GNN) ---")
    start_time_train = datetime.now()

    for epoch in range(hp['epochs']):
        epoch_start_time = datetime.now()
        model.train()
        avg_train_loss = 0.0 # Usar float
        # Barra de progreso para entrenamiento
        progress_bar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Train]', leave=False)

        batch_count_train = 0
        for inputs, _ in progress_bar_train:
            # Asegurarse de que el input tiene el shape esperado [B, T, N, F]
            if inputs.shape[1] != hp['window_size'] or inputs.shape[2] != num_nodes_in_dataset or inputs.shape[3] != 1:
                logger.warning(f"Batch de entrenamiento con shape inesperado: {inputs.shape}. Esperado: (B, {hp['window_size']}, {num_nodes_in_dataset}, 1). Omitiendo batch.")
                continue

            inputs = inputs.to(device) # Shape: [B, T, N, F]
            optimizer.zero_grad()

            try:
                outputs = model(inputs)   # <--- Llamada al modelo SIN edge_index
                loss = criterion(outputs, inputs)

                # Verificar si loss es NaN o Inf
                if not torch.isfinite(loss):
                    logger.error(f"Loss infinita o NaN detectada en época {epoch+1}, batch {batch_count_train+1}. Deteniendo entrenamiento.")
                    # Opcional: guardar estado para depuración
                    # torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'inputs': inputs.cpu()}, os.path.join(output_dir, 'error_state.pth'))
                    if file_handler: logger.removeHandler(file_handler); file_handler.close()
                    return # Detener ejecución

                loss.backward()
                # --- *** AÑADIDO GRADIENT CLIPPING *** ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # ------------------------------------------
                optimizer.step()

                current_loss = loss.item()
                avg_train_loss += current_loss
                batch_count_train += 1
                # Actualizar descripción de la barra de progreso
                progress_bar_train.set_postfix({'Loss': f'{current_loss:.6f}'})

            except Exception as e:
                logger.error(f"Error durante forward/backward en época {epoch+1}, batch {batch_count_train+1}: {e}", exc_info=True)
                # Decidir si continuar o detenerse. Por ahora, continuamos al siguiente batch.
                continue # Saltar al siguiente batch

        # Calcular pérdida promedio de entrenamiento
        avg_train_loss /= batch_count_train if batch_count_train > 0 else 1
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr']) # Guardar LR actual


        # --- Validación ---
        model.eval()
        avg_val_loss = 0.0 # Usar float
        # Barra de progreso para validación
        progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{hp["epochs"]} [Val]', leave=False)
        batch_count_val = 0

        with torch.no_grad():
            for inputs, _ in progress_bar_val:
                 # Asegurarse de que el input tiene el shape esperado [B, T, N, F]
                if inputs.shape[1] != hp['window_size'] or inputs.shape[2] != num_nodes_in_dataset or inputs.shape[3] != 1:
                    logger.warning(f"Batch de validación con shape inesperado: {inputs.shape}. Esperado: (B, {hp['window_size']}, {num_nodes_in_dataset}, 1). Omitiendo batch.")
                    continue

                inputs = inputs.to(device)
                try:
                    outputs = model(inputs) # <--- Llamada al modelo SIN edge_index
                    loss = criterion(outputs, inputs)

                    if not torch.isfinite(loss):
                       logger.warning(f"Loss infinita o NaN detectada en validación época {epoch+1}, batch {batch_count_val+1}. Omitiendo batch de cálculo.")
                       continue # No sumar loss inválida

                    avg_val_loss += loss.item()
                    batch_count_val += 1
                     # Actualizar descripción de la barra de progreso
                    progress_bar_val.set_postfix({'Val Loss': f'{loss.item():.6f}'})
                except Exception as e:
                    logger.error(f"Error durante validación en época {epoch+1}, batch {batch_count_val+1}: {e}", exc_info=True)
                    continue # Saltar al siguiente batch

        # Calcular pérdida promedio de validación
        avg_val_loss /= batch_count_val if batch_count_val > 0 else 1
        history['val_loss'].append(avg_val_loss)

        epoch_duration = datetime.now() - epoch_start_time

        # Verificar si avg_val_loss es válido antes de loggear y usar en scheduler/early stopping
        if not np.isfinite(avg_val_loss):
             logger.error(f"Epoch {epoch + 1}/{hp['epochs']} -> Train Loss: {avg_train_loss:.6f}, Val Loss: INVALID (NaN/Inf). Deteniendo.")
             if file_handler: logger.removeHandler(file_handler); file_handler.close()
             return

        logger.info(
            f"Epoch {epoch + 1}/{hp['epochs']} -> Lr: {optimizer.param_groups[0]['lr']:.2e}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f} (Dur: {epoch_duration})"
            )

        # --- Scheduler y Early Stopping ---
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                logger.info(f"   -> Nuevo mejor modelo (No GNN) guardado en {best_model_path} (Val Loss: {best_val_loss:.6f})")
            except Exception as e:
                 logger.error(f"   -> Error guardando el mejor modelo: {e}")
                 # Decidir si continuar o detenerse. Por ahora, continuamos.
        else:
            patience_counter += 1
            logger.debug(f"   Patience counter: {patience_counter}/{hp['patience']}") # Debug

        # Revisar si el LR se ha vuelto extremadamente pequeño
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-7: # Umbral de LR muy bajo
            logger.info(f"--- Parada Temprana: Learning Rate ({current_lr:.2e}) demasiado bajo. ---")
            break

        if patience_counter >= hp['patience']:
            logger.info(f"--- Parada Temprana: La pérdida de validación no mejoró por {hp['patience']} épocas consecutivas. ---")
            break

    # --- Fin del Bucle de Entrenamiento ---
    end_time_train = datetime.now()
    total_training_duration = end_time_train - start_time_train
    logger.info(f"--- Entrenamiento Finalizado (No GNN) ---")
    logger.info(f"Duración total del entrenamiento: {total_training_duration}")
    logger.info(f"Mejor pérdida de validación alcanzada: {best_val_loss:.6f}")

    # --- Guardar artefactos finales ---
    logger.info("Guardando artefactos finales...")

    # Guardar Scaler
    scaler_path = os.path.join(output_dir, 'scaler_no_gnn.gz')
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler (No GNN) guardado en: {scaler_path}")
    except Exception as e:
        logger.error(f"Error guardando el scaler: {e}")

    # Guardar Hiperparámetros
    hp_path = os.path.join(output_dir, 'hyperparameters_no_gnn.json')
    try:
        with open(hp_path, 'w', encoding='utf-8') as f:
            # Añadir información sobre el modelo al hp dict
            hp_save = hp.copy()
            hp_save['model_type'] = 'ST-AE (No GNN)'
            hp_save['num_nodes_used'] = num_nodes_in_dataset
            hp_save['total_params'] = total_params
            hp_save['best_val_loss'] = best_val_loss if np.isfinite(best_val_loss) else None # Guardar None si fue inf/nan
            hp_save['training_duration'] = str(total_training_duration)
            json.dump(hp_save, f, indent=4)
        logger.info(f"Hiperparámetros (No GNN) guardados en: {hp_path}")
    except Exception as e:
        logger.error(f"Error guardando hiperparámetros: {e}")

    # Guardar historial de pérdidas (opcional, útil para plots rápidos)
    history_path = os.path.join(output_dir, 'loss_history_no_gnn.json')
    try:
        # Convertir posibles NaNs/Infs a None para JSON
        history_safe = {}
        for key, values in history.items():
            history_safe[key] = [v if np.isfinite(v) else None for v in values]

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_safe, f, indent=4)
        logger.info(f"Historial de pérdidas (No GNN) guardado en: {history_path}")
    except Exception as e:
        logger.error(f"Error guardando historial de pérdidas: {e}")

    # --- Plotear curvas de pérdida ---
    try:
        # Filtrar None/NaN/Inf antes de plotear
        epochs = list(range(1, len(history_safe['train_loss']) + 1))
        train_loss_plot = [l for l in history_safe['train_loss'] if l is not None]
        val_loss_plot = [l for l in history_safe['val_loss'] if l is not None]
        epochs_train = [epochs[i] for i, l in enumerate(history_safe['train_loss']) if l is not None]
        epochs_val = [epochs[i] for i, l in enumerate(history_safe['val_loss']) if l is not None]


        if not train_loss_plot or not val_loss_plot:
             logger.warning("No hay datos de pérdida válidos para plotear.")
        else:
            plt.figure(figsize=(12, 7)) # Más grande
            plt.plot(epochs_train, train_loss_plot, label='Training Loss', marker='.', linestyle='-', markersize=4)
            plt.plot(epochs_val, val_loss_plot, label='Validation Loss', marker='.', linestyle='--', markersize=4)
            plt.title('Training & Validation Loss (ST-AE No GNN)')
            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')

            # Determinar escala Y
            all_losses_plot = train_loss_plot + val_loss_plot
            min_loss_plot = min(all_losses_plot) if all_losses_plot else 0.01
            max_loss_plot = max(all_losses_plot) if all_losses_plot else 1.0

            if max_loss_plot / max(min_loss_plot, 1e-9) > 100:
                plt.yscale('log')
                plt.ylabel('MSE Loss (Log Scale)')
                plt.ylim(bottom=max(min_loss_plot * 0.8, 1e-9)) # Ajuste para log
            else:
                 plt.ylim(bottom=0) # Escala lineal desde 0

            plt.legend()
            plt.grid(True, linestyle=':')
            loss_curve_path = os.path.join(output_dir, 'loss_curve_no_gnn.png')
            plt.savefig(loss_curve_path, dpi=300)
            plt.close()
            logger.info(f"Gráfico de curvas de pérdida (No GNN) guardado en: {loss_curve_path}")

    except Exception as e:
        logger.error(f"Error generando gráfico de curvas de pérdida: {e}")


    # Cerrar y remover el file handler al final
    if file_handler:
        logger.info("Cerrando archivo de log de entrenamiento.")
        file_handler.close()
        logger.removeHandler(file_handler)


# --- BLOQUE DE EJECUCIÓN ---
if __name__ == '__main__':
    # --- Definir Rutas (USA LAS MISMAS QUE TU SCRIPT ORIGINAL) ---
    # Directorio que contiene los datos de entrenamiento (archivos txt de sensores)
    data_folder_path = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # Directorio base donde se guardarán los resultados de los entrenamientos
    # Se creará una subcarpeta única para esta ejecución dentro de 'resultados_entrenamiento_no_gnn'
    base_output_dir = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_no_gnn" # Directorio específico para este modelo

    # --- Definir Hiperparámetros ---
    # Ajusta rnn_hidden si necesitas que el número de parámetros sea similar al modelo GNN
    # Puedes calcular los parámetros del GNN y ajustar esto iterativamente.
    HP = {
        "window_size": 64,
        "stride": 32,
        "epochs": 50,         # Mismo número de épocas que el GNN
        "batch_size": 32,
        "learning_rate": 0.001, # Mantenemos 0.001 por ahora, pero considera bajarlo si persiste la inestabilidad
        "patience": 10,       # Misma paciencia para early stopping
        "rnn_hidden": 192,   # <--- *** AUMENTADO A 192 ***
        "rnn_layers": 2,
        "scheduler_patience": 5, # Coincidir con GNN
        "scheduler_factor": 0.5   # Coincidir con GNN
        # "weight_decay": 1e-5 # Opcional: L2 regularization
    }

    # --- Crear Directorio de Salida Único ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Nombre descriptivo para la carpeta de resultados
    output_directory = os.path.join(base_output_dir, f"run_no_gnn_h{HP['rnn_hidden']}_{timestamp}") # Añadir hidden dim al nombre
    try:
        os.makedirs(output_directory, exist_ok=True)
        print(f"Los resultados se guardarán en: {output_directory}") # Usar print aquí antes de configurar el logger de archivo
    except OSError as e:
        print(f"Error creando directorio de salida {output_directory}: {e}") # Usar print
        sys.exit(1) # Salir si no se puede crear el directorio

    # --- Validar Directorio de Datos ---
    if not os.path.isdir(data_folder_path):
        print(f"Error: Directorio de datos no encontrado en {data_folder_path}") # Usar print
        sys.exit(1)

    # --- Ejecutar Experimento ---
    try:
        run_experiment_no_gnn(data_folder_path, output_directory, HP)
    except Exception as e:
        # Loggear la excepción si el logger ya está configurado, sino imprimir
        if logger.hasHandlers():
            logger.critical(f"Error fatal durante la ejecución del experimento: {e}", exc_info=True)
        else:
            print(f"Error fatal durante la ejecución del experimento: {e}")
        sys.exit(1) # Salir con código de error

