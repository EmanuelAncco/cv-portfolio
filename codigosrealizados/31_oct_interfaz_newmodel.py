# -*- coding: utf-8 -*-
"""
simulador_tk.py

Simulador de Gemelo Digital de Comportamiento (SHM) - Puente Junín
Interfaz gráfica de ESCRITORIO (Nativa de Python) usando Tkinter y Matplotlib.

Este script reemplaza la frágil app de Dash por una aplicación nativa
robusta que no depende de servidores web ni de librerías de UI externas.

INSTRUCIONES:
1. Instala las librerías:
   pip install matplotlib torch_geometric pywt joblib networkx
2. EDITA LAS 3 RUTAS de abajo (RUN_DIRECTORY, HEALTHY_DATA_DIR, DAMAGE_DATA_DIR).
3. Ejecuta este script directamente en PyCharm (Click derecho > Run).

=============================================================================
CHANGELOG (v8.0):
- IMPLEMENTADO: Arquitectura de 2 Pestañas (Solicitud del usuario).
  - Pestaña 1: "Validación del Estado 'Sano'" (Estática).
  - Pestaña 2: "Simulador de Anomalías" (En vivo).
- IMPLEMENTADO (Pestaña 1): Gráfico de Distribución de Error 'Sano'.
  - Pre-calcula 200 simulaciones sanas al inicio para probar que el
    error 'sano' está siempre por debajo del umbral.
- IMPLEMENTADO (Pestaña 1): Gráfico de Reconstrucción 'Sana' (5 ventanas).
  - Pre-calcula 6 ejemplos de reconstrucción 'sana' para mostrar
    visualmente que son "casi idénticas".
- IMPLEMENTADO (Pestaña 2): Simulador de Anomalías.
  - Los controles (radio buttons) se movieron a esta pestaña.
  - Contiene los 3 gráficos en vivo (Error, Barras, Reconstrucción de Anomalía)
    que se actualizan a 200ms.
- REFACTOR: `update_tick` ahora solo procesa la simulación de anomalías,
  reduciendo la carga computacional.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pywt
import joblib
import json
import logging
import sys
import random
import warnings
import tkinter as tk
from tkinter import font as tkFont
from tkinter import ttk  # Importar para usar las Pestañas (Notebook)

# --- Dependencias de la GUI ---
try:
    import matplotlib
    import matplotlib.pyplot as plt  # Importación de pyplot
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    matplotlib.use('TkAgg')  # Configurar Matplotlib para usar Tkinter
except ImportError:
    print("=" * 80)
    print("Error: matplotlib no está instalado. Ejecuta: pip install matplotlib")
    print("=" * 80)
    sys.exit(1)

# --- Dependencia del Grafo ---
try:
    import networkx as nx
except ImportError:
    print("=" * 80)
    print("Error: networkx no está instalado. Ejecuta: pip install networkx")
    print("=" * 80)
    sys.exit(1)

# --- Dependencias del Modelo ---
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("=" * 80)
    print("Error: torch_geometric no está instalado. Ejecuta: pip install torch_geometric")
    print("=" * 80)
    sys.exit(1)

# Silenciar advertencias
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# TODO: ¡EDITA ESTAS 3 RUTAS! (Las mismas de antes)
# =============================================================================

# 1. Ruta a la carpeta de resultados de tu MEJOR modelo
RUN_DIRECTORY = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347"

# 2. Ruta a la carpeta de datos crudos (SANOS)
HEALTHY_DATA_DIRECTORY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

# 3. Ruta a la carpeta de datos crudos (DAÑADOS)
DAMAGE_DATA_DIRECTORY = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"

# =============================================================================

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - [SIMULATOR] - [%(levelname)s] - %(message)s')
logger = logging.getLogger("DigitalTwinSimulator")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)


# =============================================================================
# DEFINICIONES DE CLASES (Requeridas para cargar el modelo)
# =============================================================================

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
        self.gnn_encoder = GNNLayer(num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)
        self.rnn_encoder = nn.GRU(self.gnn_encoder_out_dim * num_nodes, self.rnn_encoder_hidden_dim, batch_first=True,
                                  num_layers=self.rnn_layers)
        self.latent_project_up = nn.Linear(self.rnn_encoder_hidden_dim, self.rnn_encoder_hidden_dim)
        self.relu = nn.LeakyReLU(0.01)
        self.rnn_decoder = nn.GRU(self.rnn_encoder_hidden_dim, self.rnn_decoder_output_dim, batch_first=True,
                                  num_layers=self.rnn_layers)
        self.gnn_decoder = GNNLayer(self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, T_actual, N_actual, F_actual = x.shape
        x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)
        gnn_encoded = self.gnn_encoder(x_reshaped, edge_index, edge_weight)
        gnn_encoded_view = gnn_encoded.reshape(batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
        rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)
        _, h_n = self.rnn_encoder(rnn_input)
        latent_vector_z = self.relu(self.latent_project_up(h_n[-1]))
        rnn_decoder_input = latent_vector_z.unsqueeze(1).repeat(1, T_actual, 1)
        rnn_decoded, _ = self.rnn_decoder(rnn_decoder_input)
        gnn_input_decoder = rnn_decoded.reshape(batch_size * T_actual, N_actual, self.gnn_hidden_dim)
        reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index, edge_weight)
        reconstructed_x = reconstructed_frames.reshape(batch_size, T_actual, N_actual, F_actual)
        return reconstructed_x


# =============================================================================
# FUNCIONES DE PRE-PROCESAMIENTO
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
        approx_coeffs_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_coeffs_list, wavelet)
        reconstructed_bands.append(adjust_signal_length(rec_a, target_len))
        original_adjusted = adjust_signal_length(signal, target_len)
        ordered_bands_rev = reconstructed_bands[::-1]
        all_bands = [original_adjusted] + ordered_bands_rev
        features = np.stack(all_bands, axis=-1)
        return features
    except Exception:
        return None


def create_physics_informed_graph(num_nodes=5):
    # Coordenadas (X, Y, Z) estimadas del puente
    coords = {
        0: np.array([13.88, -4.0, -1.0]), 1: np.array([13.88, 4.0, -1.0]),
        2: np.array([27.76, -4.0, -1.0]), 3: np.array([27.76, 4.0, -1.0]),
        4: np.array([41.64, 0.0, -1.0])
    }
    edge_index_list = []
    edge_weight_list = []

    # Coordenadas (X, Y) solo para el plot 2D
    pos_2d = {i: (coords[i][0], coords[i][1]) for i in coords}  # {0: (13.88, -4.0), ...}

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])
            # Distancia 3D para los pesos
            dist_3d = np.linalg.norm(coords[i] - coords[j])
            weight = 1.0 / (dist_3d + 1e-6)
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    # Retornar también las coordenadas 2D y los pesos normalizados para el plot
    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'coords_3d': coords,
        'pos_2d': pos_2d
    }


# =============================================================================
# LÓGICA DEL SIMULADOR
# =============================================================================

@torch.no_grad()
def get_model_prediction(raw_signals_window, model, scaler, hp, graph_def):
    """
    Toma una ventana de señales crudas [T, N], aplica DWT, escala y
    ejecuta la inferencia del modelo.
    Retorna los errores, el error total, y la señal reconstruida (solo la feature original).
    """
    num_nodes = hp['num_nodes']
    num_features = hp['num_features']
    window_size = hp['window_size']
    device = graph_def['edge_index'].device

    window_features_list = []
    for i in range(num_nodes):
        signal_1d = raw_signals_window[:, i]
        features_2d = apply_dwt_features(
            signal_1d,
            wavelet=hp['wavelet_name'],
            level=hp['wavelet_level'],
            target_len=window_size
        )
        if features_2d is None:
            logger.warning(f"Error de DWT en simulación (S{i + 1})")
            return np.zeros(num_nodes), 0.0, np.zeros_like(raw_signals_window)
        window_features_list.append(features_2d)

    window_features = np.stack(window_features_list, axis=1)  # Shape: [T, N, F]
    original_shape = window_features.shape
    window_features_flat = window_features.reshape(-1, num_features)
    try:
        scaled_features_flat = scaler.transform(window_features_flat)
    except Exception as e:
        logger.error(f"Error al escalar en simulación: {e}")
        return np.zeros(num_nodes), 0.0, np.zeros_like(raw_signals_window)
    scaled_features = scaled_features_flat.reshape(original_shape)
    input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)  # Shape: [1, T, N, F]

    output_tensor = model(input_tensor, graph_def['edge_index'], graph_def['edge_weight'])

    error_tensor = (input_tensor - output_tensor) ** 2
    total_error = torch.mean(error_tensor).item()
    sensor_errors = torch.mean(error_tensor, dim=(0, 1, 3)).cpu().numpy()

    # Desescalar la primera característica (la señal original) de la reconstrucción
    reconstructed_scaled_flat = output_tensor.squeeze(0).reshape(-1, num_features).cpu().numpy()

    # La forma más segura es desescalar todo y luego tomar la primera feature
    full_descaled = scaler.inverse_transform(reconstructed_scaled_flat)
    reconstructed_signal_original_feature = full_descaled[:, 0].reshape(window_size, num_nodes)

    return sensor_errors, total_error, reconstructed_signal_original_feature


def get_simulated_data_window(tick, state, base_sano, base_dano, hp):
    """
    Genera una ventana de datos crudos [T, N] basada en el estado.
    """
    window_size = hp['window_size']
    num_nodes = hp['num_nodes']

    # 1. Obtener ventana base de datos "sanos"
    # Usar un índice aleatorio para las simulaciones de validación
    if tick is None:
        max_start = len(base_sano) - window_size - 1
        start_idx = random.randint(0, max_start if max_start > 0 else 0)
    else:
        start_idx = (tick * window_size) % (len(base_sano) - window_size)

    if start_idx < 0: start_idx = 0
    base_window = base_sano[start_idx: start_idx + window_size]

    # Asegurar que la ventana tenga el tamaño correcto (importante para datos aleatorios)
    if len(base_window) < window_size:
        base_window = np.pad(base_window, (0, window_size - len(base_window)), 'constant')

    window_amplitude = np.std(base_window)
    if window_amplitude < 1e-6: window_amplitude = 1e-6

    raw_window = np.zeros((window_size, num_nodes))
    for i in range(num_nodes):
        raw_window[:, i] = base_window

    if state == 'sano':
        pass  # No añadir ningún ruido. El error debería ser cercano a cero.
    elif state == 'trafico':
        traffic_noise = np.random.randn(window_size) * 0.5 * window_amplitude
        for i in range(num_nodes):
            raw_window[:, i] += traffic_noise
    elif state == 'viento_temp':
        wind_drift = np.sin(np.linspace(0, 2 * np.pi, window_size)) * window_amplitude * 2.0
        for i in range(num_nodes):
            raw_window[:, i] += wind_drift
    elif state == 'fallo_sensor':
        raw_window[:, 2] = 0.0  # Sensor S3 falla
    elif state == 'dano_rigidez':
        # Simular daño en S2 por ruido no correlacionado
        uncorrelated_noise = np.random.randn(window_size) * 0.8 * window_amplitude
        raw_window[:, 1] += uncorrelated_noise  # Sensor S2
    elif state == 'dano_real':
        if base_dano is not None and len(base_dano) > window_size:
            # Usar índice aleatorio si tick es None, sino secuencial
            if tick is None:
                max_start_dmg = len(base_dano) - window_size - 1
                start_idx_dmg = random.randint(0, max_start_dmg if max_start_dmg > 0 else 0)
            else:
                start_idx_dmg = (tick * window_size) % (len(base_dano) - window_size)

            if start_idx_dmg < 0: start_idx_dmg = 0
            base_damage_window = base_dano[start_idx_dmg: start_idx_dmg + window_size]

            # Asegurar tamaño
            if len(base_damage_window) < window_size:
                base_damage_window = np.pad(base_damage_window, (0, window_size - len(base_damage_window)), 'constant')

            for i in range(num_nodes):
                raw_window[:, i] = base_damage_window
        else:
            if tick is None or (tick % 10 == 0):
                logger.warning("No se cargaron datos de 'Daño Real', usando 'Daño Rigidez' en su lugar.")
            uncorrelated_noise = np.random.randn(window_size) * 0.8 * window_amplitude
            raw_window[:, 1] += uncorrelated_noise

    return raw_window


# =============================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN (TKINTER)
# =============================================================================

class DigitalTwinApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Gemelo Digital de Comportamiento (SHM) - Puente Junín (v8)")
        self.geometry("2000x800")
        self.configure(bg="#f0f0f0")

        # --- Estado de la App ---
        self.tick_counter = 0
        self.x_data_anomalia = []
        self.y_data_anomalia = []
        self.sensor_labels = []
        self.sim_state = tk.StringVar(value="dano_real")
        self.max_points = 50  # Puntos en gráficos de error
        self.update_speed = 200  # ms

        # --- Datos pre-calculados para Tab 1 ---
        self.sano_error_distribution = []
        self.sano_recon_examples = []

        # --- Cargar Artefactos ---
        logger.info("Cargando artefactos del modelo...")
        try:
            self.model, self.scaler, self.hp, self.graph_def, self.base_sano, self.base_dano = self.load_artifacts()

            base_loss = self.hp.get('best_val_loss', 0.0084)
            self.alarm_threshold = max(base_loss * 3.0, 0.05)  # Umbral robusto

            self.num_nodes = self.hp['num_nodes']
            self.sensor_labels = [f'S{i + 1}' for i in range(self.num_nodes)]
            logger.info(f"Umbral de alarma fijado en: {self.alarm_threshold:.6f}")
        except Exception as e:
            logger.critical(f"Error fatal al cargar artefactos: {e}", exc_info=True)
            self.destroy()
            sys.exit(1)

        # --- Pre-calcular datos de validación (NUEVO v8) ---
        logger.info("Pre-calculando gráficos de validación del estado 'Sano'...")
        self.pre_calculate_sano_validation(num_samples=200)
        self.pre_calculate_sano_reconstructions(num_samples=6)  # 6 para un grid de 3x2

        # --- Construir GUI ---
        self.build_gui()

        # --- Iniciar simulación (solo para Tab 2) ---
        logger.info("Iniciando bucle de simulación...")
        self.start_simulation()

    @torch.no_grad()
    def load_artifacts(self):
        """Carga el modelo, scaler, hp y datos base en memoria."""
        logger.info(f"Cargando artefactos desde: {RUN_DIRECTORY}")
        hp_path = os.path.join(RUN_DIRECTORY, 'hyperparameters_stgae_physics.json')
        scaler_path = os.path.join(RUN_DIRECTORY, 'scaler_stgae_physics.gz')
        model_path = os.path.join(RUN_DIRECTORY, 'best_model_stgae_physics.pth')

        with open(hp_path, 'r') as f:
            hp = json.load(f)
        scaler = joblib.load(scaler_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {device}")

        model = SpatioTemporalAutoencoder(hp['num_nodes'], hp['num_features'], hp['window_size'], hp.get(
            'gnn_hidden', 128), hp.get('gnn_out', 64), hp.get('rnn_hidden', 256), hp.get('rnn_layers', 2))

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        # Guardar la definición del grafo físico (incluyendo pos_2d)
        graph_def = create_physics_informed_graph(num_nodes=hp['num_nodes'])
        graph_def['edge_index'] = graph_def['edge_index'].to(device)
        graph_def['edge_weight'] = graph_def['edge_weight'].to(device)
        logger.info("Modelo y scaler cargados.")

        # Cargar datos sanos
        base_data_file = None
        sano_files = [f for f in os.listdir(HEALTHY_DATA_DIRECTORY) if f.endswith('.txt')]
        if '1_SANO_0.txt' in sano_files:
            base_data_file = os.path.join(HEALTHY_DATA_DIRECTORY, '1_SANO_0.txt')
        elif sano_files:
            base_data_file = os.path.join(HEALTHY_DATA_DIRECTORY, sano_files[0])
        if not base_data_file:
            raise FileNotFoundError(f"No se encontró ningún archivo .txt en {HEALTHY_DATA_DIRECTORY}")
        logger.info(f"Cargando datos base 'Sano' desde: {base_data_file}")
        base_sano_data = pd.read_csv(base_data_file, sep='\s+', header=None, usecols=[1],
                                     engine='python', on_bad_lines='warn', nrows=100000).values.squeeze()

        # Cargar datos dañados
        base_damage_data = None
        if os.path.isdir(DAMAGE_DATA_DIRECTORY):
            base_damage_file = None
            dano_files = [f for f in os.listdir(DAMAGE_DATA_DIRECTORY) if f.endswith('.txt')]
            if '1_DANO_0.txt' in dano_files:
                base_damage_file = os.path.join(DAMAGE_DATA_DIRECTORY, '1_DANO_0.txt')
            elif dano_files:
                base_damage_file = os.path.join(DAMAGE_DATA_DIRECTORY, dano_files[0])
            if base_damage_file:
                logger.info(f"Cargando datos base 'Daño' desde: {base_damage_file}")
                base_damage_data = pd.read_csv(base_damage_file, sep='\s+', header=None, usecols=[1],
                                               engine='python', on_bad_lines='warn', nrows=100000).values.squeeze()

        if base_damage_data is None:
            logger.warning("=" * 50)
            logger.warning("ADVERTENCIA: No se cargaron datos de 'Daño Real'.")
            logger.warning("La simulación 'Daño Real' se reemplazará con 'Daño Rigidez'.")
            logger.warning(f"Verifica la ruta: {DAMAGE_DATA_DIRECTORY}")
            logger.warning("=" * 50)

        logger.info("Datos base cargados.")
        return model, scaler, hp, graph_def, base_sano_data, base_damage_data

    def pre_calculate_sano_validation(self, num_samples=200):
        """Ejecuta N simulaciones 'sanas' para el gráfico de distribución."""
        self.sano_error_distribution = []
        for _ in range(num_samples):
            # Usar tick=None para obtener una ventana aleatoria
            raw_window_sano = get_simulated_data_window(
                None, 'sano',
                self.base_sano, self.base_dano, self.hp
            )
            _, total_error_sano, _ = get_model_prediction(
                raw_window_sano, self.model, self.scaler,
                self.hp, self.graph_def
            )
            self.sano_error_distribution.append(total_error_sano)

        # Ordenar para el gráfico de línea
        self.sano_error_distribution.sort()
        logger.info(f"Pre-cálculo de distribución de error 'Sano' (N={num_samples}) completado.")

    def pre_calculate_sano_reconstructions(self, num_samples=6):
        """Obtiene N ejemplos de reconstrucción 'sana'."""
        self.sano_recon_examples = []
        for _ in range(num_samples):
            raw_window_sano = get_simulated_data_window(
                None, 'sano',
                self.base_sano, self.base_dano, self.hp
            )
            _, _, reconstructed_signal = get_model_prediction(
                raw_window_sano, self.model, self.scaler,
                self.hp, self.graph_def
            )
            # Guardar solo el Sensor 1
            self.sano_recon_examples.append(
                (raw_window_sano[:, 0], reconstructed_signal[:, 0])
            )
        logger.info(f"Pre-cálculo de {num_samples} ejemplos de reconstrucción 'Sana' completado.")

    def build_gui(self):
        # --- Fuentes ---
        self.title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.header_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
        self.body_font = tkFont.Font(family="Helvetica", size=10)

        # --- Marco Superior (Título y Estado) ---
        top_frame = tk.Frame(self, bg="#f0f0f0")
        top_frame.pack(fill='x', side='top')

        header_frame = tk.Frame(top_frame, bg="#f0f0f0", padx=10, pady=10)
        header_frame.pack(fill='x', side='top')

        tk.Label(header_frame, text="Gemelo Digital de Comportamiento (SHM) - Puente Junín",
                 font=self.title_font, bg="#f0f0f0").pack(side='left')

        self.status_label = tk.Label(header_frame, text="ESTADO: SALUDABLE", font=self.title_font,
                                     bg="green", fg="white", padx=10, pady=5)
        self.status_label.pack(side='right')

        # --- PESTAÑAS (Notebook) ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # --- Pestaña 1: Validación del Estado 'Sano' ---
        self.tab_sano_validation = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_sano_validation, text="Validación del Estado 'Sano'")
        self.build_sano_tab(self.tab_sano_validation)

        # --- Pestaña 2: Simulador de Anomalías ---
        self.tab_anomalia_simulator = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_anomalia_simulator, text="Simulador de Anomalías (Daños)")
        self.build_anomalia_tab(self.tab_anomalia_simulator)

    def build_sano_tab(self, parent_tab):
        """Construye la UI estática para la pestaña de validación 'Sana'."""

        # --- Contenedor Superior (Distribución y Grafo) ---
        top_plot_frame = tk.Frame(parent_tab, bg="#f0f0f0")
        top_plot_frame.pack(fill='x', side='top', pady=5)

        # --- Gráfico 1 (Distribución de Error Sano) ---
        plot_frame_dist = tk.Frame(top_plot_frame, bg="white", padx=5, pady=5)
        plot_frame_dist.pack(side='left', fill='both', expand=True, padx=5)

        fig_dist = Figure(figsize=(8, 5), dpi=100)
        ax_dist = fig_dist.add_subplot(111)

        # Graficar la distribución de errores sanos
        if self.sano_error_distribution:
            y_data = np.array(self.sano_error_distribution)
            x_data = np.arange(len(y_data))

            ax_dist.plot(x_data, y_data,
                         label=f'Error "Sano" (N={len(y_data)} ventanas)',
                         color='green')
            ax_dist.axhline(y=self.alarm_threshold, color='red', linestyle='--',
                            label=f'Umbral de Alarma ({self.alarm_threshold:.4f})')
            ax_dist.set_title("Validación de Umbral: Errores 'Sanos' vs. Alarma")
            ax_dist.set_xlabel("Muestras de Ventanas 'Sanas' (ordenadas por error)")
            ax_dist.set_ylabel("Error (MSE) - Log")
            ax_dist.set_yscale('log')
            ax_dist.legend()
        else:
            ax_dist.text(0.5, 0.5, "No se pudieron calcular los datos de distribución.",
                         horizontalalignment='center', color='red')
        fig_dist.tight_layout()

        canvas_dist = FigureCanvasTkAgg(fig_dist, master=plot_frame_dist)
        canvas_dist.draw()
        canvas_dist.get_tk_widget().pack(side='top', fill='both', expand=True)

        # --- Gráfico 2 (Justificación del Grafo) ---
        plot_frame_graph = tk.Frame(top_plot_frame, bg="white", padx=5, pady=5)
        plot_frame_graph.pack(side='right', fill='both', expand=True, padx=5)

        fig_graph = Figure(figsize=(8, 5), dpi=100)
        ax_graph = fig_graph.add_subplot(111)
        self.draw_nx_graph(ax_graph)  # Llamar a la función de dibujo
        fig_graph.tight_layout()

        canvas_graph = FigureCanvasTkAgg(fig_graph, master=plot_frame_graph)
        canvas_graph.draw()
        canvas_graph.get_tk_widget().pack(side='top', fill='both', expand=True)

        # --- Gráfico 3 (Reconstrucciones Sanas) ---
        plot_frame_recon = tk.Frame(parent_tab, bg="white", padx=5, pady=5)
        plot_frame_recon.pack(fill='both', expand=True, side='bottom', pady=5)

        tk.Label(plot_frame_recon, text="Ejemplos de Reconstrucción 'Sana' (Sensor 1)",
                 font=self.header_font, bg="white").pack(pady=5)

        # 6 subplots (3x2)
        fig_recon_sano, axes = plt.subplots(3, 2, figsize=(16, 8), dpi=100, sharex=True)
        axes_flat = axes.flatten()

        time_points = np.arange(self.hp['window_size'])

        for i in range(len(self.sano_recon_examples)):
            if i >= 6: break  # Solo graficar 6
            ax = axes_flat[i]
            original, reconstructed = self.sano_recon_examples[i]

            ax.plot(time_points, original, label='Original', color='red', alpha=0.7)
            ax.plot(time_points, reconstructed, label='Reconstruida', color='blue', linestyle='--')
            ax.set_title(f"Ejemplo Sano #{i + 1}")
            ax.legend()
            ax.set_ylabel("Aceleración")

        fig_recon_sano.tight_layout()

        canvas_recon_sano = FigureCanvasTkAgg(fig_recon_sano, master=plot_frame_recon)
        canvas_recon_sano.draw()
        canvas_recon_sano.get_tk_widget().pack(side='top', fill='both', expand=True)

    def build_anomalia_tab(self, parent_tab):
        """Construye la UI para la pestaña del simulador de anomalías."""

        # --- Controles (Radios) ---
        controls_frame = tk.Frame(parent_tab, bg="#e0e0e0", padx=10, pady=10)
        controls_frame.pack(fill='x', side='top', pady=(0, 5))

        tk.Label(controls_frame, text="Seleccionar Anomalía:", font=self.header_font, bg="#e0e0e0").pack(side='left',
                                                                                                         padx=10)
        sim_options = [
            ("Tráfico Pesado", "trafico"),
            ("Viento/Temp.", "viento_temp"),
            ("Fallo Sensor S3", "fallo_sensor"),
            ("Daño (Rigidez S2)", "dano_rigidez"),
            ("Daño Real (Datos)", "dano_real")
        ]
        for text, value in sim_options:
            tk.Radiobutton(controls_frame, text=text, variable=self.sim_state, value=value,
                           font=self.body_font, bg="#e0e0e0", indicatoron=0,
                           selectcolor="#b0b0b0", relief="raised", borderwidth=2,
                           padx=10, pady=5, command=self.on_state_change).pack(side='left', expand=True, fill='x',
                                                                               padx=5)

        # --- Contenedor para 3 Gráficos en vivo ---
        plots_frame = tk.Frame(parent_tab, bg="#f0f0f0")
        plots_frame.pack(fill='both', expand=True, side='top', pady=5)

        # --- Gráfico 1 (ANOMALÍA SELECCIONADA) ---
        plot_frame_anomalia = tk.Frame(plots_frame, bg="white", padx=5, pady=5)
        plot_frame_anomalia.pack(side='left', fill='both', expand=True, padx=5)
        self.label_anomalia = tk.Label(plot_frame_anomalia, text="Error en Tiempo Real (ANOMALÍA)",
                                       font=self.header_font, bg="white")
        self.label_anomalia.pack()

        self.fig_anomalia = Figure(figsize=(5, 5), dpi=100)
        self.ax_anomalia = self.fig_anomalia.add_subplot(111)
        self.ax_anomalia_line, = self.ax_anomalia.plot([], [], label='Error Anomalía', color='blue')
        self.ax_anomalia.axhline(y=self.alarm_threshold, color='red', linestyle='--', label='Umbral')
        self.ax_anomalia.set_yscale('log')
        self.ax_anomalia.set_xlim(0, self.max_points)
        self.ax_anomalia_title = self.ax_anomalia.set_title("Simulación de Anomalía")
        self.ax_anomalia.set_xlabel("Tiempo (ticks)")
        self.ax_anomalia.set_ylabel("Error (MSE) - Log")
        self.ax_anomalia.legend()
        self.fig_anomalia.tight_layout()
        self.canvas_anomalia = FigureCanvasTkAgg(self.fig_anomalia, master=plot_frame_anomalia)
        self.canvas_anomalia.draw()
        self.canvas_anomalia.get_tk_widget().pack(side='top', fill='both', expand=True)

        # --- Gráfico 2 (Localización de Barras) ---
        plot_frame_bars = tk.Frame(plots_frame, bg="white", padx=5, pady=5)
        plot_frame_bars.pack(side='left', fill='both', expand=True, padx=5)
        self.label_barras = tk.Label(plot_frame_bars, text="Localización de Error (Anomalía)",
                                     font=self.header_font, bg="white")
        self.label_barras.pack()

        self.fig_bars = Figure(figsize=(5, 5), dpi=100)
        self.ax_bars = self.fig_bars.add_subplot(111)
        self.ax_bars_bars = self.ax_bars.bar(self.sensor_labels, [0] * self.num_nodes, color='blue')
        self.ax_bars.axhline(y=self.alarm_threshold, color='red', linestyle='--', label='Umbral')
        self.ax_bars.set_yscale('log')
        self.ax_bars_title = self.ax_bars.set_title("Error por Sensor (Anomalía)")
        self.ax_bars.set_ylabel("Error (MSE) - Log")
        self.fig_bars.tight_layout()
        self.canvas_bars = FigureCanvasTkAgg(self.fig_bars, master=plot_frame_bars)
        self.canvas_bars.draw()
        self.canvas_bars.get_tk_widget().pack(side='top', fill='both', expand=True)

        # --- Gráfico 3 (Reconstrucción de Anomalía - S1) ---
        plot_frame_recon = tk.Frame(plots_frame, bg="white", padx=5, pady=5)
        plot_frame_recon.pack(side='right', fill='both', expand=True, padx=5)
        self.label_reconstruccion = tk.Label(plot_frame_recon, text="Señal vs. Reconstrucción (Sensor 1)",
                                             font=self.header_font, bg="white")
        self.label_reconstruccion.pack()

        self.fig_recons, (self.ax_recons_signal, self.ax_recons_error) = plt.subplots(2, 1,
                                                                                      figsize=(5, 5),
                                                                                      dpi=100,
                                                                                      sharex=True)
        self.ax_recons_original, = self.ax_recons_signal.plot([], [], label='Original', color='red', alpha=0.7)
        self.ax_recons_reconstructed, = self.ax_recons_signal.plot([], [], label='Reconstruida', color='blue',
                                                                   linestyle='--')
        self.ax_recons_signal.set_title("Reconstrucción de Señal (Anomalía)")
        self.ax_recons_signal.set_ylabel("Aceleración")
        self.ax_recons_signal.legend()
        self.ax_recons_residual, = self.ax_recons_error.plot([], [], label='Error (Residual)', color='black')
        self.ax_recons_error.set_xlabel("Puntos de Tiempo")
        self.ax_recons_error.set_ylabel("Error Residual")
        self.ax_recons_error.legend()
        self.fig_recons.tight_layout()

        self.canvas_recons = FigureCanvasTkAgg(self.fig_recons, master=plot_frame_recon)
        self.canvas_recons.draw()
        self.canvas_recons.get_tk_widget().pack(side='top', fill='both', expand=True)

    def draw_nx_graph(self, ax_graph):
        """Dibuja el grafo físico en el 'ax' proporcionado."""
        try:
            G = nx.Graph()
            nodes = self.graph_def['pos_2d'].keys()
            pos = self.graph_def['pos_2d']
            labels = {i: f'S{i + 1}' for i in nodes}
            G.add_nodes_from(nodes)

            edge_index = self.graph_def['edge_index'].cpu().numpy().T
            edge_weight = self.graph_def['edge_weight'].cpu().numpy()

            if edge_weight.size == 0:
                raise ValueError("El tensor 'edge_weight' está vacío.")

            min_w = np.min(edge_weight)
            max_w = np.max(edge_weight)

            # Evitar división por cero si todos los pesos son iguales
            if max_w - min_w == 0:
                normalized_weights = [5.0 for _ in edge_weight]  # Grosor fijo
            else:
                normalized_weights = [(w - min_w) / (max_w - min_w) * 10 + 1 for w in edge_weight]  # Grosor de 1 a 11

            for i in range(len(edge_index)):
                u, v = edge_index[i]
                if u < v:  # Evitar duplicados
                    G.add_edge(u, v, weight=normalized_weights[i])

            edge_widths = [G[u][v]['weight'] for u, v in G.edges()]

            nx.draw_networkx(G,
                             pos=pos,
                             ax=ax_graph,
                             with_labels=True,
                             labels=labels,
                             node_color='#4CAF50',
                             node_size=2000,
                             font_size=12,
                             font_weight='bold',
                             font_color='white',
                             width=edge_widths,
                             edge_color='#888888')

            ax_graph.set_title("Topología del Grafo Físico (Vista en Planta)")
            ax_graph.set_xlabel("Coordenada X (m)")
            ax_graph.set_ylabel("Coordenada Y (m)")
            ax_graph.set_aspect('equal', 'box')
        except Exception as e:
            logger.error(f"Error al construir la visualización del grafo: {e}", exc_info=True)
            ax_graph.text(0.5, 0.5, f"Error al dibujar el grafo:\n{e}",
                          horizontalalignment='center', verticalalignment='center',
                          color='red')

    def on_state_change(self):
        """Reinicia los datos de los gráficos de anomalías y ajusta títulos."""
        new_state = self.sim_state.get()
        logger.info(f"Cambiando estado de simulación a: {new_state}")

        # Limpiar SOLO los datos del gráfico de anomalías
        self.x_data_anomalia = []
        self.y_data_anomalia = []
        self.tick_counter = 0  # Reiniciar contador

        # Actualizar Títulos de los gráficos de anomalía
        self.label_anomalia.config(text=f"Error en Tiempo Real ({new_state.upper()})")
        self.ax_anomalia_title.set_text(f"Simulación {new_state.upper()}")

        self.label_barras.config(text=f"Localización de Error ({new_state.upper()})")
        self.ax_bars_title.set_text(f"Error por Sensor ({new_state.upper()})")

        self.label_reconstruccion.config(text=f"Señal vs. Reconstrucción S1 ({new_state.upper()})")
        self.ax_recons_signal.set_title(f"Reconstrucción S1 ({new_state.upper()})")

        # Limpiar y re-dibujar la gráfica de reconstrucción
        self.ax_recons_original.set_data([], [])
        self.ax_recons_reconstructed.set_data([], [])
        self.ax_recons_residual.set_data([], [])
        self.canvas_recons.draw_idle()

    def start_simulation(self):
        """Inicia el bucle de actualización."""
        self.on_state_change()  # Llamar una vez para setear títulos iniciales
        self.after(self.update_speed, self.update_tick)

    def update_tick(self):
        """El "cerebro" de la app. Se ejecuta en cada tick."""

        # 1. Simulación de ANOMALÍA (seleccionada)
        state_anomalia = self.sim_state.get()
        if state_anomalia == 'dano_real' and self.base_dano is None:
            state_anomalia = 'dano_rigidez'  # Fallback

        raw_window_anomalia = get_simulated_data_window(
            self.tick_counter, state_anomalia,
            self.base_sano, self.base_dano, self.hp
        )
        sensor_errors_anomalia, total_error_anomalia, reconstructed_signal_anomalia = get_model_prediction(
            raw_window_anomalia, self.model, self.scaler,
            self.hp, self.graph_def
        )

        # 2. Actualizar datos de gráficos
        self.tick_counter += 1

        self.x_data_anomalia.append(self.tick_counter)
        self.y_data_anomalia.append(total_error_anomalia if total_error_anomalia > 1e-9 else 1e-9)

        if len(self.x_data_anomalia) > self.max_points:
            self.x_data_anomalia = self.x_data_anomalia[-self.max_points:]
            self.y_data_anomalia = self.y_data_anomalia[-self.max_points:]

        # 3. Actualizar Gráfico ANOMALÍA
        self.ax_anomalia_line.set_data(self.x_data_anomalia, self.y_data_anomalia)
        if self.x_data_anomalia:
            self.ax_anomalia.set_xlim(min(self.x_data_anomalia), max(self.x_data_anomalia) + 1)
        self.ax_anomalia.relim()
        self.ax_anomalia.autoscale_view(scalex=False, scaley=True)
        self.canvas_anomalia.draw_idle()

        # 4. Actualizar Gráfico de Barras (basado en anomalía)
        for i, (bar, err) in enumerate(zip(self.ax_bars_bars, sensor_errors_anomalia)):
            bar.set_height(err if err > 1e-9 else 1e-9)  # Floor para log
            if err > self.alarm_threshold * 2:
                color = 'red'
            elif err > self.alarm_threshold:
                color = 'orange'
            else:
                color = 'blue'
            bar.set_color(color)
        self.ax_bars.relim()
        self.ax_bars.autoscale_view(scalex=False, scaley=True)
        self.canvas_bars.draw_idle()

        # 5. Actualizar Gráfico de Reconstrucción (REDISEÑADO V7)
        window_size = self.hp['window_size']
        time_points = np.arange(window_size)

        if raw_window_anomalia.shape[1] > 0:
            original_signal_s1 = raw_window_anomalia[:, 0]
            reconstructed_signal_s1 = reconstructed_signal_anomalia[:, 0]
            residual_error = original_signal_s1 - reconstructed_signal_s1

            # Parcela superior (Señales)
            self.ax_recons_original.set_data(time_points, original_signal_s1)
            self.ax_recons_reconstructed.set_data(time_points, reconstructed_signal_s1)

            all_recon_data = np.concatenate([original_signal_s1, reconstructed_signal_s1])
            min_val = np.min(all_recon_data)
            max_val = np.max(all_recon_data)
            buffer = (max_val - min_val) * 0.1
            if buffer == 0: buffer = 1.0
            self.ax_recons_signal.set_ylim(min_val - buffer, max_val + buffer)
            self.ax_recons_signal.set_xlim(0, window_size)

            # Parcela inferior (Error Residual)
            self.ax_recons_residual.set_data(time_points, residual_error)
            min_err = np.min(residual_error)
            max_err = np.max(residual_error)
            buffer_err = (max_err - min_err) * 0.1
            if buffer_err == 0: buffer_err = 1.0
            self.ax_recons_error.set_ylim(min_err - buffer_err, max_err + buffer_err)
            self.ax_recons_error.set_xlim(0, window_size)

        self.canvas_recons.draw_idle()

        # 6. Actualizar Estado (Alarma Robusta basada en anomalía)
        recent_errors = self.y_data_anomalia[-5:]  # Media de 5 ticks para estabilidad
        avg_recent_error = np.mean(recent_errors) if recent_errors else 0.0
        is_alarm = avg_recent_error > self.alarm_threshold

        if is_alarm:
            self.status_label.config(text="ESTADO: ¡ALARMA DETECTADA!", bg='red', fg='white')
        else:
            self.status_label.config(text="ESTADO: SALUDABLE", bg='green', fg='white')

        # 7. Programar el próximo tick
        self.after(self.update_speed, self.update_tick)  # Usar variable de velocidad


# --- Ejecutar la App ---
if __name__ == '__main__':
    if not os.path.isdir(RUN_DIRECTORY) or \
            not os.path.isdir(HEALTHY_DATA_DIRECTORY):
        logger.error("=" * 80)
        logger.error("¡ERROR DE RUTA! Edita las variables RUN_DIRECTORY y HEALTHY_DATA_DIRECTORY.")
        logger.error("=" * 80)
    else:
        logger.info("Iniciando aplicación de Tkinter...")
        app = DigitalTwinApp()
        app.mainloop()

