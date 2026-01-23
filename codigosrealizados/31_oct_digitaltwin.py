# -*- coding: utf-8 -*-
"""
digital_twin_app.py

Aplicación web interactiva (Gemelo Digital de Comportamiento)
Carga el modelo STG-AE (Físico) entrenado y ejecuta inferencia en tiempo real
sobre datos simulados (Sano, Tráfico, Viento, Daño, etc.)

INSTRUCIONES:
1. Instala las librerías: pip install -r requirements_app.txt (ver archivo)
2. EDITA LAS 3 RUTAS de abajo (RUN_DIRECTORY, HEALTHY_DATA_DIR, DAMAGE_DATA_DIR).
3. Ejecuta este script directamente en PyCharm (Click derecho > Run).
4. Abre http://127.0.0.1:8050/ en tu navegador.

=============================================================================
CHANGELOG (v2):
- Corregida la lógica de simulación de daño (ahora usa np.std en lugar de np.mean).
- Corregida la lógica de alarma (ahora usa la media de 5 ticks en lugar de ser instantánea).
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

# --- Dependencias de la App ---
try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    import dash_mantine_components as dmc
except ImportError:
    print("=" * 80)
    print("Error: Librerías de Dash no instaladas. Por favor, ejecuta en tu terminal de PyCharm:")
    print("pip install dash dash-mantine-components plotly")
    print("=" * 80)
    sys.exit(1)

# --- Dependencias del Modelo ---
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("=" * 80)
    print("Error: torch_geometric no está instalado. Ejecuta en tu terminal de PyCharm:")
    print("pip install torch_geometric")
    print("=" * 80)
    sys.exit(1)

# Silenciar advertencias de Torch/Joblib en la app
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# TODO: ¡EDITA ESTAS 3 RUTAS!
# =============================================================================

# 1. Ruta a la carpeta de resultados de tu MEJOR modelo (el que terminó en la época 80)
RUN_DIRECTORY = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347"

# 2. Ruta a la carpeta de datos crudos (SANOS) - (Se usa 1 archivo para la simulación base)
HEALTHY_DATA_DIRECTORY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

# 3. Ruta a la carpeta de datos crudos (DAÑADOS) - (Opcional, para el botón 'Daño Real')
DAMAGE_DATA_DIRECTORY = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"

# =============================================================================

# --- Configuración del Logging de la App ---
log_formatter = logging.Formatter('%(asctime)s - [APP] - [%(levelname)s] - %(message)s')
logger = logging.getLogger("DigitalTwinApp")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)


# =============================================================================
# DEFINICIONES DE CLASES (Requeridas para cargar el modelo)
# =============================================================================

class GNNLayer(nn.Module):
    """Capa GNN personalizada (GCNConv > LeakyReLU > GCNConv)"""

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
    """Arquitectura del Autoencoder Espacio-Temporal (STG-AE)"""

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

        # --- Encoder ---
        self.gnn_encoder = GNNLayer(num_features, self.gnn_hidden_dim, self.gnn_encoder_out_dim)
        self.rnn_encoder = nn.GRU(self.gnn_encoder_out_dim * num_nodes, self.rnn_encoder_hidden_dim, batch_first=True,
                                  num_layers=self.rnn_layers)

        # --- Puente Latente ---
        self.latent_project_up = nn.Linear(self.rnn_encoder_hidden_dim, self.rnn_encoder_hidden_dim)
        self.relu = nn.LeakyReLU(0.01)

        # --- Decoder ---
        self.rnn_decoder = nn.GRU(self.rnn_encoder_hidden_dim, self.rnn_decoder_output_dim, batch_first=True,
                                  num_layers=self.rnn_layers)
        self.gnn_decoder = GNNLayer(self.gnn_hidden_dim, self.gnn_hidden_dim, num_features)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, T_actual, N_actual, F_actual = x.shape

        # 1. GNN Encoder (procesa espacialmente cada frame de tiempo)
        # (B, T, N, F) -> (B*T, N, F)
        x_reshaped = x.reshape(batch_size * T_actual, N_actual, F_actual)
        gnn_encoded = self.gnn_encoder(x_reshaped, edge_index, edge_weight)  # (B*T, N, F_gnn_out)

        # 2. RNN Encoder (procesa temporalmente la secuencia de features espaciales)
        # (B*T, N, F_gnn_out) -> (B, T, N * F_gnn_out)
        gnn_encoded_view = gnn_encoded.reshape(batch_size, T_actual, N_actual, self.gnn_encoder_out_dim)
        rnn_input = gnn_encoded_view.reshape(batch_size, T_actual, -1)
        _, h_n = self.rnn_encoder(rnn_input)  # h_n es el estado latente [n_layers, B, F_rnn_hidden]

        # 3. Vector Latente (el "cuello de botella")
        latent_vector_z = self.relu(self.latent_project_up(h_n[-1]))  # Tomamos solo la última capa [B, F_rnn_hidden]

        # 4. RNN Decoder (expande el vector latente de vuelta a una secuencia)
        # [B, F_rnn_hidden] -> [B, T, F_rnn_hidden]
        rnn_decoder_input = latent_vector_z.unsqueeze(1).repeat(1, T_actual, 1)
        rnn_decoded, _ = self.rnn_decoder(rnn_decoder_input)  # (B, T, F_rnn_dec_out = N * F_gnn_hidden)

        # 5. GNN Decoder (expande las features de vuelta a la señal original)
        # (B, T, N * F_gnn_hidden) -> (B*T, N, F_gnn_hidden)
        gnn_input_decoder = rnn_decoded.reshape(batch_size * T_actual, N_actual, self.gnn_hidden_dim)
        reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index, edge_weight)  # (B*T, N, F_original)

        # 6. Reshape Final
        # (B*T, N, F) -> (B, T, N, F)
        reconstructed_x = reconstructed_frames.reshape(batch_size, T_actual, N_actual, F_actual)

        return reconstructed_x


# =============================================================================
# FUNCIONES DE PRE-PROCESAMIENTO
# =============================================================================

def adjust_signal_length(signal, target_len):
    """Ajusta la longitud de la señal 1D (padding o truncado)."""
    current_len = len(signal)
    if current_len == target_len:
        return signal
    if current_len > target_len:
        # Truncar
        return signal[:target_len]
    # Padding
    return np.concatenate((signal, np.zeros(target_len - current_len)))


def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    """Aplica DWT multinivel y reconstruye bandas."""
    if signal is None or not isinstance(signal, np.ndarray) or signal.ndim != 1:
        return None
    original_len = len(signal)
    if target_len is None:
        target_len = original_len
    if original_len == 0:
        return None
    try:
        # Descomposición
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        reconstructed_bands = []
        # Reconstruir detalles D_i
        for i in range(level, 0, -1):
            detail_level_index = level - i + 1
            if detail_level_index >= len(coeffs):
                return None  # Error
            detail_coeffs_list = [np.zeros_like(c) for c in coeffs]
            detail_coeffs_list[detail_level_index] = coeffs[detail_level_index]
            rec_d = pywt.waverec(detail_coeffs_list, wavelet)
            reconstructed_bands.append(adjust_signal_length(rec_d, target_len))
        # Reconstruir aproximación A_level
        approx_coeffs_list = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(approx_coeffs_list, wavelet)
        reconstructed_bands.append(adjust_signal_length(rec_a, target_len))
        # Apilar [Original, A_level, D_level, ..., D1]
        original_adjusted = adjust_signal_length(signal, target_len)
        ordered_bands_rev = reconstructed_bands[::-1]
        all_bands = [original_adjusted] + ordered_bands_rev
        features = np.stack(all_bands, axis=-1)
        return features
    except Exception:
        return None  # Fallo en DWT


def create_physics_informed_graph(num_nodes=5):
    """Crea el grafo ponderado (idéntico al de entrenamiento)."""
    coords = {
        0: np.array([13.88, -4.0, -1.0]),  # Sensor 1
        1: np.array([13.88, 4.0, -1.0]),  # Sensor 2
        2: np.array([27.76, -4.0, -1.0]),  # Sensor 3
        3: np.array([27.76, 4.0, -1.0]),  # Sensor 4
        4: np.array([41.64, 0.0, -1.0])  # Sensor 5
    }
    edge_index_list = []
    edge_weight_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])
            # Calcular distancia euclidiana 3D
            dist = np.linalg.norm(coords[i] - coords[j])
            # Peso = Inverso de la distancia (sensores más cercanos tienen más influencia)
            weight = 1.0 / (dist + 1e-6)
            edge_weight_list.append(weight)
            edge_weight_list.append(weight)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
    return {'edge_index': edge_index, 'edge_weight': edge_weight, 'coords': coords}


# =============================================================================
# CARGA GLOBAL DE MODELO Y DATOS (Se ejecuta una vez al iniciar la app)
# =============================================================================

@torch.no_grad()
def load_artifacts():
    """Carga el modelo, scaler, hp y datos base en memoria."""
    try:
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

        # Cargar el modelo entrenado
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()  # Poner modelo en modo de evaluación

        # Crear la definición del grafo
        graph_def = create_physics_informed_graph(num_nodes=hp['num_nodes'])
        edge_index = graph_def['edge_index'].to(device)
        edge_weight = graph_def['edge_weight'].to(device)

        logger.info("Modelo y scaler cargados exitosamente.")

        # --- Cargar datos base para la simulación ---
        # Usamos 100,000 puntos de un sensor SANO para la simulación base
        base_data_file = None
        sano_files = [f for f in os.listdir(HEALTHY_DATA_DIRECTORY) if f.endswith('.txt')]
        # Buscar un archivo específico primero
        if '1_SANO_0.txt' in sano_files:
            base_data_file = os.path.join(HEALTHY_DATA_DIRECTORY, '1_SANO_0.txt')
        elif sano_files:  # Si no, tomar el primero que encuentre
            base_data_file = os.path.join(HEALTHY_DATA_DIRECTORY, sano_files[0])

        if not base_data_file:
            raise FileNotFoundError(f"No se encontró ningún archivo .txt en {HEALTHY_DATA_DIRECTORY}")

        logger.info(f"Cargando datos base 'Sano' desde: {base_data_file}")
        base_sano_data = pd.read_csv(base_data_file, sep='\s+', header=None, usecols=[1],
                                     engine='python', on_bad_lines='warn', nrows=100000).values.squeeze()

        # --- Cargar datos base de DAÑO REAL (opcional) ---
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
            else:
                logger.warning(f"No se encontraron datos de DAÑO REAL en {DAMAGE_DATA_DIRECTORY}")
        else:
            logger.warning(f"Directorio de DAÑO REAL no encontrado: {DAMAGE_DATA_DIRECTORY}")

        logger.info("Datos base cargados. Aplicación lista.")

        return model, scaler, hp, device, edge_index, edge_weight, base_sano_data, base_damage_data

    except Exception as e:
        logger.critical(f"Error fatal al iniciar la aplicación: {e}", exc_info=True)
        sys.exit(1)


# Cargar artefactos globales
MODEL, SCALER, HP, DEVICE, EDGE_INDEX, EDGE_WEIGHT, BASE_SANO_DATA, BASE_DAMAGE_DATA = load_artifacts()
WINDOW_SIZE = HP['window_size']
NUM_NODES = HP['num_nodes']
NUM_FEATURES = HP['num_features']
# Usar el 'best_val_loss' real del entrenamiento como base, con un multiplicador de seguridad
ALARM_THRESHOLD = HP.get('best_val_loss', 0.0084) * 1.5
logger.info(f"Umbral de alarma (best_val_loss * 1.5) fijado en: {ALARM_THRESHOLD:.6f}")


# =============================================================================
# LÓGICA DEL SIMULADOR
# =============================================================================

@torch.no_grad()
def get_model_prediction(raw_signals_window):
    """
    Toma una ventana de señales crudas [T, N], aplica DWT, escala y
    ejecuta la inferencia del modelo.
    """
    # 1. Aplicar DWT a cada sensor
    window_features_list = []
    for i in range(NUM_NODES):
        signal_1d = raw_signals_window[:, i]
        features_2d = apply_dwt_features(
            signal_1d,
            wavelet=HP['wavelet_name'],
            level=HP['wavelet_level'],
            target_len=WINDOW_SIZE
        )
        if features_2d is None:
            logger.warning(f"Error de DWT en simulación (S{i + 1})")
            return np.zeros(NUM_NODES), 0.0  # Error

        window_features_list.append(features_2d)

    # Shape: (T, N, F)
    window_features = np.stack(window_features_list, axis=1)

    # 2. Escalar (feature por feature)
    # SCALER espera [samples, features]
    # Guardamos el shape, aplanamos (T*N, F), escalamos, y recuperamos (T, N, F)
    original_shape = window_features.shape
    window_features_flat = window_features.reshape(-1, NUM_FEATURES)

    try:
        scaled_features_flat = SCALER.transform(window_features_flat)
    except Exception as e:
        logger.error(f"Error al escalar en simulación: {e}")
        return np.zeros(NUM_NODES), 0.0

    scaled_features = scaled_features_flat.reshape(original_shape)

    # 3. Preparar Tensor para el Modelo
    # Shape: (B=1, T, N, F)
    input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(DEVICE)

    # 4. Inferencia
    output_tensor = MODEL(input_tensor, EDGE_INDEX, EDGE_WEIGHT)

    # 5. Calcular Error
    # (B, T, N, F)
    error_tensor = (input_tensor - output_tensor) ** 2

    # Error total (promedio de todo)
    total_error = torch.mean(error_tensor).item()

    # Error por sensor (promedio sobre B, T, F) -> shape [N]
    sensor_errors = torch.mean(error_tensor, dim=(0, 1, 3)).cpu().numpy()

    return sensor_errors, total_error


def get_simulated_data_window(tick, state):
    """
    Genera una ventana de datos crudos [T, N] basada en el estado.
    """
    # 1. Obtener ventana base de datos "sanos"
    start_idx = (tick * WINDOW_SIZE) % (len(BASE_SANO_DATA) - WINDOW_SIZE)
    if start_idx < 0: start_idx = 0

    base_window = BASE_SANO_DATA[start_idx: start_idx + WINDOW_SIZE]

    # --- CAMBIO LÓGICA V2: Usar std para amplitud, no la media ---
    window_amplitude = np.std(base_window)
    # Evitar división por cero si la ventana está plana
    if window_amplitude < 1e-6:
        window_amplitude = 1e-6

    # Simular 5 sensores (N) correlacionados
    # [T, N]
    raw_window = np.zeros((WINDOW_SIZE, NUM_NODES))
    for i in range(NUM_NODES):
        # Todos los sensores comparten la señal base, con ruido único
        noise = np.random.randn(WINDOW_SIZE) * 0.05 * window_amplitude
        raw_window[:, i] = base_window + noise

    # 2. Aplicar simulaciones de eventos
    # Explicación: El modelo aprendió que el ruido correlacionado (tráfico, viento)
    # es 'normal'. El ruido NO correlacionado (daño) es 'anormal'.

    if state == 'trafico':
        # Ruido de alta frecuencia, alta amplitud, CORRELACIONADO
        traffic_noise = np.random.randn(WINDOW_SIZE) * 0.5 * window_amplitude
        for i in range(NUM_NODES):
            raw_window[:, i] += traffic_noise  # Se añade el *mismo* ruido a todos

    elif state == 'viento_temp':
        # Ruido de baja frecuencia (deriva), CORRELACIONADO
        wind_drift = np.sin(np.linspace(0, 2 * np.pi, WINDOW_SIZE)) * window_amplitude * 2.0
        for i in range(NUM_NODES):
            raw_window[:, i] += wind_drift  # Se añade la *misma* deriva a todos

    elif state == 'fallo_sensor':
        # El sensor 3 se va a cero (esto es una anomalía si la señal no es cero)
        raw_window[:, 2] = 0.0

    elif state == 'dano_rigidez':
        # Pérdida de rigidez (envejecimiento)
        # El daño rompe la correlación física.
        # Simulamos que S2 vibra "diferente" (desfasado y con más ruido)
        uncorrelated_noise = np.random.randn(WINDOW_SIZE) * 0.8 * window_amplitude
        raw_window[:, 1] += uncorrelated_noise  # Ruido *solo* en S2

    elif state == 'dano_real':
        if BASE_DAMAGE_DATA is not None and len(BASE_DAMAGE_DATA) > WINDOW_SIZE:
            start_idx_dmg = (tick * WINDOW_SIZE) % (len(BASE_DAMAGE_DATA) - WINDOW_SIZE)
            if start_idx_dmg < 0: start_idx_dmg = 0
            base_damage_window = BASE_DAMAGE_DATA[start_idx_dmg: start_idx_dmg + WINDOW_SIZE]

            # --- CAMBIO LÓGICA V2: Usar std de la ventana de daño ---
            damage_amplitude = np.std(base_damage_window)
            if damage_amplitude < 1e-6:
                damage_amplitude = 1e-6

            # En este caso, solo usamos el dato del S1 de daño para todos (simulación)
            # Una simulación más avanzada cargaría los 5 archivos de daño
            for i in range(NUM_NODES):
                noise = np.random.randn(WINDOW_SIZE) * 0.05 * damage_amplitude
                raw_window[:, i] = base_damage_window + noise
        else:
            # Fallback a 'dano_rigidez' si no hay datos de daño
            if tick % 10 == 0:  # Loggear solo de vez en cuando
                logger.warning("No se cargaron datos de 'Daño Real', usando 'Daño Rigidez' en su lugar.")
            uncorrelated_noise = np.random.randn(WINDOW_SIZE) * 0.8 * window_amplitude
            raw_window[:, 1] += uncorrelated_noise

    return raw_window


# =============================================================================
# DEFINICIÓN DE LA APP DASH
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=[])

# --- Todas las correcciones de API de DMC v2 están aplicadas ---
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    # withGlobalStyles y withNormalizeCSS eliminados (ahora por defecto)
    children=[
        dmc.Container(
            fluid=True,
            px="lg",
            children=[
                # --- Store and Interval ---
                dcc.Store(id='store-tick-counter', data=0),
                dcc.Interval(id='interval-tick', interval=1000, n_intervals=0),  # 1 segundo

                # --- Header ---
                dmc.Loader(
                    h=70,  # height -> h
                    p="md",
                    children=[
                        dmc.Group(
                            justify="space-between",  # position -> justify
                            children=[
                                dmc.Group(
                                    gap="xs",  # spacing -> gap
                                    children=[
                                        dmc.Loader(variant="dots", c="blue"),  # color -> c
                                        dmc.Text("Gemelo Digital de Comportamiento (SHM) - Puente Junín",
                                                 fw=700,  # weight -> fw
                                                 size="xl")
                                    ]
                                ),
                                dmc.Alert(
                                    id="alert-status",
                                    title="ESTADO: SALUDABLE",
                                    color="green",
                                    variant="outline"
                                )
                            ]
                        )
                    ]
                ),

                # --- Controles ---
                dmc.Paper(
                    shadow="md",
                    p="md",
                    my="md",
                    children=[
                        dmc.SegmentedControl(
                            id="segment-sim-state",
                            value="sano",
                            fullWidth=True,
                            size="md",
                            data=[
                                {"label": "Sano", "value": "sano"},
                                {"label": "Tráfico Pesado", "value": "trafico"},
                                {"label": "Viento/Temp.", "value": "viento_temp"},
                                {"label": "Fallo Sensor S3", "value": "fallo_sensor"},
                                {"label": "Daño (Rigidez S2)", "value": "dano_rigidez"},
                                {"label": "Daño Real (Datos)", "value": "dano_real"},
                            ],
                        ),
                        dmc.Text(id="text-sim-description", mt="sm", size="sm", c="dimmed")  # color -> c
                    ]
                ),

                # --- Visualización ---
                dmc.Grid(
                    children=[
                        # --- Columna Izquierda: Puente y Localización ---
                        dmc.GridCol(  # dmc.Col -> dmc.GridCol
                            span=4,
                            children=[
                                dmc.Paper(
                                    shadow="md",
                                    p="md",
                                    h="100%",
                                    children=[
                                        dmc.Text("Esquema del Puente",
                                                 fw=600,  # weight -> fw
                                                 size="lg",
                                                 ta="center"),  # align -> ta
                                        # --- SVG del Puente ---
                                        html.Div(
                                            style={"position": "relative", "height": "200px", "width": "100%",
                                                   "backgroundColor": "#f1f3f5", "borderRadius": "8px",
                                                   "border": "1px solid #dee2e6", "overflow": "hidden"},
                                            children=[
                                                # Viga Superior
                                                html.Div(style={"position": "absolute", "top": "25%", "left": "10%",
                                                                "right": "10%", "height": "10px",
                                                                "backgroundColor": "#868e96"}),
                                                # Viga Inferior
                                                html.Div(style={"position": "absolute", "top": "75%", "left": "10%",
                                                                "right": "10%", "height": "10px",
                                                                "backgroundColor": "#868e96"}),
                                                # Sensores
                                                dmc.Badge("S1", id="badge-s1", variant="filled", color="green",
                                                          style={"position": "absolute", "top": "25%", "left": "25%",
                                                                 "transform": "translate(-50%, -50%)"}),
                                                dmc.Badge("S2", id="badge-s2", variant="filled", color="green",
                                                          style={"position": "absolute", "top": "75%", "left": "25%",
                                                                 "transform": "translate(-50%, -50%)"}),
                                                dmc.Badge("S3", id="badge-s3", variant="filled", color="green",
                                                          style={"position": "absolute", "top": "25%", "left": "50%",
                                                                 "transform": "translate(-50%, -50%)"}),
                                                dmc.Badge("S4", id="badge-s4", variant="filled", color="green",
                                                          style={"position": "absolute", "top": "75%", "left": "50%",
                                                                 "transform": "translate(-50%, -50%)"}),
                                                dmc.Badge("S5", id="badge-s5", variant="filled", color="green",
                                                          style={"position": "absolute", "top": "50%", "left": "75%",
                                                                 "transform": "translate(-50%, -50%)"}),
                                            ]
                                        ),
                                        dmc.Divider(my="md"),
                                        dmc.Text("Localización de Error por Sensor",
                                                 fw=600,  # weight -> fw
                                                 size="lg",
                                                 ta="center"),  # align -> ta
                                        dcc.Graph(id='graph-error-location', config={'displayModeBar': False},
                                                  style={"height": "300px"})
                                    ]
                                )
                            ]
                        ),

                        # --- Columna Derecha: Error en el Tiempo ---
                        dmc.GridCol(  # dmc.Col -> dmc.GridCol
                            span=8,
                            children=[
                                dmc.Paper(
                                    shadow="md",
                                    p="md",
                                    h="100%",
                                    children=[
                                        dmc.Text("Error de Reconstrucción Total (Tiempo Real)",
                                                 fw=600,  # weight -> fw
                                                 size="lg",
                                                 ta="center"),  # align -> ta
                                        dcc.Graph(id='graph-error-time', config={'displayModeBar': False},
                                                  style={"height": "500px"})
                                    ]
                                )
                            ]
                        ),
                    ]
                )
            ]
        )
    ]
)


# =============================================================================
# CALLBACKS DE LA APP (El "Cerebro")
# =============================================================================

@app.callback(
    Output('graph-error-time', 'figure'),
    Output('graph-error-location', 'figure'),
    Output('alert-status', 'title'),
    Output('alert-status', 'color'),
    Output('text-sim-description', 'children'),
    Output('badge-s1', 'color'),
    Output('badge-s2', 'color'),
    Output('badge-s3', 'color'),
    Output('badge-s4', 'color'),
    Output('badge-s5', 'color'),
    Input('interval-tick', 'n_intervals'),
    Input('segment-sim-state', 'value'),
    State('graph-error-time', 'figure')  # Pasamos el estado anterior para continuar
)
def update_live_graphs(n, sim_state, prev_fig_time):
    # --- 1. Obtener datos y correr inferencia ---
    current_tick = n
    raw_window = get_simulated_data_window(current_tick, sim_state)
    sensor_errors, total_error = get_model_prediction(raw_window)

    # --- 2. Actualizar Figura 1: Error en el Tiempo ---
    if prev_fig_time is None:
        # Inicialización
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Error Total', line=dict(color='blue')))
        fig_time.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Umbral', line=dict(color='red', dash='dash')))
        fig_time.update_layout(
            xaxis_title="Tiempo (ticks)",
            yaxis_title="Error (MSE) - Log",
            yaxis_type="log",
            yaxis_range=[-3, 2],
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white"
        )
        x_data = []
        y_data = []
        y_thresh = []
    else:
        # Extender datos existentes
        fig_time = go.Figure(prev_fig_time)
        x_data = list(fig_time.data[0].x)
        y_data = list(fig_time.data[0].y)
        y_thresh = list(fig_time.data[1].y)

    x_data.append(current_tick)
    y_data.append(total_error)
    y_thresh.append(ALARM_THRESHOLD)

    # Mantener solo 100 puntos
    max_points = 100
    if len(x_data) > max_points:
        x_data = x_data[-max_points:]
        y_data = y_data[-max_points:]
        y_thresh = y_thresh[-max_points:]

    fig_time.data[0].x = x_data
    fig_time.data[0].y = y_data
    fig_time.data[1].x = x_data
    fig_time.data[1].y = y_thresh

    # --- 3. Actualizar Figura 2: Localización de Error ---
    sensor_labels = [f'S{i + 1}' for i in range(NUM_NODES)]
    bar_colors = ['red' if err > ALARM_THRESHOLD else 'blue' for err in sensor_errors]

    fig_loc = go.Figure(data=[
        go.Bar(
            x=sensor_labels,
            y=sensor_errors,
            marker_color=bar_colors
        )
    ])
    fig_loc.add_shape(
        type="line",
        x0=-0.5, y0=ALARM_THRESHOLD,
        x1=NUM_NODES - 0.5, y1=ALARM_THRESHOLD,
        line=dict(color="Red", width=2, dash="dash"),
        name="Umbral"
    )
    fig_loc.update_layout(
        xaxis_title="Sensor",
        yaxis_title="Error (MSE) - Log",
        yaxis_type="log",
        yaxis_range=[-3, 2],
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white"
    )

    # --- 4. Actualizar UI (Status, Badges, Descripciones) ---

    # --- CAMBIO LÓGICA V2: Lógica de Alarma Robusta ---
    # Usar la media de los últimos 5 ticks para evitar parpadeos
    # Si hay menos de 5 ticks, usa los que haya
    recent_errors = y_data[-5:]
    avg_recent_error = np.mean(recent_errors) if recent_errors else 0.0
    is_alarm = avg_recent_error > ALARM_THRESHOLD

    # Status global
    status_title = "ESTADO: ¡ALARMA DETECTADA!" if is_alarm else "ESTADO: SALUDABLE"
    status_color = "red" if is_alarm else "green"

    # Descripciones
    descriptions = {
        'sano': 'Simulando vibraciones ambientales normales. El modelo (Wavelet+GNN) filtra el ruido. El error se mantiene bajo.',
        'trafico': 'Simulando tráfico pesado (ruido correlacionado). El modelo entiende que esto es normal. El error se mantiene bajo.',
        'viento_temp': 'Simulando viento/temperatura (deriva correlacionada). El modelo ignora esta deriva de baja frecuencia. El error se mantiene bajo.',
        'fallo_sensor': '¡FALLO! El Sensor 3 (S3) reporta 0. El modelo, basado en el grafo, detecta la inconsistencia. El error del S3 se dispara.',
        'dano_rigidez': '¡ALARMA! Simulando daño (pérdida de rigidez). La correlación física del S2 se rompe. El modelo no puede reconstruirlo y el error se dispara.',
        'dano_real': 'Cargando datos de DAÑO REAL. El modelo detecta un comportamiento anómalo que excede el umbral de salud.'
    }
    sim_description = descriptions[sim_state]

    # Colores de badges (sensores)
    sensor_colors = []
    for err in sensor_errors:
        if err > ALARM_THRESHOLD * 5:  # Alarma grave
            sensor_colors.append("red")
        elif err > ALARM_THRESHOLD:  # Alerta
            sensor_colors.append("yellow")
        else:  # Sano
            sensor_colors.append("green")

    return (
        fig_time, fig_loc,
        status_title, status_color, sim_description,
        *sensor_colors
    )


# --- Ejecutar la App ---
if __name__ == '__main__':
    logger.info("Verificando rutas...")
    if not os.path.isdir(RUN_DIRECTORY) or \
            not os.path.isdir(HEALTHY_DATA_DIRECTORY):
        logger.error("=" * 80)
        logger.error("¡ERROR DE RUTA! Edita las variables RUN_DIRECTORY y HEALTHY_DATA_DIRECTORY al inicio del script.")
        logger.error("La ruta de DAÑO (DAMAGE_DATA_DIRECTORY) es opcional, pero las otras dos son requeridas.")
        logger.error("=" * 80)
    else:
        logger.info("Rutas validadas.")
        logger.info("Iniciando servidor de Dash...")
        logger.info(">>> Abre http://127.0.0.1:8050/ en tu navegador <<<")
        app.run(debug=True, port=8050)
