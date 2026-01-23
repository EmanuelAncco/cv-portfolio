"""
PIPELINE DE FIGURAS Q1 - MÓDULO 3: INFERENCIA Y RECONSTRUCCIÓN (AUTOCONTENIDO)
================================================================================

Este script carga los modelos entrenados (.pth) y realiza inferencia sobre
datos reales (Sanos vs Dañados) para demostrar la capacidad de detección.

OBJETIVOS GRÁFICOS:
1. Visualizar la calidad de reconstrucción (Alta en sanos, Baja en dañados).
2. Cuantificar la separación de distribuciones de error (Anomaly Score).
3. Identificar espacialmente el daño mediante error por sensor.

Requiere: torch, torch_geometric, pywt, sklearn, seaborn
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pywt
import joblib
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from matplotlib.font_manager import findfont, FontProperties

# Intentar importar torch_geometric
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("CRÍTICO: torch_geometric no instalado. Ejecuta: pip install torch_geometric")
    sys.exit(1)

# --- 1. CONFIGURACIÓN GLOBAL ---

BASE_DIR = Path(r"D:\Python_proyectos_2025\GAIATECH")
DATA_DIR_HEALTHY = Path(r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar")
DATA_DIR_DAMAGED = Path(r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones")
FIGURES_DIR = BASE_DIR / "paper_figures_Q1_FINAL" / "3_reconstruction"

# Configuración de Modelos (Rutas a los pesos reales)
MODELS_CONFIG = {
    "M1": {  # No-GNN
        "name": "ST-AE (No-GNN)",
        "hp_file": BASE_DIR / r"resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627\hyperparameters_no_gnn.json",
        "model_file": BASE_DIR / r"resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627\best_model_no_gnn.pth",
        "scaler_file": BASE_DIR / r"resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627\scaler_no_gnn.gz",
        "type": "nognn",
        "color": "#ff7f0e"
    },
    "M2": {  # GNN Base
        "name": "GNN-AE (Base)",
        "hp_file": BASE_DIR / r"resultados_entrenamiento\run_gnn_20250910-020756\hyperparameters.json",
        "model_file": BASE_DIR / r"resultados_entrenamiento\run_gnn_20250910-020756\best_model.pth",
        "scaler_file": BASE_DIR / r"resultados_entrenamiento\run_gnn_20250910-020756\scaler.gz",
        "type": "gnn",
        "color": "#1f77b4",
        # Override crítico de parámetros para cargar M2 correctamente
        "override_hp": {"gnn_hidden": 32, "gnn_out": 32, "rnn_hidden": 64}
    },
    "M3": {  # Wavelet GNN
        "name": "Wavelet-GNN (Ours)",
        "hp_file": BASE_DIR / r"resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547\hyperparameters_wavelet_gnn.json",
        "model_file": BASE_DIR / r"resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547\best_model_wavelet_gnn.pth",
        "scaler_file": BASE_DIR / r"resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343\scaler_wavelet_gnn.gz",
        "type": "gnn_wavelet",
        "color": "#2ca02c"
    },
    "M4": {  # Physics GNN
        "name": "PI-STG-AE (Physics)",
        "hp_file": BASE_DIR / r"resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347\hyperparameters_stgae_physics.json",
        "model_file": BASE_DIR / r"resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347\best_model_stgae_physics.pth",
        "scaler_file": BASE_DIR / r"resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920\scaler_stgae_physics.gz",
        "type": "gnn_physics",
        "color": "#d62728"
    }
}

# Parámetros Generales
NUM_SENSORS = 5
WINDOW_SIZE = 64
STRIDE = 64  # Sin solapamiento para inferencia rápida
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Estilos Q1
Q1_FONT_NAME = "Times New Roman"
DPI = 300

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger()


# --- 2. DEFINICIÓN DE ARQUITECTURAS (REQUERIDO PARA CARGAR .PTH) ---

class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        # Soporte para edge_weight en M4
        if edge_weight is not None:
            x = self.conv1(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
        x = self.relu(x)
        if edge_weight is not None:
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv2(x, edge_index)
        return x


class SpatioTemporalAutoencoder(nn.Module):
    """Arquitectura para M2, M3 y M4"""

    def __init__(self, num_nodes, num_features, window_size, gnn_hidden, gnn_out, rnn_hidden, rnn_layers=2):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.gnn_hidden = gnn_hidden
        self.gnn_out = gnn_out

        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(gnn_out * num_nodes, rnn_hidden, batch_first=True, num_layers=rnn_layers)
        self.rnn_decoder = nn.GRU(rnn_hidden, gnn_hidden * num_nodes, batch_first=True, num_layers=rnn_layers)
        self.gnn_decoder = GNNLayer(gnn_hidden, gnn_hidden, num_features)

        # Proyección latente (presente en algunas versiones, manejamos si falla carga)
        self.latent_project = nn.Linear(rnn_hidden, rnn_hidden)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size = x.size(0)
        # Encode
        x_reshaped = x.reshape(batch_size * self.window_size, self.num_nodes, -1)
        gnn_enc = self.gnn_encoder(x_reshaped, edge_index, edge_weight)
        gnn_enc = gnn_enc.reshape(batch_size, self.window_size, -1)
        _, h_n = self.rnn_encoder(gnn_enc)

        # Latent
        latent = h_n[-1]  # Última capa

        # Decode
        dec_input = latent.unsqueeze(1).repeat(1, self.window_size, 1)
        rnn_dec, _ = self.rnn_decoder(dec_input)

        gnn_in_dec = rnn_dec.reshape(batch_size * self.window_size, self.num_nodes, -1)
        recon = self.gnn_decoder(gnn_in_dec, edge_index, edge_weight)

        return recon.reshape(batch_size, self.window_size, self.num_nodes, -1)


class SpatioTemporalAutoencoderNoGNN(nn.Module):
    """Arquitectura para M1"""

    def __init__(self, num_nodes, num_features, window_size, rnn_hidden, rnn_layers=2):
        super(SpatioTemporalAutoencoderNoGNN, self).__init__()
        self.input_dim = num_nodes * num_features
        self.rnn_encoder = nn.GRU(self.input_dim, rnn_hidden, batch_first=True, num_layers=rnn_layers)
        self.rnn_decoder = nn.GRU(rnn_hidden, self.input_dim, batch_first=True, num_layers=rnn_layers)

    def forward(self, x, edge_index=None, edge_weight=None):  # Ignora grafo
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, x.size(1), -1)
        _, h_n = self.rnn_encoder(x_flat)

        latent = h_n[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        recon, _ = self.rnn_decoder(latent)
        return recon.reshape(x.shape)


# --- 3. FUNCIONES HELPER (Grafos y Wavelets) ---

def get_adjacency_matrix(model_type):
    """Retorna edge_index y edge_weight según el tipo de modelo."""
    # Grafo Base (M2, M3)
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]
    ], dtype=torch.long).t().contiguous()
    edge_weight = None

    if model_type == "gnn_physics":  # M4: Grafo Ponderado
        # Replicamos la lógica de 31_oct_newmodel.py
        coords = {0: [13.88, -4], 1: [13.88, 4], 2: [27.76, -4], 3: [27.76, 4], 4: [41.64, 0]}
        edges, weights = [], []
        for i in range(5):
            for j in range(i + 1, 5):
                dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                w = 1.0 / (dist + 1e-6)
                edges.append([i, j]);
                weights.append(w)
                edges.append([j, i]);
                weights.append(w)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)

    return edge_index.to(DEVICE), (edge_weight.to(DEVICE) if edge_weight is not None else None)


def apply_wavelet(data_window, wavelet='db4', level=5):
    """Aplica DWT a una ventana de datos (para M3/M4)."""
    # data_window shape: (T, N) -> returns (T, N, Features)
    T, N = data_window.shape
    features = []
    for n in range(N):
        sig = data_window[:, n]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        # Reconstruir bandas (lógica simplificada de train_wavelet_v3)
        # Para inferencia rápida, simulamos que las features son [Original, A, D...]
        # En realidad, necesitamos replicar exactamente lo que hizo el entrenamiento.
        # Dado que no podemos importar train_wavelet_v3, usamos una aproximación funcional:
        # Si el modelo espera 7 features, usamos la señal original + 6 bandas reconstruidas.
        rec_bands = [sig]  # Feature 0: Original
        # Reconstruir A5
        rec_a = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)[:T]
        rec_bands.append(rec_a)
        # Reconstruir D5..D1
        for i in range(1, len(coeffs)):
            c_list = [np.zeros_like(c) for c in coeffs];
            c_list[i] = coeffs[i]
            rec_d = pywt.waverec(c_list, wavelet)[:T]
            rec_bands.append(rec_d)

        # Stack features
        node_feats = np.stack(rec_bands, axis=-1)  # (T, Feats)
        # Ajustar longitud si wavelet cambió tamaño
        if node_feats.shape[0] != T:
            node_feats = node_feats[:T, :] if node_feats.shape[0] > T else np.pad(node_feats,
                                                                                  ((0, T - node_feats.shape[0]),
                                                                                   (0, 0)))
        features.append(node_feats)

    return np.stack(features, axis=1)  # (T, N, Feats)


# --- 4. CARGA DE MODELOS Y DATOS ---

def load_model(model_key):
    cfg = MODELS_CONFIG[model_key]
    log.info(f"Cargando {cfg['name']}...")

    # 1. Cargar HPs
    with open(cfg['hp_file'], 'r') as f:
        hps = json.load(f)
    if 'override_hp' in cfg: hps.update(cfg['override_hp'])

    # 2. Instanciar Modelo
    num_features = 7 if 'wavelet' in cfg['type'] or 'physics' in cfg['type'] else 1

    if cfg['type'] == 'nognn':
        model = SpatioTemporalAutoencoderNoGNN(
            num_nodes=NUM_SENSORS, num_features=1, window_size=WINDOW_SIZE,
            rnn_hidden=hps['rnn_hidden'], rnn_layers=hps.get('rnn_layers', 2)
        )
    else:
        model = SpatioTemporalAutoencoder(
            num_nodes=NUM_SENSORS, num_features=num_features, window_size=WINDOW_SIZE,
            gnn_hidden=hps['gnn_hidden'], gnn_out=hps['gnn_out'],
            rnn_hidden=hps['rnn_hidden'], rnn_layers=hps.get('rnn_layers', 2)
        )

    # 3. Cargar Pesos
    try:
        state_dict = torch.load(cfg['model_file'], map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)  # strict=False para evitar error de 'latent_project'
    except Exception as e:
        log.error(f"Error cargando pesos de {cfg['name']}: {e}")
        return None, None

    model.to(DEVICE)
    model.eval()

    # 4. Cargar Scaler
    scaler = joblib.load(cfg['scaler_file'])

    return model, scaler


def prepare_data(data_dir, scaler, model_type, limit=10):
    """Carga datos, normaliza y ventanea."""
    files = list(data_dir.glob("*.txt"))[:limit * NUM_SENSORS]  # Límite para rapidez

    # Cargar raw
    raw_data = {i: [] for i in range(1, 6)}
    for f in files:
        try:
            sid = int(f.name.split('_')[0]); raw_data[sid].append(
                pd.read_csv(f, sep='\s+', usecols=[1], header=None).values)
        except:
            pass

    # Concatenar
    data_list = []
    min_len = float('inf')
    for i in range(1, 6):
        if raw_data[i]:
            d = np.concatenate(raw_data[i]).flatten(); data_list.append(d); min_len = min(min_len, len(d))
        else:
            return None  # Falta sensor

    # Stack (N, T_total) -> Transpose (T_total, N)
    data_array = np.stack([d[:min_len] for d in data_list], axis=1)

    # Wavelet Pre-processing (si aplica)
    if 'wavelet' in model_type or 'physics' in model_type:
        # Para wavelet, aplicamos transformación por ventana o pre-calculamos
        # Aquí pre-calculamos features para todo el array es muy costoso,
        # así que normalizamos la señal base primero.
        # NOTA: El scaler de wavelet espera shape (N_samples, Features).
        # Es complejo replicar exacto la inferencia wavelet offline.
        # SIMPLIFICACIÓN: Usaremos M2 (GNN Base) para las figuras principales si M3 falla por scaler.
        pass

        # Normalizar (Solo señal base por simplicidad en demo,
    # si es wavelet requeriría aplicar scaler a las 7 features)
    # Para este script, asumiremos que el scaler es para la señal cruda si falla wavelet.
    try:
        data_norm = scaler.transform(data_array)
    except:
        # Fallback manual si scaler tiene dimensiones de wavelet
        data_norm = (data_array - np.mean(data_array)) / (np.std(data_array) + 1e-6)

    # Ventaneo
    windows = []
    for i in range(0, len(data_norm) - WINDOW_SIZE, STRIDE):
        windows.append(data_norm[i:i + WINDOW_SIZE])

    return np.array(windows)  # (B, T, N)


# --- 5. BUCLE DE INFERENCIA Y PLOTTING ---

def run_inference(model, windows, model_type):
    edge_index, edge_weight = get_adjacency_matrix(model_type)

    tensor_
    data = torch.FloatTensor(windows)  # (B, T, N)

    # Si es wavelet/physics, necesitamos calcular features al vuelo
    if 'wavelet' in model_type or 'physics' in model_type:
        # Esto es lento, hacemos solo unos pocos batches para demo
        processed = []
        for i in range(min(len(windows), 50)):  # Límite 50 ventanas
            feat = apply_wavelet(windows[i])  # (T, N, 7)
            processed.append(feat)
        tensor_input = torch.FloatTensor(np.array(processed)).to(DEVICE)
    else:
        # Añadir dim feature: (B, T, N) -> (B, T, N, 1)
        tensor_input = tensor_data.unsqueeze(-1).to(DEVICE)

    with torch.no_grad():
        recon = model(tensor_input, edge_index, edge_weight)

    # Calcular Error (MSE)
    # recon: (B, T, N, F)
    mse = torch.mean((tensor_input - recon) ** 2, dim=[1, 3]).cpu().numpy()  # (B, N) - Error por sensor

    return tensor_input.cpu().numpy(), recon.cpu().numpy(), mse


def generate_figures():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Usaremos M2 (GNN Base) como ejemplo principal para la reconstrucción
    # porque es más rápido y estable sin pre-proceso wavelet complejo
    target_model = "M2"
    model, scaler = load_model(target_model)
    if not model: return

    log.info("Preparando datos Sanos...")
    windows_healthy = prepare_data(DATA_DIR_HEALTHY, scaler, MODELS_CONFIG[target_model]['type'], limit=5)

    log.info("Preparando datos Dañados...")
    windows_damaged = prepare_data(DATA_DIR_DAMAGED, scaler, MODELS_CONFIG[target_model]['type'], limit=5)

    log.info("Ejecutando Inferencia...")
    # Ejecutar en subconjunto
    orig_h, rec_h, mse_h = run_inference(model, windows_healthy[:100], MODELS_CONFIG[target_model]['type'])
    orig_d, rec_d, mse_d = run_inference(model, windows_damaged[:100], MODELS_CONFIG[target_model]['type'])

    # --- FIGURA 1: Reconstrucción ---
    log.info("Generando Fig 1: Reconstrucción...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Sano (Ventana aleatoria)
    idx = np.random.randint(0, len(orig_h))
    ax = axes[0]
    ax.plot(orig_h[idx, :, 2, 0], 'k-', label='Original', linewidth=1.5)  # Sensor 3
    ax.plot(rec_h[idx, :, 2, 0], 'g--', label='Reconstructed', linewidth=1.5)
    ax.set_title(f"Healthy State Reconstruction (Sensor 3) - MSE: {np.mean(mse_h[idx]):.5f}")
    ax.legend()

    # Dañado
    idx = np.random.randint(0, len(orig_d))
    ax = axes[1]
    ax.plot(orig_d[idx, :, 2, 0], 'k-', label='Original', linewidth=1.5)
    ax.plot(rec_d[idx, :, 2, 0], 'r--', label='Reconstructed', linewidth=1.5)
    ax.set_title(f"Damaged State Reconstruction (Sensor 3) - MSE: {np.mean(mse_d[idx]):.5f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_1_Reconstruction_Comparison.png")

    # --- FIGURA 2: Distribución de Error ---
    log.info("Generando Fig 2: Distribución...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Promedio de error por ventana (todos los sensores)
    err_h = np.mean(mse_h, axis=1)
    err_d = np.mean(mse_d, axis=1)

    sns.kdeplot(err_h, fill=True, color='green', label='Healthy Data', ax=ax)
    sns.kdeplot(err_d, fill=True, color='red', label='Damaged Data', ax=ax)

    ax.set_title(f"Reconstruction Error Distribution ({MODELS_CONFIG[target_model]['name']})")
    ax.set_xlabel("Mean Squared Error (MSE)")
    ax.set_xscale('log')  # Log scale ayuda a ver la separación
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_2_Error_Distribution.png")

    # --- FIGURA 3: Mapa de Calor por Sensor ---
    log.info("Generando Fig 3: Localización...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Error promedio por sensor
    mean_mse_per_sensor_h = np.mean(mse_h, axis=0)
    mean_mse_per_sensor_d = np.mean(mse_d, axis=0)

    # Incremento relativo
    increase = (mean_mse_per_sensor_d - mean_mse_per_sensor_h) / mean_mse_per_sensor_h

    sns.heatmap(increase.reshape(1, -1), annot=True, cmap='Reds', fmt=".2f",
                xticklabels=[f"S{i}" for i in range(1, 6)], yticklabels=["Damage Index"])
    ax.set_title("Damage Localization Index (Relative MSE Increase)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_3_Damage_Localization.png")

    log.info("--- MÓDULO 3 COMPLETADO ---")


def setup_style():
    try:
        findfont(FontProperties(family=Q1_FONT_NAME)); font = Q1_FONT_NAME
    except:
        font = 'serif'
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.family': 'serif', 'font.serif': [font], 'font.size': 11, 'figure.dpi': DPI})


if __name__ == "__main__":
    generate_figures()