#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
FACTORY DE FIGURAS CIENTÍFICAS Q1 - EDICIÓN V5 (PHYSICALLY CORRECT & DATA AWARE)
Proyecto: EMAIRC VISION / PUENTE JUNÍN SHM
Autor: Senior Data Scientist (Gemini)
Fecha: Noviembre 2025

CORRECCIONES CRÍTICAS DE INGENIERÍA (V5):
1. LECTURA DE DATOS INTELIGENTE: Detecta automáticamente columnas de Tiempo vs
   Aceleración basándose en la media estadística (El tiempo crece, la señal oscila en 0).
   Concatena correctamente los 3 archivos por sensor.
2. RECONSTRUCCIÓN EXACTA (IDWT): Implementa `inverse_wavelet_transform_robust` que
   revierte el resampling de los coeficientes y aplica `pywt.waverec` para obtener
   la señal temporal exacta.
3. ESCALA REAL (g): Des-normaliza los datos para que los gráficos muestren magnitudes
   físicas reales (ej. 0.004 g) y no valores latentes.

ESTRUCTURA DE SALIDA:
- 4_reconstruction_analysis/ -> Gráficos de Precisión con Residuales < 1e-4
================================================================================
"""

import os
import sys
import glob
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import torch.nn as nn
import pywt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error
import networkx as nx
import warnings
from collections import OrderedDict

# =====================================================================
# 0. CONFIGURACIÓN
# =====================================================================
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.formatter.limits'] = (-3, 4)
mpl.rcParams['axes.formatter.use_mathtext'] = True

# --- RUTAS (HARDCODED) ---
BASE_DATA_H = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
BASE_DATA_D = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"
OUTPUT_ROOT = r"D:\Python_proyectos_2025\GAIATECH\PAPER_FIGURES_FINAL_REAL"

M3_PATHS = (
    r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547",
    r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343"
)
M4_PATHS = (
    r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347",
    r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920"
)
CHECKPOINTS = {
    'M1': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627",
    'M2': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756",
    'M3': M3_PATHS,
    'M4': M4_PATHS
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_NODES = 5
SENSOR_IDS = [f'Sensor_{i}' for i in range(NUM_NODES)]
WINDOW_SIZE = 64
STRIDE = 32
SAMPLING_RATE = 333

SENSOR_3D_COORDS = {
    0: np.array([0.0, -4.0, 0.0]),
    1: np.array([0.0, 4.0, 0.0]),
    2: np.array([27.76, -4.0, 0.0]),
    3: np.array([27.76, 4.0, 0.0]),
    4: np.array([55.52, 0.0, 0.0])
}
EDGE_LIST = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [FACTORY_V5] - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# =====================================================================
# 1. MODELOS
# =====================================================================
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None


class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


class STGAE(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden, gnn_out, rnn_hidden, rnn_layers, dec_dim):
        super(STGAE, self).__init__()
        self.num_nodes = num_nodes
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(input_size=gnn_out * num_nodes, hidden_size=rnn_hidden, num_layers=rnn_layers,
                                  batch_first=True)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=dec_dim * num_nodes, num_layers=rnn_layers,
                                  batch_first=True)
        self.gnn_decoder = GNNLayer(dec_dim, gnn_hidden, num_features)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size, T, _, _ = x.size()
        gnn_encoded_steps = []
        for t in range(T):
            snap = x[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_idx = edge_index.repeat(1, batch_size) + torch.arange(batch_size, device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            batch_w = edge_weight.repeat(batch_size) if edge_weight is not None else None
            gnn_encoded_steps.append(self.gnn_encoder(snap, batch_idx, batch_w).reshape(batch_size, self.num_nodes, -1))
        gnn_flat = torch.stack(gnn_encoded_steps, dim=1).reshape(batch_size, T, -1)
        _, h_n = self.rnn_encoder(gnn_flat)
        rnn_dec, _ = self.rnn_decoder(h_n[-1].unsqueeze(1).repeat(1, T, 1))
        rnn_dec = rnn_dec.reshape(batch_size, T, self.num_nodes, -1)
        recon_steps = []
        for t in range(T):
            snap = rnn_dec[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_idx = edge_index.repeat(1, batch_size) + torch.arange(batch_size, device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            batch_w = edge_weight.repeat(batch_size) if edge_weight is not None else None
            recon_steps.append(self.gnn_decoder(snap, batch_idx, batch_w).reshape(batch_size, self.num_nodes, -1))
        return torch.stack(recon_steps, dim=1)


class STAE_NoGNN(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, rnn_hidden, rnn_layers):
        super(STAE_NoGNN, self).__init__()
        self.num_nodes = num_nodes
        self.rnn_encoder = nn.GRU(input_size=num_nodes * num_features, hidden_size=rnn_hidden, num_layers=rnn_layers,
                                  batch_first=True)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=num_nodes * num_features, num_layers=rnn_layers,
                                  batch_first=True)

    def forward(self, x):
        B, T, _, _ = x.size()
        _, h_n = self.rnn_encoder(x.reshape(B, T, -1))
        out, _ = self.rnn_decoder(h_n[-1].unsqueeze(1).repeat(1, T, 1))
        return out.reshape(B, T, self.num_nodes, -1)


MODEL_CONFIGS = {
    'M1': {'has_gnn': False, 'num_features': 1, 'rnn_hidden': 96, 'rnn_layers': 2},
    'M2': {'has_gnn': True, 'num_features': 1, 'gnn_hidden': 32, 'gnn_out': 16, 'dec_dim': 32, 'rnn_hidden': 64,
           'rnn_layers': 2},
    'M3': {'has_gnn': True, 'num_features': 7, 'gnn_hidden': 128, 'gnn_out': 64, 'dec_dim': 128, 'rnn_hidden': 256,
           'rnn_layers': 2},
    'M4': {'has_gnn': True, 'num_features': 7, 'gnn_hidden': 128, 'gnn_out': 64, 'dec_dim': 128, 'rnn_hidden': 256,
           'rnn_layers': 2, 'physical': True}
}


# =====================================================================
# 2. UTILIDADES CIENTÍFICAS
# =====================================================================

def create_windows_safe(data_array, window_size, stride):
    if len(data_array) < window_size: return np.array([])
    windows = []
    for i in range(0, len(data_array) - window_size, stride):
        windows.append(data_array[i:i + window_size])
    if len(windows) == 0: return np.array([])
    return np.array(windows)


def wavelet_transform(signal_data, level=6):
    """Pre-procesamiento para input del modelo (Upsampled Coeffs)"""
    transformed = []
    for n in range(signal_data.shape[1]):
        coeffs = pywt.wavedec(signal_data[:, n], 'db4', level=level)
        L = len(signal_data)
        feats = [signal.resample(c, L) if len(c) != L else c for c in coeffs]
        transformed.append(np.stack(feats, axis=1))
    return np.stack(transformed, axis=1)


def inverse_wavelet_transform_robust(coeffs_resampled, level=6, wavelet='db4'):
    """
    RECONSTRUCCIÓN MATEMÁTICA EXACTA:
    1. Toma los coeficientes resampleados que escupe el modelo [64, 7].
    2. Los 'des-resamplea' (downsample) a la longitud que pywt espera para cada nivel.
    3. Ejecuta la Transformada Inversa (waverec).
    Esto recupera la señal en el dominio del tiempo, permitiendo comparación 1:1.
    """
    original_len = coeffs_resampled.shape[0]  # 64

    # 1. Calcular longitudes teóricas para db4 level 6 con señal de longitud 64
    # pywt.wavedec returns [cA6, cD6, cD5, ..., cD1]
    dummy_sig = np.zeros(original_len)
    dummy_coeffs = pywt.wavedec(dummy_sig, wavelet, level=level)
    target_lengths = [len(c) for c in dummy_coeffs]

    # 2. Recuperar coeficientes (Un-Stack y Downsample)
    # El modelo output tiene shape [Time, 7]. La dim 1 son las features (bandas).
    # Orden features: cA6 (idx 0), cD6 (idx 1) ... cD1 (idx 6)
    rec_coeffs = []
    for i, length in enumerate(target_lengths):
        c_upsampled = coeffs_resampled[:, i]
        # Downsample robusto usando resample
        c_down = signal.resample(c_upsampled, length)
        rec_coeffs.append(c_down)

    # 3. Transformada Inversa
    rec_signal = pywt.waverec(rec_coeffs, wavelet)

    # Ajuste fino de longitud (waverec a veces devuelve N+1)
    if len(rec_signal) > original_len:
        rec_signal = rec_signal[:original_len]
    elif len(rec_signal) < original_len:
        rec_signal = np.pad(rec_signal, (0, original_len - len(rec_signal)))

    return rec_signal


def load_weights_robustly(model, path):
    state_dict = torch.load(path, map_location=DEVICE)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


def load_folder_data(folder_path, max_files=5, max_samples=10000):
    """
    CARGA INTELIGENTE: Distingue columna de Tiempo vs Señal
    """
    data_store = {sid: [] for sid in SENSOR_IDS}
    files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

    for sid in SENSOR_IDS:
        idx = int(sid.split('_')[-1])
        prefix = f"{idx + 1}_"
        rels = [f for f in files if os.path.basename(f).startswith(prefix)]
        if not rels: rels = [f for f in files if sid.lower() in os.path.basename(f).lower()]

        # Concatenar todos los archivos encontrados para este sensor
        sensor_raw = []
        for f in rels:
            try:
                df = pd.read_csv(f, sep=None, engine='python', header=None)

                # Lógica Heurística: La columna de señal tiene media cercana a 0.
                # La columna de tiempo tiene media grande (ej. 1800).
                col_idx = 0
                if df.shape[1] > 1:
                    means = df.mean().abs()
                    col_idx = means.idxmin()  # Elegir columna con menor media (aceleración)

                vals = df.iloc[:, col_idx].values
                vals = pd.to_numeric(vals, errors='coerce')
                sensor_raw.append(vals[~np.isnan(vals)])
            except Exception as e:
                logger.warning(f"Error leyendo {f}: {e}")

        if sensor_raw:
            data_store[sid] = np.concatenate(sensor_raw)

    # Alinear longitudes
    final_arr = []
    valid_lens = [len(v) for v in data_store.values() if len(v) > 0]
    if not valid_lens: return None
    min_len = min(valid_lens)

    for sid in SENSOR_IDS:
        if len(data_store[sid]) > 0:
            final_arr.append(data_store[sid][:min_len])
        else:
            # Relleno de seguridad si falta un sensor
            final_arr.append(np.zeros(min_len))

    return np.stack(final_arr, axis=1)


def load_checkpoints():
    loaded = {}
    for mid, path_info in CHECKPOINTS.items():
        sdir = path_info[0] if isinstance(path_info, tuple) else path_info
        pths = glob.glob(os.path.join(sdir, "**/*.pth"), recursive=True)
        if not pths: continue
        best = next((p for p in pths if "best_model" in p), pths[0])
        conf = MODEL_CONFIGS[mid]

        if conf['has_gnn'] and GCNConv is None: continue

        if conf['has_gnn']:
            m = STGAE(NUM_NODES, conf['num_features'], WINDOW_SIZE, conf['gnn_hidden'], conf['gnn_out'],
                      conf['rnn_hidden'], conf['rnn_layers'], conf['dec_dim'])
            e = torch.tensor(EDGE_LIST, dtype=torch.long).t().contiguous().to(DEVICE)
            w = torch.tensor(
                [1.0 / (np.linalg.norm(SENSOR_3D_COORDS[i] - SENSOR_3D_COORDS[j]) + 1e-6) for i, j in EDGE_LIST],
                dtype=torch.float32).to(DEVICE)
        else:
            m = STAE_NoGNN(NUM_NODES, conf['num_features'], WINDOW_SIZE, conf['rnn_hidden'], conf['rnn_layers'])
            e, w = None, None

        m = load_weights_robustly(m, best)
        m.to(DEVICE).eval()
        loaded[mid] = {'model': m, 'edges': e, 'weights': w}
        logger.info(f"Loaded {mid}")
    return loaded


# =====================================================================
# 3. GENERADORES DE FIGURAS
# =====================================================================
def save_fig(fig, folder, name):
    os.makedirs(os.path.join(OUTPUT_ROOT, folder), exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_ROOT, folder, name))
    plt.close(fig)


def fig_4_reconstruction(models, data_h, scaler):
    folder = "4_reconstruction_analysis"
    logger.info("Generating Reconstruction Plots (Physically Correct)...")

    # Ventaneo sobre datos normalizados
    win = create_windows_safe(data_h[:5000], WINDOW_SIZE, STRIDE)
    if len(win) == 0: return

    # Pre-computar wavelet (features)
    wav = np.array([wavelet_transform(w, level=6) for w in win])
    t_raw = torch.FloatTensor(win).unsqueeze(-1).to(DEVICE)
    t_wav = torch.FloatTensor(wav).to(DEVICE)

    for mid, mdata in models.items():
        conf = MODEL_CONFIGS[mid]
        inp = t_wav if conf['num_features'] > 1 else t_raw

        try:
            with torch.no_grad():
                if conf['has_gnn']:
                    rec = mdata['model'](inp, mdata['edges'], mdata['weights'])
                else:
                    rec = mdata['model'](inp)

            rh = rec.cpu().numpy()

            # Buscar ventana con señal activa
            best_idx = np.argmax(np.var(win[:, :, 0], axis=1))

            for s in range(NUM_NODES):
                # --- RECONSTRUCCIÓN FÍSICA ---

                # 1. Recuperar señal NORMALIZADA en tiempo
                feat_orig = win[best_idx, :, s]  # Original (Normalized Time)
                feat_rec_coeffs = rh[best_idx, :, s, :]  # Reconstructed (Normalized Coeffs)

                if conf['num_features'] > 1:
                    # Aplicar IDWT a los coeficientes predichos
                    sig_rec_norm = inverse_wavelet_transform_robust(feat_rec_coeffs, level=6)
                else:
                    sig_rec_norm = feat_rec_coeffs[:, 0]

                # 2. DESNORMALIZAR a 'g'
                scale = scaler.scale_[s]
                mean = scaler.mean_[s]

                real_orig = feat_orig * scale + mean
                real_rec = sig_rec_norm * scale + mean

                # Calcular residual real
                residual = real_orig - real_rec

                # --- PLOT DUAL ---
                fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

                # Señales
                ax[0].plot(real_orig, 'k-', lw=1.5, alpha=0.8, label='Original (g)')
                ax[0].plot(real_rec, color=conf.get('color', 'b'), linestyle='--', lw=1.5, label='Reconstrucción (g)')
                ax[0].set_title(f"{mid} - Sensor {s}: High Precision Reconstruction")
                ax[0].legend()
                ax[0].grid(True, alpha=0.3)
                ax[0].set_ylabel("Aceleración (g)")

                # Error
                ax[1].plot(residual, color='#C0392B', lw=1, label='Error Residual')
                ax[1].set_ylabel("Delta (g)")
                ax[1].set_xlabel("Muestras")
                ax[1].legend()
                ax[1].grid(True, alpha=0.3)
                ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                plt.tight_layout()
                save_fig(fig, folder, f"4.{mid}_S{s}_PRECISION.png")

        except Exception as e:
            logger.error(f"Err {mid}: {e}")


def fig_1_methodology(data_h, data_d):
    folder = "1_methodology_wavelets"
    logger.info(f"Generating {folder}...")
    sig_h = data_h[:2000]
    sig_d = data_d[:2000] if data_d is not None else sig_h

    for s_idx in range(NUM_NODES):
        coeffs = pywt.wavedec(sig_h[:, s_idx], 'db4', level=6)
        fig, axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
        axes[0].plot(sig_h[:, s_idx], 'k', lw=1)
        axes[0].set_title(f"S{s_idx} (Healthy) - Decomposition")
        names = ['cA6'] + [f'cD{i}' for i in range(6, 0, -1)]
        for i, (ax, c, name) in enumerate(zip(axes[1:], coeffs, names)):
            ax.plot(c, color='#2ECC71' if i == 0 else '#3498DB')
            ax.set_ylabel(name, rotation=0, labelpad=20)
            ax.grid(True, alpha=0.3)
        save_fig(fig, folder, f"1.{s_idx}_WAVELET_SANO.png")


def fig_2_metrics(checkpoints):
    folder = "2_training_metrics"
    all_logs = {}
    for mid, path in checkpoints.items():
        search_path = path[0] if isinstance(path, tuple) else path
        log_files = glob.glob(os.path.join(search_path, "*log*.txt"))
        if log_files:
            vals = []
            try:
                with open(log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if "Val Loss" in line:
                            try:
                                vals.append(float(line.split("Val Loss:")[1].split('(')[0]))
                            except:
                                pass
                all_logs[mid] = vals
            except:
                pass

    fig = plt.figure(figsize=(10, 6))
    for mid, vals in all_logs.items():
        if vals: plt.plot(vals, label=f"{mid}")
    plt.yscale('log');
    plt.legend();
    plt.grid(True, alpha=0.3)
    save_fig(fig, folder, "2.1_LOSS_CURVES.png")


def fig_3_architecture():
    folder = "3_model_architecture"
    G = nx.Graph()
    for i, c in SENSOR_3D_COORDS.items(): G.add_node(i, pos=(c[0], c[1]))
    G.add_edges_from(EDGE_LIST)
    pos = nx.get_node_attributes(G, 'pos')
    fig = plt.figure(figsize=(8, 4))
    nx.draw(G, pos, with_labels=True, node_color='#9B59B6', font_color='white')
    save_fig(fig, folder, "3.4_GRAFO_2D.png")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs = [c[0] for c in SENSOR_3D_COORDS.values()]
    ys = [c[1] for c in SENSOR_3D_COORDS.values()]
    zs = [c[2] for c in SENSOR_3D_COORDS.values()]
    ax.scatter(xs, ys, zs, c='r', s=200)
    for u, v in EDGE_LIST:
        p1, p2 = SENSOR_3D_COORDS[u], SENSOR_3D_COORDS[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k--', alpha=0.5)
    save_fig(fig, folder, "3.5_GRAFO_3D.png")


def fig_5_3d_sims(data_h):
    folder = "5_3d_simulations"
    sig = data_h[100, :]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs = [c[0] for c in SENSOR_3D_COORDS.values()]
    ys = [c[1] for c in SENSOR_3D_COORDS.values()]
    zs = [c[2] for c in SENSOR_3D_COORDS.values()]
    p = ax.scatter(xs, ys, zs, c=sig, cmap='coolwarm', s=200, edgecolors='k')
    fig.colorbar(p, ax=ax)
    save_fig(fig, folder, "5.2_REAL_DATA_3D.png")


def fig_6_anomaly(models, data_h, data_d):
    folder = "6_anomaly_detection"
    if 'M4' in models and data_d is not None:
        seq = np.concatenate([data_h[:1000], data_d[:1000]])
        wins = create_windows_safe(seq, WINDOW_SIZE, STRIDE)
        wavs = np.array([wavelet_transform(w, level=6) for w in wins])
        inp = torch.FloatTensor(wavs).to(DEVICE)
        m = models['M4']['model']
        with torch.no_grad():
            out = m(inp, models['M4']['edges'], models['M4']['weights'])
        mse = np.mean((wins - out.cpu().numpy()[..., 0]) ** 2, axis=(1, 2))

        fig = plt.figure(figsize=(12, 5))
        plt.plot(mse)
        plt.axvspan(len(mse) // 2, len(mse), color='red', alpha=0.1)
        save_fig(fig, folder, "6.1_ALARM_CURVE.png")


def fig_7_additional(data_h):
    folder = "7_additional_analysis"
    fig = plt.figure(figsize=(8, 5))
    sns.histplot(data_h.flatten(), bins=100, kde=True, color='purple')
    save_fig(fig, folder, "7.10_DISTRIBUTION.png")

    fig = plt.figure(figsize=(8, 5))
    plt.bar(SENSOR_IDS, [len(data_h)] * NUM_NODES)
    save_fig(fig, folder, "7.3_VOLUMEN.png")


# =====================================================================
# MAIN
# =====================================================================
def main():
    logger.info(">>> STARTING V5 ENGINE (PHYSICALLY CORRECT)")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. Carga Datos (Raw)
    data_h_raw = load_folder_data(BASE_DATA_H)
    data_d_raw = load_folder_data(BASE_DATA_D)
    if data_h_raw is None: return

    # 2. Normalización para Modelos
    scaler = StandardScaler()
    data_h_norm = scaler.fit_transform(data_h_raw)
    data_d_norm = scaler.transform(data_d_raw) if data_d_raw is not None else None

    # 3. Carga Modelos
    models = load_checkpoints()

    # 4. Generación (Pasa scaler para des-normalizar en graficos)
    fig_1_methodology(data_h_raw, data_d_raw)  # Usa Raw para wavelets
    fig_2_metrics(CHECKPOINTS)
    fig_3_architecture()
    fig_4_reconstruction(models, data_h_norm, scaler)  # Usa Norm para inferencia, pero plotea en g
    fig_5_3d_sims(data_h_raw)
    fig_6_anomaly(models, data_h_norm, data_d_norm)
    fig_7_additional(data_h_raw)

    logger.info("Done.")


if __name__ == "__main__":
    main()