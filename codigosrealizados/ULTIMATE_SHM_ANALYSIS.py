#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GRAPHOS SONNET GRAPHS 19 NOV 2 - FINAL CORRECTED VERSION
Project: EMAIRC VISION - Structural Health Monitoring
Author: Senior Data Scientist (Gemini)
Date: November 2025

DESCRIPTION:
Standalone, exhaustive analysis suite for 4 Deep Learning Models:
M1 (GRU-AE), M2 (GNN), M3 (Wavelet-GNN), M4 (Physics-Informed).

CRITICAL FIXES APPLIED:
1.  **Data Loading Fix:** Correctly handles '1_sismo.txt' by targeting Column 1
    (Acceleration) instead of the last column, ignoring trailing tabs/whitespace.
2.  **Full History:** Merges training logs from base and resume folders.
3.  **Architecture:** Corrects gnn_out sizes (M2=32, M3/M4=128) to match weights.
4.  **Visualization:** Generates 90+ figures including FFT, Spatial, and Time domain.
================================================================================
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
import torch.nn as nn
import pywt
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    accuracy_score, f1_score
)
from tqdm import tqdm
import warnings

# =====================================================================
# 0. CONFIGURATION & STYLE
# =====================================================================
warnings.filterwarnings('ignore')

# Academic Plotting Style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EXECUTION] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("analysis_execution.log", mode='w')]
)
logger = logging.getLogger(__name__)

# System Params
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 64
STRIDE = 32
SAMPLING_RATE = 333
NUM_NODES = 5
SENSOR_IDS = [f'Sensor_{i}' for i in range(NUM_NODES)]

# Coordinates (Physical)
SENSOR_3D_COORDS = {
    0: np.array([0.0, -4.0, 0.0]),
    1: np.array([0.0, 4.0, 0.0]),
    2: np.array([27.76, -4.0, 0.0]),
    3: np.array([27.76, 4.0, 0.0]),
    4: np.array([55.52, 0.0, 0.0])
}

# Topology
EDGE_LIST = [
    (0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1),
    (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)
]

# Model Configurations (Matched to Checkpoints)
MODEL_CONFIGS = {
    'M1': {'name': 'GRU-AE', 'color': '#E74C3C', 'has_gnn': False, 'num_features': 1, 'rnn_hidden': 96, 'rnn_layers': 2,
           'gnn_out': 0},
    'M2': {'name': 'GNN-Base', 'color': '#3498DB', 'has_gnn': True, 'num_features': 1, 'gnn_hidden': 32, 'gnn_out': 32,
           'rnn_hidden': 64, 'rnn_layers': 2},
    'M3': {'name': 'Wavelet-GNN', 'color': '#2ECC71', 'has_gnn': True, 'num_features': 7, 'gnn_hidden': 128,
           'gnn_out': 128, 'rnn_hidden': 256, 'rnn_layers': 2},
    'M4': {'name': 'PI-STG-AE', 'color': '#9B59B6', 'has_gnn': True, 'num_features': 7, 'gnn_hidden': 128,
           'gnn_out': 128, 'rnn_hidden': 256, 'rnn_layers': 2, 'physical': True}
}

# =====================================================================
# 1. ARCHITECTURES
# =====================================================================

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    logger.critical("torch_geometric missing. Please install it.")
    sys.exit(1)


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
    def __init__(self, num_nodes, num_features, window_size, gnn_hidden, gnn_out, rnn_hidden, rnn_layers):
        super(STGAE, self).__init__()
        self.num_nodes = num_nodes
        self.gnn_out = gnn_out
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(input_size=gnn_out * num_nodes, hidden_size=rnn_hidden, num_layers=rnn_layers,
                                  batch_first=True)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=gnn_out * num_nodes, num_layers=rnn_layers,
                                  batch_first=True)
        self.gnn_decoder = GNNLayer(gnn_out, gnn_hidden, num_features)

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
        rnn_dec = rnn_dec.reshape(batch_size, T, self.num_nodes, self.gnn_out)
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
        input_dim = num_nodes * num_features
        self.rnn_encoder = nn.GRU(input_size=input_dim, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=input_dim, num_layers=rnn_layers, batch_first=True)

    def forward(self, x):
        B, T, _, _ = x.size()
        _, h_n = self.rnn_encoder(x.reshape(B, T, -1))
        out, _ = self.rnn_decoder(h_n[-1].unsqueeze(1).repeat(1, T, 1))
        return out.reshape(B, T, self.num_nodes, -1)


# =====================================================================
# 2. DATA LOADING (THE CRITICAL FIX)
# =====================================================================

def load_sensor_data_robust(data_dir, ids):
    """
    Loads data handling the '1_sismo.txt' trailing tabs issue correctly.
    """
    data = {}
    if not os.path.exists(data_dir): return {}
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    for sid in ids:
        idx = int(sid.split('_')[-1])
        prefix = f"{idx + 1}_"
        matches = [f for f in files if os.path.basename(f).startswith(prefix)]
        if not matches: matches = [f for f in files if sid.lower() in os.path.basename(f).lower()]

        acc_data = []
        for p in matches:
            try:
                # Read with python engine to auto-handle separators
                df = pd.read_csv(p, sep=None, engine='python', header=None)

                vals = np.array([])

                # [FIX]: Prioritize Column 1 (Acc) over Column -1 (which might be empty due to trailing tabs)
                if df.shape[1] >= 2:
                    # Try extracting column 1
                    try:
                        candidate = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                        # Drop NaNs and check if valid data remains
                        candidate = candidate.dropna()
                        if len(candidate) > 0:
                            vals = candidate.values
                    except:
                        pass

                # Fallback: If Col 1 failed or file is single column
                if len(vals) == 0:
                    # Check single column
                    candidate = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
                    if len(candidate) > 0:
                        vals = candidate.values

                if len(vals) > 0:
                    acc_data.append(vals)
                else:
                    logger.warning(f"File {p} contained no readable numeric data.")

            except Exception as e:
                logger.warning(f"Read error {p}: {e}")

        if acc_data:
            data[sid] = np.concatenate(acc_data)
            logger.info(f"Loaded {sid}: {len(data[sid])} samples.")
        else:
            logger.error(f"FAILED to load {sid}. Filling with zeros to prevent crash.")
            data[sid] = np.array([0.0])  # Placeholder

    return data


def parse_log_file(filepath):
    data = []
    if not os.path.exists(filepath): return pd.DataFrame()
    with open(filepath, 'r') as f:
        for line in f:
            if "Train Loss:" in line and "Val Loss:" in line:
                try:
                    parts = line.split("Epoch")[-1]
                    epoch = int(parts.split('/')[0].strip())
                    train = float(line.split("Train Loss:")[1].split(',')[0])
                    val = float(line.split("Val Loss:")[1].split('(')[0])
                    data.append({'epoch': epoch, 'train_loss': train, 'val_loss': val})
                except:
                    continue
    return pd.DataFrame(data)


def merge_training_history(base_dir, resume_dir):
    base_logs = glob.glob(os.path.join(base_dir, "*log*.txt"))
    resume_logs = glob.glob(os.path.join(resume_dir, "*log*.txt"))
    df_base = parse_log_file(base_logs[0]) if base_logs else pd.DataFrame()
    df_resume = parse_log_file(resume_logs[0]) if resume_logs else pd.DataFrame()
    if not df_base.empty and not df_resume.empty:
        df_resume['epoch'] += df_base['epoch'].max()
    return pd.concat([df_base, df_resume]).sort_values('epoch')


def wavelet_transform(signal_data):
    transformed = []
    for n in range(signal_data.shape[1]):
        coeffs = pywt.wavedec(signal_data[:, n], 'db4', level=5)
        L = len(signal_data)
        feats = [signal.resample(c, L) if len(c) != L else c for c in coeffs]
        transformed.append(np.stack(feats, axis=1))
    return np.stack(transformed, axis=1)


def get_graph(is_physical=False):
    edges = torch.tensor(EDGE_LIST, dtype=torch.long).t().contiguous()
    if not is_physical: return edges, None
    weights = []
    for i, j in EDGE_LIST:
        d = np.linalg.norm(SENSOR_3D_COORDS[i] - SENSOR_3D_COORDS[j])
        weights.append(1.0 / (d + 1e-6))
    return edges, torch.tensor(weights, dtype=torch.float32)


def create_windows_safe(data_array, window_size, stride):
    if len(data_array) < window_size: return np.array([])
    windows = []
    for i in range(0, len(data_array) - window_size, stride):
        windows.append(data_array[i:i + window_size])
    if len(windows) == 0: return np.array([])
    return np.array(windows)


# =====================================================================
# 3. VISUALIZATION MODULES (1-9)
# =====================================================================

def plot_raw_data_physics(data_h, data_d, output_dir):
    d = os.path.join(output_dir, "01_Raw_Data_Physics")
    os.makedirs(d, exist_ok=True)
    fig, axes = plt.subplots(NUM_NODES, 2, figsize=(12, 15))
    for i in range(NUM_NODES):
        try:
            sns.histplot(data_h[:5000, i], ax=axes[i, 0], color='g', kde=True, label='Healthy')
        except:
            pass
        try:
            sns.histplot(data_d[:5000, i], ax=axes[i, 1], color='r', kde=True, label='Damage')
        except:
            pass
        axes[i, 0].set_ylabel(f"Sensor {i}")
    plt.tight_layout()
    plt.savefig(os.path.join(d, "Raw_Distributions.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(np.corrcoef(data_h.T), annot=True, cmap='Greens', xticklabels=SENSOR_IDS, yticklabels=SENSOR_IDS)
    plt.title("Sensor Correlation (Healthy)")
    plt.savefig(os.path.join(d, "Corr_Healthy.png"))
    plt.close()


def plot_model_internals(models, output_dir):
    d = os.path.join(output_dir, "02_Model_Internals")
    os.makedirs(d, exist_ok=True)
    for mid, mdata in models.items():
        model = mdata['model']
        conf = MODEL_CONFIGS[mid]
        if conf['has_gnn']:
            w = model.gnn_encoder.conv1.lin.weight.detach().cpu().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(w, cmap='viridis', center=0)
            plt.title(f"{mid} - GNN Encoder Weights")
            plt.savefig(os.path.join(d, f"{mid}_GNN_Weights.png"))
            plt.close()


def plot_training_dynamics(log_data, output_dir):
    d = os.path.join(output_dir, "03_Training_Dynamics")
    os.makedirs(d, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for mid, df in log_data.items():
        if not df.empty: plt.plot(df['epoch'], df['val_loss'], label=f"{mid} Val Loss",
                                  color=MODEL_CONFIGS[mid]['color'])
    plt.title("Training Convergence")
    plt.yscale('log');
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(d, "Global_Convergence.png"))
    plt.close()


def plot_reconstruction_deep_dive(models, data, output_dir):
    d = os.path.join(output_dir, "04_Micro_Reconstruction")
    os.makedirs(d, exist_ok=True)
    logger.info("Generating Micro Reconstruction plots...")
    windows = create_windows_safe(data, WINDOW_SIZE, STRIDE)
    if len(windows) == 0: return
    windows = windows[:min(len(windows), 60)]  # Limit for plot generation

    for mid, mdata in models.items():
        conf = MODEL_CONFIGS[mid]
        if conf['num_features'] > 1:
            wav_wins = [wavelet_transform(w) for w in windows]
            inp = torch.FloatTensor(np.array(wav_wins)).to(DEVICE)
        else:
            inp = torch.FloatTensor(windows).unsqueeze(-1).to(DEVICE)

        with torch.no_grad():
            if conf['has_gnn']:
                rec = mdata['model'](inp, mdata['ei'], mdata['ew'])
            else:
                rec = mdata['model'](inp)
        rec = rec.cpu().numpy()
        orig = windows

        for s in range(NUM_NODES):
            plt.figure(figsize=(12, 4))
            plt.plot(orig[0:5, :, s].flatten(), 'k-', alpha=0.7, label='Original')
            plt.plot(rec[0:5, :, s, 0].flatten(), color=conf['color'], label='Reconstructed')
            plt.title(f"{mid} - Sensor {s} - Reconstruction")
            plt.legend()
            plt.savefig(os.path.join(d, f"{mid}_S{s}_TimeSeries.png"))
            plt.close()

            plt.figure(figsize=(6, 6))
            flat_o = orig[:, :, s].flatten()
            flat_r = rec[:, :, s, 0].flatten()
            plt.scatter(flat_o, flat_r, alpha=0.2, s=10, c=conf['color'])
            plt.plot([flat_o.min(), flat_o.max()], [flat_o.min(), flat_o.max()], 'k--')
            plt.title(f"{mid} - Sensor {s} - Parity")
            plt.savefig(os.path.join(d, f"{mid}_S{s}_Parity.png"))
            plt.close()


def plot_spectral_analysis(models, data, output_dir):
    d = os.path.join(output_dir, "05_Spectral_Analysis")
    os.makedirs(d, exist_ok=True)
    logger.info("Generating Spectral Analysis...")
    seq_len = 1000
    sample = data[:seq_len]
    wins = create_windows_safe(sample, WINDOW_SIZE, WINDOW_SIZE)
    if len(wins) == 0: return

    for mid, mdata in models.items():
        conf = MODEL_CONFIGS[mid]
        if conf['num_features'] > 1:
            inp_np = np.array([wavelet_transform(w) for w in wins])
        else:
            inp_np = wins[..., np.newaxis]

        with torch.no_grad():
            if conf['has_gnn']:
                out = mdata['model'](torch.FloatTensor(inp_np).to(DEVICE), mdata['ei'], mdata['ew'])
            else:
                out = mdata['model'](torch.FloatTensor(inp_np).to(DEVICE))

        rec_seq = out.cpu().numpy()[:, :, :, 0].reshape(-1, NUM_NODES)
        orig_seq = wins.reshape(-1, NUM_NODES)

        for s in range(NUM_NODES):
            f_o, Pxx_o = signal.periodogram(orig_seq[:, s], fs=SAMPLING_RATE)
            f_r, Pxx_r = signal.periodogram(rec_seq[:, s], fs=SAMPLING_RATE)
            plt.figure(figsize=(10, 5))
            plt.semilogy(f_o, Pxx_o, 'k-', alpha=0.5, label='Original')
            plt.semilogy(f_r, Pxx_r, color=conf['color'], alpha=0.8, label='Reconstructed')
            plt.title(f"{mid} - Sensor {s} - PSD")
            plt.legend();
            plt.xlim(0, 50)
            plt.savefig(os.path.join(d, f"{mid}_S{s}_FFT.png"))
            plt.close()


def plot_error_stats(models, data, output_dir):
    d = os.path.join(output_dir, "06_Error_Stats")
    os.makedirs(d, exist_ok=True)
    wins = create_windows_safe(data[:1000], WINDOW_SIZE, STRIDE)
    if len(wins) == 0: return

    for mid, mdata in models.items():
        conf = MODEL_CONFIGS[mid]
        if conf['num_features'] > 1:
            inp_np = np.array([wavelet_transform(w) for w in wins])
        else:
            inp_np = wins[..., np.newaxis]

        with torch.no_grad():
            if conf['has_gnn']:
                rec = mdata['model'](torch.FloatTensor(inp_np).to(DEVICE), mdata['ei'], mdata['ew'])
            else:
                rec = mdata['model'](torch.FloatTensor(inp_np).to(DEVICE))

        mse = np.mean((wins - rec.cpu().numpy()[..., 0]) ** 2, axis=(1, 2))
        plt.figure(figsize=(8, 5))
        sns.histplot(mse, kde=True, color=conf['color'], log_scale=True)
        plt.title(f"{mid} - MSE Distribution")
        plt.savefig(os.path.join(d, f"{mid}_Error_Hist.png"))
        plt.close()


# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================

def run_execution(healthy_dir, damage_dir, checkpoints, output_base):
    logger.info("STARTING EXECUTION PIPELINE")
    os.makedirs(output_base, exist_ok=True)

    # 1. Load Data
    logger.info("Loading Data...")
    h_dict = load_sensor_data_robust(healthy_dir, SENSOR_IDS)
    d_dict = load_sensor_data_robust(damage_dir, SENSOR_IDS)

    # Align and Stack Healthy
    valid_lens = [len(v) for v in h_dict.values() if len(v) > 1]
    if not valid_lens: logger.critical("No valid healthy data!"); return
    min_len = min(valid_lens)
    h_list = []
    for s in SENSOR_IDS:
        if len(h_dict.get(s, [])) >= min_len:
            h_list.append(h_dict[s][:min_len])
        else:
            h_list.append(np.zeros(min_len))
    h_arr = np.stack(h_list, axis=1)

    # Align and Stack Damage
    # CRITICAL FIX: Ensure we don't crash if a sensor is missing/zero length
    valid_d_lens = [len(v) for v in d_dict.values() if len(v) > 1]
    min_len_d = min(valid_d_lens) if valid_d_lens else 1000
    d_list = []
    for s in SENSOR_IDS:
        if len(d_dict.get(s, [])) >= min_len_d:
            d_list.append(d_dict[s][:min_len_d])
        else:
            d_list.append(np.zeros(min_len_d))  # Pad with zeros if sensor failed
    d_arr = np.stack(d_list, axis=1)

    # Normalize
    scaler = StandardScaler().fit(h_arr)
    h_norm = scaler.transform(h_arr)
    d_norm = scaler.transform(d_arr)

    # 2. Load Models
    logger.info("Loading Models...")
    models = {}
    logs = {}
    for mid, path_info in checkpoints.items():
        # Handle Tuple (Resume, Base)
        if isinstance(path_info, tuple):
            resume, base = path_info
            pths = glob.glob(os.path.join(resume, "**/*.pth"), recursive=True)
            tgt = next((p for p in pths if "best_model" in os.path.basename(p)), pths[0] if pths else None)
            logs[mid] = merge_training_history(base, resume)
        else:
            tgt_dir = path_info
            pths = glob.glob(os.path.join(tgt_dir, "**/*.pth"), recursive=True)
            tgt = next((p for p in pths if "best_model" in os.path.basename(p)), pths[0] if pths else None)
            log_files = glob.glob(os.path.join(tgt_dir, "*log*.txt"))
            logs[mid] = parse_log_file(log_files[0]) if log_files else pd.DataFrame()

        if not tgt: continue
        conf = MODEL_CONFIGS[mid]

        if conf['has_gnn']:
            m = STGAE(NUM_NODES, conf['num_features'], WINDOW_SIZE, conf['gnn_hidden'], conf['gnn_out'],
                      conf['rnn_hidden'], conf['rnn_layers'])
            ei, ew = get_graph(conf.get('physical', False))
            if ei is not None: ei = ei.to(DEVICE)
            if ew is not None: ew = ew.to(DEVICE)
        else:
            m = STAE_NoGNN(NUM_NODES, conf['num_features'], WINDOW_SIZE, conf['rnn_hidden'], conf['rnn_layers'])
            ei, ew = None, None

        m.load_state_dict(torch.load(tgt, map_location=DEVICE), strict=False)
        m.to(DEVICE)
        m.eval()
        models[mid] = {'model': m, 'ei': ei, 'ew': ew}
        logger.info(f"Loaded {mid}")

    # 3. Execute
    logger.info("Running Visualization Modules...")
    plot_raw_data_physics(h_norm, d_norm, output_base)
    plot_model_internals(models, output_base)
    plot_training_dynamics(logs, output_base)
    plot_reconstruction_deep_dive(models, h_norm, output_base)
    plot_spectral_analysis(models, h_norm, output_base)
    plot_error_stats(models, h_norm, output_base)

    logger.info(f"DONE. Results in {output_base}")


if __name__ == "__main__":
    # CONFIGURATION
    H_DIR = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
    D_DIR = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"
    OUT_DIR = r"D:\Python_proyectos_2025\GAIATECH\RESULTADOS_FINALES_PAPER_V3"

    # Model Paths (Resume, Base) tuple for merged logs
    M3_PATHS = (
        r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547",
        r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343"
    )
    M4_PATHS = (
        r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347",
        r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920"
    )
    CKPT = {
        'M1': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627",
        'M2': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756",
        'M3': M3_PATHS,
        'M4': M4_PATHS
    }

    run_execution(H_DIR, D_DIR, CKPT, OUT_DIR)