#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
COMPREHENSIVE ANALYSIS SCRIPT FOR SHM BOWSTRING BRIDGE - CORE LIBRARY
Multi-Model Comparison: M1 (No-GNN), M2 (GNN-Base), M3 (Wavelet-GNN), M4 (PI-STG-AE)
================================================================================
Author: Research Team (Enhanced by Gemini Senior Engineer)
Date: November 2025
Purpose: Generate 90+ scientific figures for Q1 publication in Structures

CRITICAL FORENSIC CORRECTIONS (Nov 19, 2025):
1.  **M2 Architecture Fix:** Checkpoint forensic analysis reveals M2 was trained
    with gnn_out=32 (not 16). Config updated to match weights [480, 64].
2.  **M3/M4 Architecture Fix:** Checkpoint analysis reveals these were trained
    with gnn_out=128 (not 64). Config updated to match weights [1920, 256].
3.  **Data Loading:** Added 'Engineering Pessimism' logic to handle:
    - File prefix mapping (Sensor_0 -> 1_*.txt)
    - Corrupt/irregular separators in '1_sismo.txt'
4.  **Full Code Restoration:** All visualization modules (1-3) and documentation
    restored to original verbose state.
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
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
import torch.nn as nn
import pywt
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, confusion_matrix, precision_recall_curve
)
import joblib
import json
from datetime import datetime
from tqdm import tqdm
import warnings

# Filter warnings for cleaner output in scientific reports
warnings.filterwarnings('ignore')

# =====================================================================
# 0. SYSTEM CONFIGURATION & ROBUST LOGGING
# =====================================================================

# Configure logging to write to both file and console
# This ensures we capture data loading errors for audit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("shm_analysis_core_full.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Set academic plotting style (Times New Roman, High DPI)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# Device configuration (GPU prioritization)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using computation device: {DEVICE}")

# Output directory placeholder
OUTPUT_DIR = "/mnt/user-data/outputs"
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception:
        OUTPUT_DIR = "./outputs"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 1. MODEL CONFIGURATION (CORRECTED FROM CHECKPOINT ANALYSIS)
# =====================================================================

MODEL_CONFIGS = {
    'M1': {
        'name': 'No-GNN Baseline',
        'color': '#E74C3C',  # Red
        'marker': 'o',
        'linestyle': '--',
        'has_gnn': False,
        'num_features': 1,
        'rnn_hidden': 96,
        'rnn_layers': 2,
        'checkpoint_dir': None,
        'best_val_loss': 0.4773
    },
    'M2': {
        'name': 'GNN-Base',
        'color': '#3498DB',  # Blue
        'marker': 's',
        'linestyle': '-.',
        'has_gnn': True,
        'num_features': 1,
        'gnn_hidden': 32,
        # [CRITICAL FIX]: Checkpoint weights [480, 64] imply gnn_out=32.
        # (32 features * 5 nodes * 3 GRU gates = 480)
        'gnn_out': 32,
        'rnn_hidden': 64,
        'rnn_layers': 2,
        'checkpoint_dir': None,
        'best_val_loss': 0.0218
    },
    'M3': {
        'name': 'Wavelet-GNN',
        'color': '#2ECC71',  # Green
        'marker': '^',
        'linestyle': '-',
        'has_gnn': True,
        'num_features': 7,  # db4 level 5 decomposition
        'gnn_hidden': 128,
        # [CRITICAL FIX]: Checkpoint weights [1920, 256] imply gnn_out=128.
        # (128 features * 5 nodes * 3 GRU gates = 1920)
        'gnn_out': 128,
        'rnn_hidden': 256,
        'rnn_layers': 2,
        'wavelet': 'db4',
        'wavelet_level': 5,
        'checkpoint_dir': None,
        'split_training': True,
        'best_val_loss': 0.0064
    },
    'M4': {
        'name': 'PI-STG-AE (Physics-Informed)',
        'color': '#9B59B6',  # Purple
        'marker': 'D',
        'linestyle': '-',
        'has_gnn': True,
        'num_features': 7,  # db4 level 5
        'gnn_hidden': 128,
        # [CRITICAL FIX]: Same as M3, physics model uses wider latent GNN
        'gnn_out': 128,
        'rnn_hidden': 256,
        'rnn_layers': 2,
        'wavelet': 'db4',
        'wavelet_level': 5,
        'physical_graph': True,  # Uses Euclidean distance weights
        'checkpoint_dir': None,
        'split_training': True,
        'best_val_loss': 0.0084
    }
}

# Sensor configuration (Bowstring Bridge)
NUM_NODES = 5
SENSOR_IDS = [f'Sensor_{i}' for i in range(NUM_NODES)]

# 3D Coordinates (Origin: Miraflores end E1, center of deck)
SENSOR_3D_COORDS = {
    0: np.array([0.0, -4.0, 0.0]),  # E1 - Viga 1 (Miraflores)
    1: np.array([0.0, 4.0, 0.0]),  # E1 - Viga 2
    2: np.array([27.76, -4.0, 0.0]),  # E3 - Viga 1 (Centro)
    3: np.array([27.76, 4.0, 0.0]),  # E3 - Viga 2
    4: np.array([55.52, 0.0, 0.0])  # E5 - Centro Tablero (SJ)
}

# Graph topology (Physical connections)
EDGE_LIST = [
    (0, 1), (1, 0),  # E1 transverse
    (0, 2), (2, 0),  # Viga 1 longitudinal
    (1, 3), (3, 1),  # Viga 2 longitudinal
    (2, 3), (3, 2),  # E3 transverse
    (2, 4), (4, 2),  # E3-E5 Viga 1
    (3, 4), (4, 3)  # E3-E5 Viga 2
]

# Window configuration
WINDOW_SIZE = 64
STRIDE = 32
SAMPLING_RATE = 333  # Hz


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def create_physics_informed_graph(coords_dict):
    """
    Create physically weighted graph based on 3D Euclidean distances.
    Innovation of M4 model.

    Weights are calculated as Inverse Euclidean Distance.
    """
    edge_index = []
    edge_weights = []

    for i, j in EDGE_LIST:
        coord_i = coords_dict[i]
        coord_j = coords_dict[j]

        # Euclidean distance
        dist = np.linalg.norm(coord_i - coord_j)

        # Weight: Inverse distance (higher weight = closer nodes)
        weight = 1.0 / (dist + 1e-6)  # Epsilon to prevent div/0

        edge_index.append([i, j])
        edge_weights.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    return edge_index, edge_weight


def create_adjacency_graph():
    """Standard adjacency graph (M2, M3) - Binary connections"""
    edge_index = torch.tensor(EDGE_LIST, dtype=torch.long).t().contiguous()
    return edge_index


def extract_wavelet_features(signal_1d, wavelet='db4', level=5):
    """
    Extract wavelet coefficients using DWT.
    Used for M3 and M4 input processing.

    Args:
        signal_1d: Input time series
        wavelet: Wavelet family (default 'db4')
        level: Decomposition level

    Returns:
        features: np.array of shape (len(signal), 7) for level=5
    """
    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)

    # Pad or truncate to match signal length
    target_len = len(signal_1d)
    features = []

    for c in coeffs:
        if len(c) < target_len:
            # Upsample/pad
            c_resampled = signal.resample(c, target_len)
        else:
            c_resampled = c[:target_len]
        features.append(c_resampled)

    # Stack: (target_len, num_coeffs)
    features = np.column_stack(features)
    return features


def merge_training_logs(base_log_path, resume_log_path):
    """
    Merge training logs from base and resume runs to reconstruct full history.
    """

    def parse_log(log_path):
        epochs = []
        train_losses = []
        val_losses = []

        if not os.path.exists(log_path):
            logger.warning(f"Log file not found: {log_path}")
            return pd.DataFrame()

        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Epoch' in line and 'Train Loss:' in line and 'Val Loss:' in line:
                    try:
                        parts = line.split('Epoch')[1].split('->')
                        epoch_str = parts[0].split('/')[0].strip()

                        train_part = line.split('Train Loss:')[1].split(',')[0].strip()
                        val_part = line.split('Val Loss:')[1].split('(')[0].strip()

                        epochs.append(int(epoch_str))
                        train_losses.append(float(train_part))
                        val_losses.append(float(val_part))
                    except ValueError:
                        continue
        return pd.DataFrame({'epoch': epochs, 'train_loss': train_losses, 'val_loss': val_losses})

    base_df = parse_log(base_log_path)
    resume_df = parse_log(resume_log_path)

    if base_df.empty: return resume_df
    if resume_df.empty: return base_df

    merged_df = pd.concat([base_df, resume_df], ignore_index=True)
    merged_df = merged_df.sort_values('epoch').reset_index(drop=True)
    return merged_df


# =====================================================================
# 3. MODEL ARCHITECTURES
# =====================================================================

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    logger.error("torch_geometric not installed. Install via: pip install torch_geometric")
    sys.exit(1)


class GNNLayer(nn.Module):
    """GCN block with 2 convolutional layers"""

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
    """
    Spatio-Temporal Graph Autoencoder (M2, M3, M4)
    Combines GCN for spatial features and GRU for temporal correlations.
    """

    def __init__(self, num_nodes, num_features, window_size,
                 gnn_hidden, gnn_out, rnn_hidden, rnn_layers):
        super(STGAE, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.gnn_hidden = gnn_hidden
        self.gnn_out = gnn_out

        # Encoder
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(
            input_size=gnn_out * num_nodes,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True
        )

        # Decoder
        self.rnn_decoder = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=gnn_out * num_nodes,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.gnn_decoder = GNNLayer(gnn_out, gnn_hidden, num_features)

    def forward(self, x, edge_index, edge_weight=None):
        batch_size = x.size(0)
        T = x.size(1)

        # 1. Spatial Encoding
        gnn_encoded_steps = []
        for t in range(T):
            snapshot = x[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + \
                               torch.arange(batch_size, device=x.device).repeat_interleave(
                                   edge_index.size(1)) * self.num_nodes

            if edge_weight is not None:
                batch_edge_weight = edge_weight.repeat(batch_size)
            else:
                batch_edge_weight = None

            gnn_out = self.gnn_encoder(snapshot, batch_edge_index, batch_edge_weight)
            gnn_encoded_steps.append(gnn_out.reshape(batch_size, self.num_nodes, -1))

        gnn_encoded = torch.stack(gnn_encoded_steps, dim=1)
        gnn_flat = gnn_encoded.reshape(batch_size, T, -1)

        # 2. Temporal Encoding
        _, h_n = self.rnn_encoder(gnn_flat)

        # 3. Temporal Decoding
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, T, 1)
        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        rnn_decoded = rnn_decoded.reshape(batch_size, T, self.num_nodes, self.gnn_out)

        # 4. Spatial Reconstruction
        reconstructed_steps = []
        for t in range(T):
            snapshot = rnn_decoded[:, t, :, :].reshape(batch_size * self.num_nodes, -1)

            batch_edge_index = edge_index.repeat(1, batch_size) + \
                               torch.arange(batch_size, device=x.device).repeat_interleave(
                                   edge_index.size(1)) * self.num_nodes

            if edge_weight is not None:
                batch_edge_weight = edge_weight.repeat(batch_size)
            else:
                batch_edge_weight = None

            reconstructed = self.gnn_decoder(snapshot, batch_edge_index, batch_edge_weight)
            reconstructed_steps.append(reconstructed.reshape(batch_size, self.num_nodes, -1))

        return torch.stack(reconstructed_steps, dim=1)


class STAE_NoGNN(nn.Module):
    """
    Temporal-only autoencoder (No GNN). Used for M1.
    Flattens spatial dimensions.
    """

    def __init__(self, num_nodes, num_features, window_size, rnn_hidden, rnn_layers):
        super(STAE_NoGNN, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        input_dim = num_nodes * num_features

        self.rnn_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.rnn_decoder = nn.GRU(
            input_size=rnn_hidden,
            hidden_size=input_dim,
            num_layers=rnn_layers,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.size(0)
        T = x.size(1)
        x_flat = x.reshape(batch_size, T, -1)

        _, h_n = self.rnn_encoder(x_flat)
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, T, 1)
        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        reconstructed = rnn_decoded.reshape(batch_size, T, self.num_nodes, -1)

        return reconstructed


# =====================================================================
# 4. DATA LOADING & PREPROCESSING (ROBUST ENGINEERED)
# =====================================================================

def load_sensor_data(data_dir, sensor_ids):
    """
    Load raw sensor data robustly, handling file name variations and formats.

    [ENGINEERING PESSIMISM UPDATE]:
    Handles the disconnect between 'Sensor_0' (Code) and '1_*.txt' (File).
    Also handles potential delimiter corruption in files like '1_sismo.txt'.
    """
    sensor_data = {}
    logger.info(f"Scanning directory for data: {data_dir}")

    if not os.path.exists(data_dir):
        logger.error(f"CRITICAL: Data directory not found: {data_dir}")
        return {}

    # Find all txt files
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    logger.info(f"Found {len(all_files)} .txt files in total.")

    for sensor_id in sensor_ids:
        # 1. MAP CODE SENSOR ID TO FILE PREFIX
        # User files use 1-based indexing (1_, 2_, 3_...)
        # Code uses 0-based indexing (Sensor_0, Sensor_1...)
        try:
            sensor_idx = int(sensor_id.split('_')[-1])  # e.g., 0
            file_prefix = f"{sensor_idx + 1}_"  # e.g., 1_
        except ValueError:
            logger.error(f"Could not parse sensor index from {sensor_id}")
            continue

        # 2. FILTER FILES MATCHING PREFIX
        # Must start with prefix to avoid "11_" matching "1_"
        matched_files = [f for f in all_files if os.path.basename(f).startswith(file_prefix)]

        # Fallback to string matching if prefix fails
        if not matched_files:
            matched_files = [f for f in all_files if sensor_id.lower() in os.path.basename(f).lower()]

        if not matched_files:
            logger.warning(f"No files found for {sensor_id} (Prefix: '{file_prefix}'). Skipping.")
            continue

        logger.info(f"Loading {sensor_id} -> Matched {len(matched_files)} files (Prefix: '{file_prefix}')")

        sensor_accumulated_data = []

        for file_path in matched_files:
            try:
                # 3. ROBUST READING STRATEGY
                # Attempt A: Standard Python engine (auto-detects comma/tab)
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python', header=None)
                except Exception:
                    df = pd.DataFrame()  # Fail state

                # Attempt B: Force whitespace separator if A produced 1 messy column
                if not df.empty and df.shape[1] == 1 and df.iloc[0, 0] and isinstance(df.iloc[0, 0], str):
                    logger.info(f"Detected unparsed content in {file_path}, retrying with whitespace delimiter...")
                    df = pd.read_csv(file_path, sep='\s+', engine='python', header=None)

                # 4. EXTRACT DATA COLUMN
                # User data sometimes has Time (Col 0) and Acc (Col 1)
                # Or just Acc (Col 0).
                # Strategy: Take the LAST column as acceleration.
                if not df.empty:
                    if df.shape[1] >= 2:
                        data_vals = df.iloc[:, -1].values
                    else:
                        data_vals = df.iloc[:, 0].values

                    # Clean non-numeric
                    data_vals = pd.to_numeric(data_vals, errors='coerce')
                    data_vals = data_vals[~np.isnan(data_vals)]

                    if len(data_vals) > 0:
                        sensor_accumulated_data.append(data_vals)
                    else:
                        logger.warning(f"File {file_path} contained no valid numeric data.")

            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")

        if sensor_accumulated_data:
            full_sensor_data = np.concatenate(sensor_accumulated_data)
            sensor_data[sensor_id] = full_sensor_data
            logger.info(f"Successfully loaded {sensor_id}: {len(full_sensor_data)} samples.")
        else:
            logger.warning(f"Failed to load any valid data for {sensor_id}")

    return sensor_data


def preprocess_data(sensor_data, scaler=None, fit_scaler=False):
    """
    Preprocess: Align lengths and normalize.
    """
    if not sensor_data:
        return np.array([]), None

    min_len = min(len(data) for data in sensor_data.values())
    logger.info(f"Aligning all sensors to minimum length: {min_len}")

    data_list = []
    for sensor_id in SENSOR_IDS:
        if sensor_id in sensor_data:
            data_list.append(sensor_data[sensor_id][:min_len])
        else:
            logger.warning(f"{sensor_id} missing data, filling with zeros.")
            data_list.append(np.zeros(min_len))

    data_array = np.column_stack(data_list)

    if fit_scaler:
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(data_array)
        logger.info("Fitted new scaler.")
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided if fit_scaler=False")
        data_normalized = scaler.transform(data_array)
        logger.info("Transformed using existing scaler.")

    return data_normalized, scaler


def create_windows(data_array, window_size, stride):
    """Sliding window creation"""
    if len(data_array) == 0:
        return np.array([])

    num_windows = (len(data_array) - window_size) // stride + 1
    if num_windows <= 0:
        return np.array([])

    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows.append(data_array[start:end])

    return np.array(windows)


# =====================================================================
# 5. INFERENCE ROUTINE
# =====================================================================

def run_inference(model, data_windows, edge_index=None, edge_weight=None,
                  batch_size=32, device='cpu', has_gnn=True):
    """
    Run inference on data windows.
    Returns reconstructions and MSE per window.
    """
    model.eval()
    model = model.to(device)

    if edge_index is not None: edge_index = edge_index.to(device)
    if edge_weight is not None: edge_weight = edge_weight.to(device)

    num_windows = data_windows.shape[0]
    if num_windows == 0:
        return np.array([]), np.array([])

    num_batches = (num_windows + batch_size - 1) // batch_size

    all_reconstructions = []
    all_mse = []

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Running inference"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_windows)

            batch_data = data_windows[start_idx:end_idx]
            batch_tensor = torch.FloatTensor(batch_data).to(device)

            if has_gnn:
                reconstructed = model(batch_tensor, edge_index, edge_weight)
            else:
                reconstructed = model(batch_tensor)

            # MSE over features and nodes (dim 1=Time, 2=Nodes, 3=Features)
            # Shape of batch_tensor: [Batch, Time, Nodes, Features]
            mse = torch.mean((batch_tensor - reconstructed) ** 2, dim=(1, 2, 3))

            all_reconstructions.append(reconstructed.cpu().numpy())
            all_mse.append(mse.cpu().numpy())

    reconstructions = np.concatenate(all_reconstructions, axis=0)
    mse_per_window = np.concatenate(all_mse, axis=0)

    return reconstructions, mse_per_window


# =====================================================================
# 6. MODULES 1-3 (STATIC VISUALIZATION)
# =====================================================================

def generate_module1_wavelets(output_dir):
    """Module 1: Wavelet methodology visualization"""
    print("\n[MODULE 1] Generating wavelet methodology figures...")
    logger.info("Starting Module 1 generation")
    module_dir = os.path.join(output_dir, "M1_Methodology")
    os.makedirs(module_dir, exist_ok=True)

    # Synthetic signal
    t = np.linspace(0, 1, 1024, endpoint=False)
    signal_demo = (np.sin(2 * np.pi * 10 * t) +
                   0.5 * np.sin(2 * np.pi * 30 * t) +
                   0.3 * np.sin(2 * np.pi * 60 * t))
    signal_demo += 0.1 * np.random.randn(len(t))

    # Fig 1a: CWT
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, signal_demo, 'k-', linewidth=0.5)
    axes[0].set_title('(a) Raw acceleration signal')
    scales = np.arange(1, 128)
    coefficients, _ = pywt.cwt(signal_demo, scales, 'morl', sampling_period=1 / 333)
    im = axes[1].imshow(np.abs(coefficients), extent=[0, 1, 1, 128],
                        cmap='jet', aspect='auto', vmax=np.percentile(np.abs(coefficients), 98))
    axes[1].set_title('(b) Continuous Wavelet Transform (Morlet)')
    plt.colorbar(im, ax=axes[1], label='|Coefficient|')
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, "Fig1_CWT_Spectrogram.png"))
    plt.close()

    # Fig 1b: DWT
    wavelet = 'db4'
    level = 5
    coeffs = pywt.wavedec(signal_demo, wavelet, level=level)
    fig, axes = plt.subplots(level + 2, 1, figsize=(10, 10))
    axes[0].plot(t, signal_demo, 'k-', linewidth=0.5)
    axes[0].set_ylabel('Original')
    axes[0].set_title('(a) DWT Decomposition (db4, Level 5)')
    for i, coeff in enumerate(coeffs):
        axes[i + 1].plot(np.linspace(0, 1, len(coeff)), coeff, linewidth=0.5)
        axes[i + 1].set_ylabel(f'cA{level}' if i == 0 else f'cD{level - i + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, "Fig1_DWT_Decomposition.png"))
    plt.close()
    print(f"[MODULE 1] Saved figures to {module_dir}")


def generate_module2_training_metrics(model_configs, output_dir):
    """Module 2: Training loss curves and convergence analysis"""
    print("\n[MODULE 2] Generating training metrics figures...")
    logger.info("Starting Module 2 generation")
    module_dir = os.path.join(output_dir, "M2_Training_Metrics")
    os.makedirs(module_dir, exist_ok=True)

    # Fig 2b: Comparative Barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [config['name'] for config in model_configs.values()]
    val_losses = [config['best_val_loss'] for config in model_configs.values()]
    colors = [config['color'] for config in model_configs.values()]

    bars = ax.bar(range(len(model_names)), val_losses, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Best Validation Loss (MSE)')
    ax.set_title('Model Performance Comparison')
    ax.set_yscale('log')
    for bar, val in zip(bars, val_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, "Fig2_BestValidationLoss_Comparison.png"))
    plt.close()
    print(f"[MODULE 2] Saved figures to {module_dir}")


def generate_module3_graph_topology(output_dir):
    """Module 3: Graph visualization"""
    print("\n[MODULE 3] Generating graph topology figures...")
    logger.info("Starting Module 3 generation")
    module_dir = os.path.join(output_dir, "M3_Graph_Topology")
    os.makedirs(module_dir, exist_ok=True)

    # Fig 3a: 2D Topology
    fig, ax = plt.subplots(figsize=(10, 6))
    for node_id, coord in SENSOR_3D_COORDS.items():
        ax.scatter(coord[0], coord[1], s=300, c='lightblue', edgecolor='black', linewidth=2, zorder=3)
        ax.text(coord[0], coord[1], f'S{node_id}', ha='center', va='center', fontweight='bold')
    for i, j in EDGE_LIST:
        p1, p2 = SENSOR_3D_COORDS[i], SENSOR_3D_COORDS[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.5, zorder=1)
    ax.set_title('(a) 2D Graph Topology')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(module_dir, "Fig3_Graph_2D_Adjacency.png"))
    plt.close()

    # Fig 3b: 3D Physical Graph
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        edge_index, edge_weight = create_physics_informed_graph(SENSOR_3D_COORDS)
        weights_norm = (edge_weight.numpy() - edge_weight.numpy().min()) / \
                       (edge_weight.numpy().max() - edge_weight.numpy().min())

        coords = np.array([SENSOR_3D_COORDS[i] for i in range(NUM_NODES)])
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=300, c='red', edgecolor='black', zorder=3)

        for idx in range(edge_index.shape[1]):
            i, j = edge_index[:, idx].numpy()
            p1, p2 = SENSOR_3D_COORDS[i], SENSOR_3D_COORDS[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=plt.cm.viridis(weights_norm[idx]),
                    linewidth=1 + 3 * weights_norm[idx], alpha=0.7)
        ax.set_title('(b) 3D Physics-Informed Graph')
        plt.tight_layout()
        plt.savefig(os.path.join(module_dir, "Fig3_Graph_3D_PhysicsInformed.png"))
        plt.close()
    except Exception as e:
        logger.warning(f"3D plotting failed: {e}")

    print(f"[MODULE 3] Saved figures to {module_dir}")


# =====================================================================
# MAIN EXECUTION BLOCK
# =====================================================================

def main():
    """
    Main execution block to ensure library can be run standalone
    for Modules 1-3 generation.
    """
    print("=" * 80)
    print("COMPREHENSIVE SHM ANALYSIS - CORE LIBRARY STANDALONE EXECUTION")
    print("=" * 80)

    # Run Modules 1, 2, 3
    generate_module1_wavelets(OUTPUT_DIR)
    generate_module2_training_metrics(MODEL_CONFIGS, OUTPUT_DIR)
    generate_module3_graph_topology(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("LIBRARY DIAGNOSTICS COMPLETE")
    print("To run full analysis with data (Modules 4-7), execute:")
    print("python graphos_sonnet_graphs_19_nov_2.py")
    print("=" * 80)


if __name__ == "__main__":
    main()