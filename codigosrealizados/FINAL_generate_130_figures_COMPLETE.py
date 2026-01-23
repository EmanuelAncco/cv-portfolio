#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANÁLISIS DE RECONSTRUCCIÓN - INFERENCIA REAL (FINAL - 4 MODELOS)
Structural Health Monitoring - Puente Junín
Autor: Senior Data Scientist (Gemini)

MEJORAS CRÍTICAS:
1. RESTAURACIÓN M2 (GNN Base): Se agregó el modelo STG-AE estándar (Input=1).
2. NORMALIZACIÓN CORRECTA: Se aplica StandardScaler a la serie completa, no por ventana.
   (Esto soluciona el error de MSE > 20,000).
3. EVIDENCIA Q1: Estilo de gráficas ajustado a 'SciencePlots' (o similar).
================================================================================
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

# Intentar importar PyG, manejo robusto
try:
    from torch_geometric.nn import GCNConv

    TORCH_GEOM_AVAILABLE = True
except ImportError:
    TORCH_GEOM_AVAILABLE = False
    print("[WARNING] torch_geometric not available. GNN layers will fail if weights require GCNConv.")

import pywt
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURACIÓN ESTÉTICA (PAPER Q1)
# =====================================================================

# Estilo profesional
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.autolayout'] = True

BASE_DIR = r"D:\Python_proyectos_2025\GAIATECH"
OUTPUT_DIR = os.path.join(BASE_DIR, "inference_analysis_COMPLETE_4MODELS")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_HEALTHY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

# =====================================================================
# 2. DEFINICIÓN DE MODELOS (M1 a M4)
# =====================================================================
# Rutas basadas en tu historial y archivos subidos

MODEL_CONFIGS = {
    'M1_No_GNN': {  # Antes M2 en tu script, pero es el Baseline Temporal
        'checkpoint': os.path.join(BASE_DIR,
                                   r"resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627\best_model_no_gnn.pth"),
        'color': '#E74C3C',  # Rojo
        'label': 'M1: GRU-AE (No-GNN)',
        'type': 'no_gnn',
        'params': {'rnn_hidden': 96, 'rnn_layers': 2, 'num_features': 1}
    },
    'M2_GNN_Base': {  # EL MODELO QUE FALTABA
        'checkpoint': os.path.join(BASE_DIR, r"resultados_entrenamiento\run_gnn_20250910-020756\best_model.pth"),
        'color': '#3498DB',  # Azul
        'label': 'M2: STG-AE (Standard)',
        'type': 'gnn_base',
        'params': {'gnn_hidden': 32, 'gnn_out': 16, 'rnn_hidden': 64, 'rnn_layers': 2, 'num_features': 1}
    },
    'M3_Wavelet_GNN': {
        'checkpoint': os.path.join(BASE_DIR,
                                   r"resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547\best_model_wavelet_gnn.pth"),
        'color': '#2ECC71',  # Verde
        'label': 'M3: Wavelet-GNN',
        'type': 'wavelet',
        'params': {'gnn_hidden': 128, 'gnn_out': 64, 'rnn_hidden': 256, 'rnn_layers': 2, 'num_features': 7}
    },
    'M4_PI_STG_AE': {
        'checkpoint': os.path.join(BASE_DIR,
                                   r"resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347\best_model_stgae_physics.pth"),
        'color': '#9B59B6',  # Morado
        'label': 'M4: Physics-Informed',
        'type': 'wavelet',  # Usa arquitectura wavelet (7 feats)
        'params': {'gnn_hidden': 128, 'gnn_out': 64, 'rnn_hidden': 256, 'rnn_layers': 2, 'num_features': 7}
    }
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SENSORS = 5
WINDOW_SIZE = 64
SAMPLING_RATE = 333
NUM_RANDOM_WINDOWS = 100  # Muestras para estadística

# Geometría Física (Para M4 y visualización)
SENSOR_3D_COORDS = np.array([
    [0.0, -4.0, 0.0], [0.0, 4.0, 0.0],
    [27.76, -4.0, 0.0], [27.76, 4.0, 0.0],
    [55.52, 0.0, 0.0]
])
EDGE_INDEX = torch.tensor([
    [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
    [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]
], dtype=torch.long).t().contiguous()

# Pesos físicos
distances = pdist(SENSOR_3D_COORDS)
dist_matrix = squareform(distances)
edge_weights = []
for i in range(EDGE_INDEX.shape[1]):
    src, dst = EDGE_INDEX[0, i].item(), EDGE_INDEX[1, i].item()
    edge_weights.append(1.0 / (dist_matrix[src, dst] + 1e-6))
EDGE_WEIGHT = torch.tensor(edge_weights, dtype=torch.float32)

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Output: {OUTPUT_DIR}")


# =====================================================================
# 3. ARQUITECTURAS DE RED
# =====================================================================

class GNNLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        if TORCH_GEOM_AVAILABLE:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.01)
        self.use_pyg = TORCH_GEOM_AVAILABLE

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.use_pyg and edge_index is not None:
            x = self.conv1(x, edge_index, edge_weight)
            x = self.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        return x


class STAE_NoGNN(nn.Module):
    """M1: No-GNN"""

    def __init__(self, num_nodes=5, num_features=1, window_size=64, rnn_hidden=96, rnn_layers=2):
        super().__init__()
        self.input_size = num_nodes * num_features
        self.encoder = nn.GRU(self.input_size, rnn_hidden, batch_first=True, num_layers=rnn_layers)
        self.decoder = nn.GRU(rnn_hidden, self.input_size, batch_first=True, num_layers=rnn_layers)
        self.num_nodes = num_nodes
        self.num_features = num_features

    def forward(self, x):
        B, T, N, F = x.shape
        # Flatten nodes into features
        x_flat = x.reshape(B, T, -1)
        _, h = self.encoder(x_flat)
        latent = h[-1].unsqueeze(1).repeat(1, T, 1)
        out, _ = self.decoder(latent)
        return out.reshape(B, T, N, F), h


class STGAE_Standard(nn.Module):
    """M2: GNN Standard (1 feature, sin wavelets)"""

    def __init__(self, num_nodes=5, num_features=1, gnn_hidden=32, gnn_out=16, rnn_hidden=64, rnn_layers=2):
        super().__init__()
        self.gnn_enc = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_enc = nn.GRU(gnn_out * num_nodes, rnn_hidden, batch_first=True, num_layers=rnn_layers)
        self.rnn_dec = nn.GRU(rnn_hidden, gnn_hidden * num_nodes, batch_first=True, num_layers=rnn_layers)
        self.gnn_dec = GNNLayer(gnn_hidden, gnn_hidden, num_features)
        self.num_nodes = num_nodes
        self.gnn_out = gnn_out
        self.gnn_hidden = gnn_hidden

    def forward(self, x, edge_index, edge_weight=None):
        B, T, N, F = x.shape
        # GNN Encoding per timestep
        x_flat = x.reshape(B * T, N, F)
        g_enc = self.gnn_enc(x_flat, edge_index, edge_weight)
        g_enc = g_enc.reshape(B, T, -1)

        # RNN
        _, h = self.rnn_enc(g_enc)
        latent = h[-1].unsqueeze(1).repeat(1, T, 1)
        rnn_out, _ = self.rnn_dec(latent)

        # GNN Decoding
        rnn_out = rnn_out.reshape(B * T, N, self.gnn_hidden)
        out = self.gnn_dec(rnn_out, edge_index, edge_weight)
        return out.reshape(B, T, N, F), h


class WaveletGNN(nn.Module):
    """M3/M4: Wavelet-GNN (7 features)"""

    def __init__(self, num_features=7, gnn_hidden=128, gnn_out=64, rnn_hidden=256, num_layers=2, num_nodes=5):
        super().__init__()
        self.gnn_enc = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_enc = nn.GRU(gnn_out * num_nodes, rnn_hidden, batch_first=True, num_layers=num_layers)
        # Decoder output matches GNN hidden dim for reconstruction
        self.rnn_dec = nn.GRU(rnn_hidden, gnn_hidden * num_nodes, batch_first=True, num_layers=num_layers)
        self.gnn_dec = GNNLayer(gnn_hidden, gnn_hidden, num_features)
        self.num_nodes = num_nodes
        self.gnn_out = gnn_out
        self.gnn_hidden = gnn_hidden

    def forward(self, x, edge_index, edge_weight=None):
        B, T, N, F = x.shape
        x_flat = x.reshape(B * T, N, F)
        g_enc = self.gnn_enc(x_flat, edge_index, edge_weight)
        g_enc = g_enc.reshape(B, T, -1)

        _, h = self.rnn_enc(g_enc)
        latent = h[-1].unsqueeze(1).repeat(1, T, 1)
        rnn_out, _ = self.rnn_dec(latent)

        rnn_out = rnn_out.reshape(B * T, N, self.gnn_hidden)
        out = self.gnn_dec(rnn_out, edge_index, edge_weight)
        return out.reshape(B, T, N, F), h


# =====================================================================
# 4. PROCESAMIENTO DE DATOS (CORREGIDO)
# =====================================================================

def load_and_process_data():
    print("\n[PHASE 1] Loading and Normalizing Data (GLOBAL)...")
    sensor_data_raw = {}

    # 1. Cargar Datos Crudos
    for i in range(NUM_SENSORS):
        pattern = os.path.join(DATA_HEALTHY, f"{i + 1}_*.txt")
        files = glob.glob(pattern)
        if files:
            # Tomamos 2-3 archivos para tener suficiente data
            acc = []
            for f in sorted(files)[:3]:
                try:
                    df = pd.read_csv(f, sep=None, engine='python', header=None)
                    col = 1 if df.shape[1] > 1 else 0
                    val = pd.to_numeric(df.iloc[:, col], errors='coerce').dropna().values
                    acc.append(val)
                except:
                    pass
            if acc:
                sensor_data_raw[i] = np.concatenate(acc)

    # 2. Filtrado y Normalización GLOBAL
    # IMPORTANTE: Ajustamos el scaler a toda la señal, no por ventanas
    # Esto asegura que las ventanas mantengan su amplitud relativa
    sensor_data_norm = {}

    # Concatenar todos los sensores para un scaler global (o por sensor, usaremos por sensor)
    for i, data in sensor_data_raw.items():
        # Filtro
        b, a = signal.butter(4, [0.3 / 166.5, 25 / 166.5], btype='band')
        filtered = signal.filtfilt(b, a, data)

        # Scaling
        scaler = StandardScaler()
        norm = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
        sensor_data_norm[i] = norm
        print(f"  Sensor {i}: {len(norm)} samples (Normalized)")

    return sensor_data_norm


def apply_dwt_features(window, wavelet='db4', level=5):
    """Extrae features Wavelet para una ventana (64, ) -> (64, 7)"""
    coeffs = pywt.wavedec(window, wavelet, level=level)
    feats = []

    # Reconstruir aproximación y detalles a la longitud original (64)
    # A5
    a5 = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)
    feats.append(signal.resample(a5, len(window)))

    # D5 a D1
    for i in range(1, len(coeffs)):
        d_coeffs = [np.zeros_like(c) for c in coeffs]
        d_coeffs[i] = coeffs[i]
        d = pywt.waverec(d_coeffs, wavelet)
        feats.append(signal.resample(d, len(window)))

    # Stack: [Original, A5, D5, D4, D3, D2, D1] -> Total 7 (si original se incluye explícitamente)
    # En tu entrenamiento anterior incluías la original + 6 bandas = 7
    feats = [window] + feats  # Original + 6 bandas
    # Ajuste: wavedec level 5 da 6 coeficientes (A5, D5, D4, D3, D2, D1).

    return np.stack(feats[:7], axis=-1)  # Asegurar 7


def create_datasets(sensor_data_norm, num_windows=100):
    print(f"\n[PHASE 2] Creating {num_windows} test windows...")
    windows = []
    wavelet_windows = []

    min_len = min([len(v) for v in sensor_data_norm.values()])
    indices = np.random.choice(range(min_len - WINDOW_SIZE), num_windows, replace=False)

    for idx in tqdm(indices):
        win_std = np.zeros((WINDOW_SIZE, NUM_SENSORS))
        win_wav = np.zeros((WINDOW_SIZE, NUM_SENSORS, 7))

        for s in range(NUM_SENSORS):
            segment = sensor_data_norm[s][idx:idx + WINDOW_SIZE]
            win_std[:, s] = segment

            # Wavelets sobre el segmento ya normalizado
            w_feats = apply_dwt_features(segment)
            # Asegurar dimensiones
            if w_feats.shape[1] > 7: w_feats = w_feats[:, :7]
            if w_feats.shape[0] != WINDOW_SIZE: w_feats = signal.resample(w_feats, WINDOW_SIZE)

            win_wav[:, s, :] = w_feats

        windows.append(win_std)
        wavelet_windows.append(win_wav)

    return np.array(windows), np.array(wavelet_windows)


# =====================================================================
# 5. CARGA Y EJECUCIÓN
# =====================================================================

def load_and_infer(models_conf, windows_std, windows_wav):
    print("\n[PHASE 3] Loading Models & Inferring...")
    results = {}

    # Grafos a GPU
    ei = EDGE_INDEX.to(DEVICE)
    ew = EDGE_WEIGHT.to(DEVICE)

    for name, conf in models_conf.items():
        print(f"  Processing {name}...")

        # 1. Instanciar
        p = conf['params']
        try:
            if conf['type'] == 'no_gnn':
                model = STAE_NoGNN(num_features=p['num_features'], rnn_hidden=p['rnn_hidden'])
            elif conf['type'] == 'gnn_base':
                model = STGAE_Standard(gnn_hidden=p['gnn_hidden'], gnn_out=p['gnn_out'], rnn_hidden=p['rnn_hidden'])
            else:  # Wavelet
                model = WaveletGNN(num_features=p['num_features'], gnn_hidden=p['gnn_hidden'],
                                   rnn_hidden=p['rnn_hidden'])

            # 2. Cargar Pesos
            if not os.path.exists(conf['checkpoint']):
                print(f"    [SKIP] Checkpoint not found: {conf['checkpoint']}")
                continue

            ckpt = torch.load(conf['checkpoint'], map_location=DEVICE)
            # Manejo de diccionarios vs state_dict puro
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            state = ckpt['state_dict'] if 'state_dict' in ckpt else state

            model.load_state_dict(state, strict=False)
            model.to(DEVICE).eval()

            # 3. Inferencia
            preds = []
            errors = []

            # Seleccionar input correcto
            is_wav = (conf['type'] == 'wavelet')
            data_in = windows_wav if is_wav else windows_std

            with torch.no_grad():
                for i in range(len(data_in)):
                    # Prepare tensor
                    sample = data_in[i]
                    tensor = torch.FloatTensor(sample).unsqueeze(0).to(DEVICE)  # [1, 64, 5, F] or [1, 64, 5]

                    if not is_wav and conf['type'] != 'no_gnn' and tensor.ndim == 3:
                        tensor = tensor.unsqueeze(-1)  # [1, 64, 5, 1] for GNN Base
                    elif conf['type'] == 'no_gnn' and tensor.ndim == 3:
                        tensor = tensor.unsqueeze(-1)

                    # Forward
                    if conf['type'] == 'no_gnn':
                        rec, _ = model(tensor)
                    else:
                        rec, _ = model(tensor, ei, ew)

                    # Calcular Error (Siempre contra la señal original Feature 0)
                    rec_np = rec.cpu().numpy()[0, :, :, 0]  # [64, 5]
                    orig_np = windows_std[i]  # Siempre comparamos contra la señal standard

                    mse = np.mean((orig_np - rec_np) ** 2)
                    preds.append(rec_np)
                    errors.append(mse)

            results[name] = {
                'recons': np.array(preds),
                'errors': np.array(errors),
                'conf': conf
            }
            print(f"    ✓ MSE: {np.mean(errors):.4f}")

        except Exception as e:
            print(f"    [ERROR] Failed to load/infer {name}: {e}")
            import traceback
            traceback.print_exc()

    return results


# =====================================================================
# 6. GRAFICACIÓN
# =====================================================================

def plot_results(results, windows_gt, output_dir):
    print("\n[PHASE 4] Plotting...")

    # 1. Comparativa Visual (Reconstrucción) - Estilo Paper
    # Plotear 3 ejemplos
    for i in range(3):
        fig, axes = plt.subplots(len(results), NUM_SENSORS, figsize=(15, 2.5 * len(results)), sharex=True)

        # Asegurar indexing 2D
        if len(results) == 1: axes = np.expand_dims(axes, 0)

        for r_idx, (name, res) in enumerate(results.items()):
            rec = res['recons'][i]
            gt = windows_gt[i]

            for s in range(NUM_SENSORS):
                ax = axes[r_idx, s]

                # Plot
                ax.plot(gt[:, s], 'k-', lw=1, alpha=0.6, label='Real')
                ax.plot(rec[:, s], color=res['conf']['color'], lw=1.5, ls='--', label='Pred')

                # Estilizado
                if r_idx == 0: ax.set_title(f"Sensor {s + 1}")
                if s == 0: ax.set_ylabel(f"{res['conf']['label']}\nAmp (norm)")

                ax.grid(True, alpha=0.2, ls=':')
                # Solo leyenda en el último
                if r_idx == 0 and s == NUM_SENSORS - 1:
                    ax.legend(frameon=True, fontsize=8)

        plt.suptitle(f"Reconstruction Performance - Sample {i + 1}", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"RECON_Paper_Sample_{i + 1}.png"), bbox_inches='tight')
        plt.close()

    # 2. Boxplot de Errores
    plt.figure(figsize=(10, 6))
    data = [res['errors'] for res in results.values()]
    labels = [res['conf']['label'] for res in results.values()]
    colors = [res['conf']['color'] for res in results.values()]

    bplot = plt.boxplot(data, patch_artist=True, labels=labels, showfliers=False)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.title("Distribution of Reconstruction Errors (MSE) - Validation Set")
    plt.ylabel("Mean Squared Error (Log Scale)")
    plt.yscale('log')
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.savefig(os.path.join(output_dir, "ERROR_DISTRIBUTION_BOXPLOT.png"))
    plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():
    # 1. Datos
    raw_norm = load_and_process_data()
    win_std, win_wav = create_datasets(raw_norm, NUM_RANDOM_WINDOWS)

    # 2. Inferencia
    results = load_and_infer(MODEL_CONFIGS, win_std, win_wav)

    # 3. Reporte
    plot_results(results, win_std, OUTPUT_DIR)
    print("\n✅ DONE. Check output directory.")


if __name__ == "__main__":
    main()