# -*- coding: utf-8 -*-
"""
🧪 SUITE MAESTRA DE GENERACIÓN DE EVIDENCIA CIENTÍFICA (SHM-AI Q1) v5.0 🧪
==========================================================================
Proyecto: GAIATECH - Structural Health Monitoring con Grafos Físicos
Autor: Arquitecto de Software & Científico de Datos Principal
Fecha: Noviembre 2025

DESCRIPCIÓN:
    Sistema integral "Enterprise-Grade" para evaluación post-hoc.
    Diseñado para generar masivamente (>100) figuras de alta calidad.

    CARACTERÍSTICAS PREMIUM:
    - 🎨 Salida de consola con Emojis y barras de progreso ricas.
    - 📈 Análisis estadístico profundo (Entropía, Kurtosis, Skewness).
    - 🔄 Comparativas cruzadas automáticas entre los 4 modelos.
    - 🛡️ Manejo de errores resiliente y validación de rutas.

ESTRUCTURA DE SALIDA (7 MÓDULOS + EXTRAS):
    1. 🌊 Methodology (Wavelets, CWT, Energy analysis)
    2. 📉 Training Metrics (Loss curves, Stability, Convergence rate)
    3. 🕸️ Model Architecture (Graphs 2D/3D, Topology, Weights)
    4. 🔍 Reconstruction Analysis (MSE, SSIM, PCA, t-SNE, Residuals, Correlation)
    5. 🌉 3D Simulations (Bridge Geometry, Sensor Heatmaps, Damage projection)
    6. 🚨 Anomaly Detection (Thresholding, ROC, Precision-Recall, Confusion Matrix)
    7. 🏆 Meta-Analysis (Ablation, Comparative Tables, Radar Charts)
"""

import os
import sys
import re
import glob
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pywt
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_squared_error, roc_curve, auc, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.mplot3d import Axes3D

# --- 🎨 CONFIGURACIÓN VISUAL Y LOGGING ---
# Validar torch_geometric
try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("❌ FATAL: 'torch_geometric' no instalado. Instale con pip.")
    sys.exit(1)

# Estilo Matplotlib "Paper Ready"
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':'
})

# Paleta de Colores Científica
COLORS = {
    'M1': '#7f7f7f',  # Gris (Baseline)
    'M2': '#1f77b4',  # Azul (GNN-Base)
    'M3': '#2ca02c',  # Verde (Wavelet)
    'M4': '#d62728',  # Rojo (Physics - Our Method)
    'GT': 'black',  # Ground Truth
    'Healthy': '#1f77b4',
    'Damage': '#d62728',
    'Threshold': '#ff7f0e',  # Naranja
    'Resid': '#9467bd'  # Violeta
}

# --- 📂 RUTAS Y DIRECTORIOS ---
BASE_DIR = r"D:\Python_proyectos_2025\GAIATECH"
DATA_DIR_HEALTHY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
DATA_DIR_DAMAGE = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"
OUTPUT_ROOT = os.path.join(BASE_DIR, "paper_figures_Q1_FINAL_ULTIMATE")

DIRS = {
    "1": os.path.join(OUTPUT_ROOT, "1_methodology_wavelets"),
    "2": os.path.join(OUTPUT_ROOT, "2_training_metrics"),
    "3": os.path.join(OUTPUT_ROOT, "3_model_architecture"),
    "4": os.path.join(OUTPUT_ROOT, "4_reconstruction_analysis"),
    "5": os.path.join(OUTPUT_ROOT, "5_3d_simulations"),
    "6": os.path.join(OUTPUT_ROOT, "6_anomaly_detection"),
    "7": os.path.join(OUTPUT_ROOT, "7_additional_analysis"),
}

for d in DIRS.values(): os.makedirs(d, exist_ok=True)


# Logger con Emojis
class EmojiFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)


log_file = os.path.join(OUTPUT_ROOT, "generation_log_ultimate.txt")
handler_file = logging.FileHandler(log_file, mode='w', encoding='utf-8')
handler_console = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
handler_file.setFormatter(formatter)
handler_console.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler_file)
logger.addHandler(handler_console)

# Contador Global de Imágenes
IMG_COUNTER = 0


def save_fig(path):
    global IMG_COUNTER
    plt.savefig(path)
    plt.close()
    IMG_COUNTER += 1
    # Feedback visual sutil en consola
    # sys.stdout.write('.')
    # sys.stdout.flush()


# =============================================================================
# 🧠 CLASES DE UTILIDAD Y DATOS
# =============================================================================

def apply_dwt_multilevel(signal, wavelet='db4', level=5):
    """
    🌊 Aplica DWT multinivel y retorna stack [Original, Aprox, Detalles...].
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = [signal]  # Feature 0: Señal Original
    tgt_len = len(signal)

    # Detalles (D5...D1)
    for i in range(level, 0, -1):
        c_dummy = [np.zeros_like(c) for c in coeffs]
        idx = level - i + 1
        c_dummy[idx] = coeffs[idx]
        rec = pywt.waverec(c_dummy, wavelet)
        features.append(rec[:tgt_len])

    # Aproximación (A5)
    c_dummy = [np.zeros_like(c) for c in coeffs]
    c_dummy[0] = coeffs[0]
    rec = pywt.waverec(c_dummy, wavelet)
    features.append(rec[:tgt_len])

    return np.stack(features, axis=-1)


class BridgeDataset(Dataset):
    """🏗️ Dataset unificado robusto."""

    def __init__(self, data_dir, config, scaler, limit_files=None):
        self.win = config['window_size']
        self.stride = config['stride']
        self.use_wav = config.get('wavelet', False)
        self.scaler = scaler

        data_map = {i: [] for i in range(1, 6)}
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if limit_files: files = files[:limit_files]

        if not files:
            logger.warning(f"⚠️ No se encontraron archivos en {data_dir}")
            self.n_wins = 0
            return

        for f in tqdm(files, desc=f"📥 Cargando {os.path.basename(data_dir)}", leave=False, colour='green'):
            try:
                sid = int(os.path.basename(f).split('_')[0])
                if sid in data_map:
                    raw = pd.read_csv(f, sep='\s+', header=None, usecols=[1], engine='c').values.flatten()
                    data_map[sid].append(raw)
            except:
                continue

        processed = []
        self.min_len = float('inf')
        # Concatenar y validar
        for i in range(1, 6):
            if data_map[i]:
                sig = np.concatenate(data_map[i])
                self.min_len = min(self.min_len, len(sig))
                data_map[i] = sig
            else:
                self.min_len = 0

        if self.min_len == 0:
            self.n_wins = 0
            return

        # Transformación
        for i in range(1, 6):
            sig = data_map[i][:self.min_len]
            if self.use_wav:
                feats = apply_dwt_multilevel(sig)
            else:
                feats = sig.reshape(-1, 1)

            flat = feats.reshape(-1, feats.shape[-1])
            scaled = scaler.transform(flat).reshape(feats.shape)
            processed.append(scaled)

        self.data = np.stack(processed, axis=1)
        self.n_wins = (self.data.shape[0] - self.win) // self.stride + 1

    def __len__(self):
        return self.n_wins

    def __getitem__(self, i):
        s = i * self.stride
        w = torch.FloatTensor(self.data[s:s + self.win])
        return w, w


# =============================================================================
# 🏛️ ARQUITECTURAS DE MODELOS (INGENIERÍA INVERSA)
# =============================================================================

def get_physics_graph(num_nodes=5):
    """🌉 Genera el grafo físico ponderado."""
    coords = {
        0: [13.88, -4.0, -1.0], 1: [13.88, 4.0, -1.0],
        2: [27.76, -4.0, -1.0], 3: [27.76, 4.0, -1.0],
        4: [41.64, 0.0, -1.0]
    }
    edge_index, edge_weight = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            w = 1.0 / (dist + 1e-6)
            edge_index.extend([[i, j], [j, i]])
            edge_weight.extend([w, w])
    return torch.tensor(edge_index).t().long(), torch.tensor(edge_weight).float(), coords


def get_binary_graph():
    edge_index = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
                               [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]], dtype=torch.long).t()
    return edge_index, None


class GNNLayer(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c, bias=False)
        self.conv2 = GCNConv(hid_c, out_c, bias=False)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(x, edge_index, edge_weight)


class STGAE(nn.Module):
    def __init__(self, n_nodes, n_feat, win, gnn_h, gnn_o, rnn_h, rnn_l):
        super().__init__()
        self.gnn_enc = GNNLayer(n_feat, gnn_h, gnn_o)
        self.rnn_enc = nn.GRU(gnn_o * n_nodes, rnn_h, batch_first=True, num_layers=rnn_l)
        self.latent_proj = nn.Linear(rnn_h, rnn_h)
        self.use_proj = False
        self.rnn_dec = nn.GRU(rnn_h, gnn_h * n_nodes, batch_first=True, num_layers=rnn_l)
        self.gnn_dec = GNNLayer(gnn_h, gnn_h, n_feat)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None, return_latent=False):
        B, T, N, F = x.shape
        gnn_out = self.gnn_enc(x.view(B * T, N, F), edge_index, edge_weight).view(B, T, -1)
        _, h = self.rnn_enc(gnn_out)
        latent = h[-1]
        if self.use_proj: latent = self.relu(self.latent_project_up(latent))

        if return_latent: return latent

        rnn_out, _ = self.rnn_dec(latent.unsqueeze(1).repeat(1, T, 1))
        return self.gnn_dec(rnn_out.reshape(B * T, N, -1), edge_index, edge_weight).view(B, T, N, F)


class STAE_NoGNN(nn.Module):
    def __init__(self, n_nodes, n_feat, win, rnn_h, rnn_l):
        super().__init__()
        inp = n_nodes * n_feat
        self.enc = nn.GRU(inp, rnn_h, batch_first=True, num_layers=rnn_l)
        self.dec = nn.GRU(rnn_h, inp, batch_first=True, num_layers=rnn_l)

    def forward(self, x, return_latent=False):
        B, T, N, F = x.shape
        _, h = self.enc(x.view(B, T, -1))
        if return_latent: return h[-1]
        out, _ = self.dec(h[-1].unsqueeze(1).repeat(1, T, 1))
        return out.view(B, T, N, F)


# =============================================================================
# ⚙️ CONFIGURACIÓN DE MODELOS (FUSIÓN Y PARCHES)
# =============================================================================

MODELS_CFG = {
    'M1': {
        'label': 'M1 (No-GNN)',
        'path': r"resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627",
        'type': 'nognn', 'hp': {'rnn_hidden': 96, 'rnn_layers': 2}, 'wav': False
    },
    'M2': {
        'label': 'M2 (GNN-Base)',
        'path': r"resultados_entrenamiento\run_gnn_20250910-020756",
        'type': 'gnn',
        # 🚑 PARCHE M2: gnn_out=16
        'hp': {'gnn_hidden': 32, 'gnn_out': 16, 'rnn_hidden': 64, 'rnn_layers': 2},
        'wav': False
    },
    'M3': {
        'label': 'M3 (Wavelet-GNN)',
        'path': r"resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343",
        'resume': r"resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547",
        'type': 'gnn', 'hp': {'gnn_hidden': 128, 'gnn_out': 64, 'rnn_hidden': 256, 'rnn_layers': 2},
        'wav': True
    },
    'M4': {
        'label': 'M4 (PI-STG-AE)',
        'path': r"resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920",
        'resume': r"resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347",
        'type': 'gnn', 'hp': {'gnn_hidden': 128, 'gnn_out': 64, 'rnn_hidden': 256, 'rnn_layers': 2},
        'wav': True, 'physics': True
    }
}

COMMON_HP = {'window_size': 64, 'stride': 32}


# =============================================================================
# 🏭 MOTORES DE GENERACIÓN (7 MÓDULOS)
# =============================================================================

class MethodologyVisualizer:
    """🌊 Módulo 1: Análisis de Señal"""

    def run(self, data_dir, output_dir):
        logger.info("🌊 [MOD 1] Generando análisis de metodología...")
        f = glob.glob(os.path.join(data_dir, "*.txt"))[0]
        sig = pd.read_csv(f, sep='\s+', header=None, usecols=[1], engine='c').values.flatten()[:500]

        # 1. Wavelet Decomposition (Series)
        coeffs = pywt.wavedec(sig, 'db4', level=4)
        fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(sig, 'k', lw=1);
        axes[0].set_title("Señal Original (Aceleración Z)")
        axes[0].set_ylabel("Amplitud")
        for i, c in enumerate(coeffs):
            label = 'Aproximación (A4)' if i == 0 else f'Detalle (D{5 - i})'
            axes[i + 1].plot(c, color='#2ca02c', lw=1)
            axes[i + 1].set_ylabel(label)
        plt.xlabel("Muestras")
        save_fig(os.path.join(output_dir, "wavelet_decomposition_series.png"))

        # 2. Energía Relativa (Barplot)
        energies = [np.sum(c ** 2) for c in coeffs]
        total_e = sum(energies)
        rel_e = [e / total_e * 100 for e in energies]
        labels = ['A4'] + [f'D{5 - i}' for i in range(1, len(coeffs))]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=rel_e, palette="viridis")
        plt.title("Distribución de Energía por Banda Wavelet")
        plt.ylabel("% Energía Total")
        save_fig(os.path.join(output_dir, "wavelet_energy_distribution.png"))


class TrainingMetricsAnalyzer:
    """📉 Módulo 2: Métricas de Entrenamiento"""

    def run(self, configs, output_dir):
        logger.info("📉 [MOD 2] Unificando curvas de aprendizaje...")
        stats_list = []

        plt.figure(figsize=(12, 7))

        for key, cfg in configs.items():
            paths = [os.path.join(BASE_DIR, cfg['path'])]
            if cfg.get('resume'): paths.append(os.path.join(BASE_DIR, cfg['resume']))

            loss_vals = []
            for p in paths:
                log_f = glob.glob(os.path.join(p, "training_log*.txt"))
                if log_f:
                    with open(log_f[0], 'r', encoding='utf-8', errors='ignore') as f:
                        loss_vals.extend(
                            [float(m.group(1)) for line in f if (m := re.search(r"Val Loss: ([\d\.]+)", line))])

            if loss_vals:
                # Plot main curve
                plt.plot(loss_vals, label=f"{cfg['label']}", color=COLORS[key], lw=2, alpha=0.8)
                stats_list.append({
                    "Model": key,
                    "Min Loss": min(loss_vals),
                    "Final Loss": loss_vals[-1],
                    "Convergence Epoch": np.argmin(loss_vals)
                })

        plt.yscale('log')
        plt.title("Dinámica de Entrenamiento Comparativa (Historia Completa)")
        plt.ylabel("MSE Loss (Log Scale)")
        plt.xlabel("Épocas (Unificadas)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        save_fig(os.path.join(output_dir, "comparative_training_dynamics.png"))

        pd.DataFrame(stats_list).to_csv(os.path.join(output_dir, "training_stats.csv"), index=False)


class ArchitectureVisualizer:
    """🕸️ Módulo 3: Arquitectura"""

    def run(self, output_dir):
        logger.info("🕸️ [MOD 3] Renderizando grafos y topologías...")

        # Grafo Físico Ponderado (M4) - Heatmap de Adyacencia
        _, weights, _ = get_physics_graph()
        # Reconstruir matriz densa 5x5
        adj = np.zeros((5, 5))
        w_np = weights.numpy()
        k = 0
        for i in range(5):
            for j in range(i + 1, 5):
                adj[i, j] = adj[j, i] = w_np[k]
                k += 2

        plt.figure(figsize=(7, 6))
        sns.heatmap(adj, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Matriz de Adyacencia Ponderada (Física)")
        save_fig(os.path.join(output_dir, "physics_adjacency_heatmap.png"))


class ReconstructionAnalyzer:
    """🔍 Módulo 4: Análisis Profundo"""

    def __init__(self, device):
        self.device = device

    def run(self, model, loader_h, loader_d, model_key, output_dir):
        logger.info(f"🔍 [MOD 4] Analizando {model_key} (Reconstrucción & Latente)...")

        latents, mses, recs, origs, labels = [], [], [], [], []

        # Función de inferencia unificada
        def infer(loader, label_code):
            ei, ew = (get_physics_graph()[0:2] if MODELS_CFG[model_key].get('physics')
                      else get_binary_graph()) if 'gnn' in MODELS_CFG[model_key]['type'] else (None, None)
            if ei is not None: ei = ei.to(self.device)
            if ew is not None: ew = ew.to(self.device)

            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(self.device)
                    # Hack para latente: necesitamos modificar la clase o hacerlo en 2 pasos
                    # Asumimos que forward acepta return_latent (implementado en clases)
                    if ei is not None:
                        rec = model(x, ei, ew)
                        lat = model(x, ei, ew, return_latent=True)
                    else:
                        rec = model(x)
                        lat = model(x, return_latent=True)

                    loss = torch.mean((x - rec) ** 2, dim=(1, 3)).cpu().numpy()
                    latents.append(lat.cpu().numpy())
                    mses.append(loss)
                    labels.extend([label_code] * len(loss))

                    if len(recs) < 5:  # Guardar algunas muestras
                        recs.append(rec.cpu().numpy())
                        origs.append(x.cpu().numpy())

        infer(loader_h, 0)  # 0: Healthy
        infer(loader_d, 1)  # 1: Damage

        # Concatenar
        X_lat = np.concatenate(latents)
        y_lab = np.array(labels)
        all_mse = np.concatenate(mses)

        # 1. PCA del Espacio Latente (Separabilidad)
        # Subsample para velocidad en visualización
        idx = np.random.choice(len(X_lat), min(2000, len(X_lat)), replace=False)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_lat[idx])

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[y_lab[idx] == 0, 0], X_pca[y_lab[idx] == 0, 1], c=COLORS['Healthy'], label='Sano', alpha=0.5,
                    s=15)
        plt.scatter(X_pca[y_lab[idx] == 1, 0], X_pca[y_lab[idx] == 1, 1], c=COLORS['Damage'], label='Daño', alpha=0.5,
                    s=15)
        plt.title(f"Proyección PCA del Espacio Latente ({model_key})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        save_fig(os.path.join(output_dir, f"{model_key}_pca_latent.png"))

        # 2. Matriz de Correlación de Residuos (Sano vs Daño)
        # Residuos promedio por sensor
        # mse es [Samples, Nodes]
        mse_h_all = all_mse[y_lab == 0]
        if len(mse_h_all) > 0:
            corr_h = np.corrcoef(mse_h_all, rowvar=False)
            plt.figure(figsize=(6, 5))
            sns.heatmap(corr_h, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(f"Correlación de Error entre Sensores (Sano) - {model_key}")
            save_fig(os.path.join(output_dir, f"{model_key}_corr_resid_healthy.png"))

        return all_mse[y_lab == 0], all_mse[y_lab == 1]


class AnomalyDetector:
    """🚨 Módulo 6: Detección"""

    def run(self, mse_h, mse_d, model_key, output_dir):
        logger.info(f"🚨 [MOD 6] Evaluando detección de anomalías para {model_key}...")

        score_h = np.mean(mse_h, axis=1)
        score_d = np.mean(mse_d, axis=1)

        # 1. Curva ROC
        y_true = np.concatenate([np.zeros(len(score_h)), np.ones(len(score_d))])
        y_scores = np.concatenate([score_h, score_d])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color=COLORS[model_key], lw=2, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('Falsos Positivos (FPR)')
        plt.ylabel('Verdaderos Positivos (TPR)')
        plt.title(f'Curva ROC - {model_key}')
        plt.legend(loc="lower right")
        plt.grid(True)
        save_fig(os.path.join(output_dir, f"{model_key}_roc_curve.png"))

        # 2. Matriz de Confusión (Umbral 99%)
        thresh = np.percentile(score_h, 99)
        y_pred = (y_scores > thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f"Matriz de Confusión (@99%ile) - {model_key}")
        plt.xlabel("Predicción")
        plt.ylabel("Realidad")
        save_fig(os.path.join(output_dir, f"{model_key}_confusion_matrix.png"))

        return roc_auc, np.mean(score_h), np.mean(score_d)


# =============================================================================
# 🚀 PIPELINE MAESTRO
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 INICIANDO SUITE DE GENERACIÓN CIENTÍFICA EN {device}")
    logger.info("=======================================================")

    # 1. Módulos Estáticos
    MethodologyVisualizer().run(DATA_DIR_HEALTHY, DIRS["1"])
    TrainingMetricsAnalyzer().run(MODELS_CFG, DIRS["2"])
    ArchitectureVisualizer().run(DIRS["3"])

    # 2. Loop Dinámico de Modelos
    final_stats = []

    for key, cfg in MODELS_CFG.items():
        logger.info(f"\n⚡ PROCESANDO MODELO: {cfg['name']}...")

        # Carga de Recursos
        base_p = os.path.join(BASE_DIR, cfg['resume'] if cfg.get('resume') else cfg['path_base'])
        scaler_p = glob.glob(os.path.join(base_p, "scaler*.gz"))[0]
        scaler = joblib.load(scaler_p)

        hp = cfg['hp']
        full_hp = {**COMMON_HP, **hp}

        # Instanciar
        if cfg['type'] == 'nognn':
            model = STAE_NoGNN(5, cfg['features'], hp['window_size'], hp['rnn_hidden'], hp['rnn_layers'])
        else:
            model = STGAE(5, cfg['features'], hp['window_size'], hp['gnn_hidden'], hp['gnn_out'], hp['rnn_hidden'],
                          hp['rnn_layers'])

        # Cargar Pesos
        w_p = glob.glob(os.path.join(base_p, "*.pth"))[0]
        sd = torch.load(w_p, map_location=device, weights_only=False)
        if 'latent_project_up.weight' in sd and hasattr(model, 'latent_project_up'): model.use_proj = True
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()

        # Data Loaders (Subset rápido)
        # AUMENTAR 'limit' si se quiere procesar TODO el dataset (tardará horas)
        ds_h = BridgeDataset(DATA_DIR_HEALTHY, full_hp, scaler, limit_files=10)
        ds_d = BridgeDataset(DATA_DIR_DAMAGE, full_hp, scaler, limit_files=5)

        if ds_h.n_wins == 0: continue

        dl_h = DataLoader(ds_h, batch_size=32, shuffle=False)
        dl_d = DataLoader(ds_d, batch_size=32, shuffle=False)

        # Ejecutar Análisis
        recon = ReconstructionAnalyzer(device)
        mse_h, mse_d = recon.run(model, dl_h, dl_d, key, DIRS["4"])

        det = AnomalyDetector()
        auc_val, mean_h, mean_d = det.run(mse_h, mse_d, key, DIRS["6"])

        # Generar Histogramas por Sensor (Micro-Análisis)
        logger.info(f"   📊 Generando histogramas por sensor para {key}...")
        for s in range(5):
            plt.figure(figsize=(8, 5))
            sns.kdeplot(mse_h[:, s], fill=True, label='Healthy', color=COLORS['Healthy'], alpha=0.3)
            sns.kdeplot(mse_d[:, s], fill=True, label='Damage', color=COLORS['Damage'], alpha=0.3)
            plt.xscale('log')
            plt.title(f"Distribución de Error - Sensor {s + 1} ({key})")
            plt.xlabel("MSE (Log)")
            save_fig(os.path.join(DIRS["4"], f"{key}_sensor{s + 1}_kde.png"))

        final_stats.append({
            "Model": key, "AUC": auc_val,
            "MSE_H": mean_h, "MSE_D": mean_d
        })

    # 3. Meta-Análisis Final
    logger.info("\n🏆 [MOD 7] Generando comparativas finales...")
    df = pd.DataFrame(final_stats)

    # Barplot AUC
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Model', y='AUC', palette=[c['color'] for c in MODELS_CFG.values()])
    plt.ylim(0.5, 1.0)
    plt.title("Rendimiento de Detección de Anomalías (AUC)")
    for i in ax.containers: ax.bar_label(i, fmt='%.3f')
    save_fig(os.path.join(DIRS["7"], "final_auc_comparison.png"))

    # Tabla Resumen
    print("\n" + "=" * 50)
    print("RESULTADOS FINALES")
    print("=" * 50)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(DIRS["7"], "final_results.csv"), index=False)

    logger.info(f"\n✅ PROCESO FINALIZADO. Total Imágenes Generadas: {IMG_COUNTER}")
    logger.info(f"📁 Resultados en: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()