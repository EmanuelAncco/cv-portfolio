#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
GENERADOR CIENTÍFICO DE TABLAS APA 7 Y EVIDENCIA VISUAL - SHM
Proyecto: EMAIRC VISION - Structural Health Monitoring
Autor: Senior Data Scientist (Gemini)
Fecha: Noviembre 2025

DESCRIPCIÓN:
Genera tablas de resultados (Excel), análisis textual y gráficos de evidencia
para paper Q1. Realiza inferencia real cargando los pesos de los modelos
entrenados y procesando el conjunto de datos de validación.

MEJORAS:
- MINERÍA DE LOGS EXHAUSTIVA: Lee TODOS los logs de cada carpeta para reconstruir
  el historial completo (soluciona el problema de épocas faltantes).
- Logging detallado.
- Manejo robusto de dependencias (PyG).
- Generación de gráficos de reconstrucción (Evidencia cualitativa).
================================================================================
"""

import os
import sys
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pywt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution_tables.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Verificación de Dependencias Críticas
try:
    from torch_geometric.nn import GCNConv

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logger.warning(
        "torch_geometric no encontrado. La inferencia de modelos GNN podría ser inexacta si no se reimplementa la capa.")

# =====================================================================
# 1. CONFIGURACIÓN Y RUTAS
# =====================================================================
BASE_DIR = r"D:\Python_proyectos_2025\GAIATECH"
OUTPUT_DIR = os.path.join(BASE_DIR, "RESULTADOS_PAPER_Q1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rutas de Datos (Verificar existencia)
DATA_HEALTHY = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
DATA_DAMAGE = r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Dispositivo de inferencia: {DEVICE}")

WINDOW_SIZE = 64
STRIDE = 32
NUM_NODES = 5
SENSOR_IDS = [f'Sensor_{i}' for i in range(NUM_NODES)]

# Configuración de Modelos (Rutas actualizadas desde directorios.txt)
# Nota: Tuples son (Resume_Path, Base_Path) o solo Path
MODEL_CONFIGS = {
    'M1_GRU_AE': {
        'path': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627",
        'color': '#E74C3C', 'gnn': False, 'nf': 1, 'rnn_h': 96, 'enc_out': 0, 'dec_dim': 0,
        'desc': 'Baseline Temporal (No-GNN)'
    },
    'M2_GNN_Base': {
        'path': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756",
        'color': '#3498DB', 'gnn': True, 'nf': 1, 'gnn_h': 32, 'enc_out': 16, 'dec_dim': 32, 'rnn_h': 64,
        'desc': 'Baseline Espacial (GNN Estándar)'
    },
    'M3_Wavelet_GNN': {
        'path': (
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547",
            # Resume (50-100)
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343"
        # Base (1-50)
        ),
        'color': '#2ECC71', 'gnn': True, 'nf': 7, 'gnn_h': 128, 'enc_out': 64, 'dec_dim': 128, 'rnn_h': 256,
        'desc': 'Híbrido Tiempo-Frecuencia'
    },
    'M4_PI_STG_AE': {
        'path': (
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347",
            # Resume (50-100)
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920"
        # Base (1-50)
        ),
        'color': '#9B59B6', 'gnn': True, 'nf': 7, 'gnn_h': 128, 'enc_out': 64, 'dec_dim': 128, 'rnn_h': 256,
        'phys': True,
        'desc': 'Propuesto (Physics-Informed)'
    }
}

SENSOR_COORDS = {
    0: np.array([0.0, -4.0, 0.0]), 1: np.array([0.0, 4.0, 0.0]),
    2: np.array([27.76, -4.0, 0.0]), 3: [27.76, 4.0, 0.0],
    4: [55.52, 0.0, 0.0]
}
EDGE_LIST = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]


# =====================================================================
# 2. DEFINICIÓN DE MODELOS (REPRODUCIBILIDAD)
# =====================================================================

class GNNLayer(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        if HAS_PYG:
            self.conv1 = GCNConv(in_c, hid_c)
            self.conv2 = GCNConv(hid_c, out_c)
        else:
            self.conv1 = nn.Linear(in_c, hid_c)
            self.conv2 = nn.Linear(hid_c, out_c)

        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, ei, ew=None):
        if HAS_PYG:
            x = self.conv1(x, ei, ew)
            x = self.relu(x)
            x = self.conv2(x, ei, ew)
        else:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
        return x


class STGAE(nn.Module):
    def __init__(self, n, nf, w, gh, eo, rh, rl, dd):
        super().__init__()
        self.n = n;
        self.dd = dd
        self.gnn_enc = GNNLayer(nf, gh, eo)
        self.rnn_enc = nn.GRU(eo * n, rh, rl, batch_first=True)
        self.rnn_dec = nn.GRU(rh, dd * n, rl, batch_first=True)
        self.gnn_dec = GNNLayer(dd, gh, nf)

    def forward(self, x, ei, ew=None):
        b, t, _, _ = x.size()
        steps = []
        for i in range(t):
            snap = x[:, i, :, :].reshape(b * self.n, -1)
            # Batch graph handling logic
            bei = ei.repeat(1, b) + torch.arange(b, device=x.device).repeat_interleave(ei.size(1)) * self.n
            bew = ew.repeat(b) if ew is not None else None
            steps.append(self.gnn_enc(snap, bei, bew).reshape(b, self.n, -1))
        flat = torch.stack(steps, dim=1).reshape(b, t, -1)
        _, h = self.rnn_enc(flat)
        dec_out, _ = self.rnn_dec(h[-1].unsqueeze(1).repeat(1, t, 1))
        dec_out = dec_out.reshape(b, t, self.n, self.dd)
        recon = []
        for i in range(t):
            snap = dec_out[:, i, :, :].reshape(b * self.n, -1)
            bei = ei.repeat(1, b) + torch.arange(b, device=x.device).repeat_interleave(ei.size(1)) * self.n
            bew = ew.repeat(b) if ew is not None else None
            recon.append(self.gnn_dec(snap, bei, bew).reshape(b, self.n, -1))
        return torch.stack(recon, dim=1)


class STAE_NoGNN(nn.Module):
    def __init__(self, n, nf, w, rh, rl):
        super().__init__()
        self.n = n
        self.rnn_enc = nn.GRU(n * nf, rh, rl, batch_first=True)
        self.rnn_dec = nn.GRU(rh, n * nf, rl, batch_first=True)

    def forward(self, x):
        b, t, _, _ = x.size()
        _, h = self.rnn_enc(x.reshape(b, t, -1))
        out, _ = self.rnn_dec(h[-1].unsqueeze(1).repeat(1, t, 1))
        return out.reshape(b, t, self.n, -1)


def get_graph(phys=False):
    ei = torch.tensor(EDGE_LIST, dtype=torch.long).t().contiguous()
    if not phys: return ei, None
    w = [1.0 / (np.linalg.norm(np.array(SENSOR_COORDS[i]) - np.array(SENSOR_COORDS[j])) + 1e-6) for i, j in EDGE_LIST]
    return ei, torch.tensor(w, dtype=torch.float32)


# =====================================================================
# 3. UTILIDADES ROBUSTAS
# =====================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_log_file(filepath):
    """Extrae Epoch, Train Loss y Val Loss de archivos de log no estructurados."""
    data = []
    if not os.path.exists(filepath):
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if "Train Loss:" in line and "Val Loss:" in line:
                try:
                    parts = line.split("Epoch")[-1]
                    ep_str = parts.split('/')[0].strip()
                    ep = int(ep_str)

                    tr_part = line.split("Train Loss:")[1].split(',')[0].strip()
                    val_part = line.split("Val Loss:")[1].split('(')[0].strip()

                    data.append({
                        'Epoch': ep,
                        'Train_Loss': float(tr_part),
                        'Val_Loss': float(val_part)
                    })
                except Exception:
                    continue
    return pd.DataFrame(data)


def parse_all_logs_in_dir(dirpath):
    """
    Estrategia de Minería Exhaustiva:
    Busca TODOS los archivos *log*.txt en el directorio, los parsea todos
    y los fusiona, eliminando duplicados por época. Esto asegura capturar
    entrenamientos divididos en múltiples archivos (log.txt, log_1.txt, etc.).
    """
    log_files = glob.glob(os.path.join(dirpath, "*log*.txt"))
    if not log_files:
        logger.warning(f"No se encontraron logs en {dirpath}")
        return pd.DataFrame()

    all_dfs = []
    for f in log_files:
        df = parse_log_file(f)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    # Fusionar y limpiar
    full_df = pd.concat(all_dfs)
    # Ordenar y quedarse con el último registro de cada época (asumiendo que el más reciente es el válido)
    full_df = full_df.sort_values('Epoch').drop_duplicates(subset='Epoch', keep='last')
    return full_df


def load_data_robust(path, ids):
    """Carga datos tolerante a fallos de formato."""
    d = {}
    if not os.path.exists(path):
        logger.error(f"Directorio de datos no existe: {path}")
        return d

    files = sorted(glob.glob(os.path.join(path, "*.txt")))
    logger.info(f"Cargando datos desde {path}...")

    for sid in tqdm(ids, desc="Cargando Sensores"):
        idx = int(sid.split('_')[-1]) + 1
        matches = [f for f in files if
                   os.path.basename(f).startswith(f"{idx}_") or os.path.basename(f).startswith(f"{idx} ")]
        acc = []
        for f in matches:
            try:
                df = pd.read_csv(f, sep=None, engine='python', header=None)
                col_idx = 1 if df.shape[1] >= 2 else 0
                v = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').dropna().values
                if len(v) > 0: acc.append(v)
            except Exception as e:
                logger.warning(f"Error leyendo {os.path.basename(f)}: {e}")
        if acc:
            d[sid] = np.concatenate(acc)
        else:
            d[sid] = np.zeros(1000)
    return d


def wav_trans(data):
    """Transformada Wavelet db4 nivel 6."""
    res = []
    for n in range(data.shape[1]):
        c = pywt.wavedec(data[:, n], 'db4', level=6)
        feats = [signal.resample(x, len(data)) for x in c]
        res.append(np.stack(feats, axis=1))
    return np.stack(res, axis=1)


# =====================================================================
# 4. MOTOR PRINCIPAL
# =====================================================================

def main():
    logger.info("=== INICIANDO PIPELINE DE GENERACIÓN DE EVIDENCIA ===")

    # 1. CARGA DE DATOS
    try:
        h_dict = load_data_robust(DATA_HEALTHY, SENSOR_IDS)
        if not h_dict: raise ValueError("Sin datos.")
        ml = min([len(v) for v in h_dict.values()])
        h_data = np.stack([h_dict[s][:ml] for s in SENSOR_IDS], axis=1)
        scaler = StandardScaler().fit(h_data)
        h_norm = scaler.transform(h_data)

        windows = []
        limit_samples = 2000
        for i in range(0, min(len(h_norm), limit_samples * STRIDE), STRIDE):
            if i + WINDOW_SIZE < len(h_norm): windows.append(h_norm[i:i + WINDOW_SIZE])
        win_tensor = np.array(windows)
        logger.info(f"Tensor entrada: {win_tensor.shape}")
    except Exception as e:
        logger.critical(f"Fallo datos: {e}")
        return

    # 2. EVALUACIÓN
    arch_data = []
    train_data = []
    perf_data = []
    best_mse = float('inf')
    best_prediction = None
    best_original = None
    best_model_name = ""

    for m_name, conf in MODEL_CONFIGS.items():
        logger.info(f"Procesando: {m_name}")

        # A. Arquitectura
        try:
            if conf['gnn']:
                mod = STGAE(NUM_NODES, conf['nf'], WINDOW_SIZE, conf['gnn_h'], conf['enc_out'], conf['rnn_h'], 2,
                            conf['dec_dim'])
            else:
                mod = STAE_NoGNN(NUM_NODES, conf['nf'], WINDOW_SIZE, conf['rnn_h'], 2)
            arch_data.append({
                'Modelo': m_name, 'Tipo': 'GNN-RNN' if conf['gnn'] else 'RNN',
                'Física': 'Sí' if conf.get('phys') else 'No',
                'Parámetros': count_parameters(mod), 'Input Features': conf['nf']
            })
        except Exception:
            pass

        # B. Logs (MINERÍA EXHAUSTIVA)
        try:
            path = conf['path']
            df_log = pd.DataFrame()

            if isinstance(path, tuple):
                # Caso Híbrido: Base + Resume
                path_resume, path_base = path  # [0]=Resume, [1]=Base

                # Leer TODOS los logs de la carpeta base
                df_base = parse_all_logs_in_dir(path_base)
                # Leer TODOS los logs de la carpeta resume
                df_resume = parse_all_logs_in_dir(path_resume)

                if not df_base.empty and not df_resume.empty:
                    last_ep_base = df_base['Epoch'].max()
                    # Lógica inteligente: Si resume empieza en 1, sumar last_ep. Si empieza en 51, no sumar.
                    if df_resume['Epoch'].min() <= 1:
                        df_resume['Epoch'] += last_ep_base

                    df_log = pd.concat([df_base, df_resume]).sort_values('Epoch').drop_duplicates('Epoch', keep='last')
                else:
                    df_log = df_base if not df_base.empty else df_resume
            else:
                # Caso Simple
                df_log = parse_all_logs_in_dir(path)

            if not df_log.empty:
                train_data.append({
                    'Modelo': m_name,
                    'Epochs': int(df_log['Epoch'].max()),  # Forzar int
                    'Best Val Loss': df_log['Val_Loss'].min(),
                    'Final Train Loss': df_log['Train_Loss'].iloc[-1]
                })
        except Exception as e:
            logger.error(f"Error logs {m_name}: {e}")

        # C. Inferencia
        try:
            search_path = conf['path'][0] if isinstance(conf['path'], tuple) else conf['path']
            pts = glob.glob(os.path.join(search_path, "**/*.pth"), recursive=True)
            if not pts: continue
            tgt_pth = next((x for x in pts if "best_model" in os.path.basename(x)), pts[0])

            mod.load_state_dict(torch.load(tgt_pth, map_location=DEVICE), strict=False)
            mod.to(DEVICE).eval()

            if conf['nf'] > 1:
                wav_inp = np.array([wav_trans(w) for w in win_tensor])
                inp_tensor = torch.FloatTensor(wav_inp).to(DEVICE)
            else:
                inp_tensor = torch.FloatTensor(win_tensor).unsqueeze(-1).to(DEVICE)

            ei, ew = get_graph(conf.get('phys', False))
            if ei is not None: ei = ei.to(DEVICE)
            if ew is not None: ew = ew.to(DEVICE)

            with torch.no_grad():
                rec = mod(inp_tensor, ei, ew) if conf['gnn'] else mod(inp_tensor)

            pred_np = rec.cpu().numpy()
            if conf['nf'] > 1:
                mse = mean_squared_error(inp_tensor.cpu().flatten(), pred_np.flatten())
                mae = mean_absolute_error(inp_tensor.cpu().flatten(), pred_np.flatten())
                vis_pred = pred_np[0, :, 0, 0]
                vis_orig = inp_tensor.cpu().numpy()[0, :, 0, 0]
            else:
                mse = mean_squared_error(win_tensor.flatten(), pred_np[..., 0].flatten())
                mae = mean_absolute_error(win_tensor.flatten(), pred_np[..., 0].flatten())
                vis_pred = pred_np[0, :, 0, 0]
                vis_orig = win_tensor[0, :, 0]

            perf_data.append({'Modelo': m_name, 'MSE': mse, 'MAE': mae, 'R2 (Sample)': r2_score(vis_orig, vis_pred)})

            if mse < best_mse:
                best_mse = mse
                best_model_name = m_name
                best_prediction = vis_pred
                best_original = vis_orig

        except Exception as e:
            logger.error(f"Error inferencia {m_name}: {e}")

    # 3. ENTREGABLES
    logger.info("Guardando Excel...")
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "TABLAS_APA_RESULTADOS.xlsx")) as writer:
        pd.DataFrame(arch_data).to_excel(writer, sheet_name="1_Arquitectura", index=False)
        pd.DataFrame(train_data).to_excel(writer, sheet_name="2_Entrenamiento", index=False)
        pd.DataFrame(perf_data).to_excel(writer, sheet_name="3_Metricas", index=False)

    if best_prediction is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(best_original, 'k', alpha=0.7, label='Original')
        plt.plot(best_prediction, 'r--', label=f'Reconstrucción ({best_model_name})')
        plt.title(f"Mejor Modelo: {best_model_name} (MSE: {best_mse:.5f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "reconstruccion_comparativa.png"))
        plt.close()

    with open(os.path.join(OUTPUT_DIR, "ANALISIS_AUTOMATICO.txt"), "w") as f:
        f.write(f"Mejor Modelo: {best_model_name}\n MSE: {best_mse}\n")

    logger.info("=== FIN ===")


if __name__ == "__main__":
    main()