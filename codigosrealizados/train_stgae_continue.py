# train_stgae_continue_WINDOWS.py
"""
Continuación de entrenamiento PI-STG-AE: Época 50 → 100
CORREGIDO para Windows + multiprocessing

Autor: Emanuel Ancco (EmanuelAncco)
Fecha: 2025-11-13 02:37:45 UTC
Hardware: i9-14900 + RTX 4060 8GB
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import joblib
import pywt
import gc
import warnings

warnings.filterwarnings('ignore')

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("❌ torch_geometric no instalado")
    sys.exit(1)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN (AJUSTADA PARA WINDOWS)
# ============================================================================
RESUME_DIR = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920"
DATA_DIR = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"
OUTPUT_BASE = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm"

HP = {
    'epochs': 50,
    'batch_size': 24,  # Reducido un poco para Windows
    'learning_rate': 0.00025,
    'weight_decay': 1e-5,
    'patience': 20,
    'scheduler_patience': 8,
    'scheduler_factor': 0.5,
    'window_size': 64,
    'stride': 32,
    'wavelet_name': 'db4',
    'wavelet_level': 5,
    'gnn_hidden': 128,
    'gnn_out': 64,
    'rnn_hidden': 256,
    'rnn_layers': 2,
}

# WINDOWS: num_workers debe ser 0
NUM_WORKERS = 0  # ← CRÍTICO PARA WINDOWS
PERSISTENT_WORKERS = False
USE_AMP = torch.cuda.is_available()
GRAD_ACCUM_STEPS = 2


# ============================================================================
# GRAFO FÍSICO
# ============================================================================
def create_physics_informed_graph(num_nodes=5):
    coords = {
        0: np.array([13.88, -4.0, -1.0]),
        1: np.array([13.88, 4.0, -1.0]),
        2: np.array([27.76, -4.0, -1.0]),
        3: np.array([27.76, 4.0, -1.0]),
        4: np.array([41.64, 0.0, -1.0])
    }

    edge_index_list, edge_weight_list = [], []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            weight = 1.0 / (dist + 1e-6)
            edge_index_list.extend([[i, j], [j, i]])
            edge_weight_list.extend([weight, weight])

    return {
        'edge_index': torch.tensor(edge_index_list, dtype=torch.long).t().contiguous(),
        'edge_weight': torch.tensor(edge_weight_list, dtype=torch.float32)
    }


def define_bridge_graph(num_nodes=5, custom_definition=None):
    if custom_definition:
        return custom_definition['edge_index'], custom_definition['edge_weight']
    edges = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3]]
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), None


# ============================================================================
# FUNCIONES DE DATOS
# ============================================================================
def adjust_signal_length(signal, target_len):
    if len(signal) == target_len:
        return signal
    if len(signal) > target_len:
        return signal[:target_len]
    return np.pad(signal, (0, target_len - len(signal)), mode='constant')


def apply_dwt_features(signal, wavelet='db4', level=5, target_len=None):
    if signal is None or len(signal) == 0:
        return None

    if target_len is None:
        target_len = len(signal)

    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        bands = []

        for i in range(level, 0, -1):
            idx = level - i + 1
            if idx >= len(coeffs):
                return None
            coeffs_zero = [np.zeros_like(c) for c in coeffs]
            coeffs_zero[idx] = coeffs[idx]
            rec = pywt.waverec(coeffs_zero, wavelet)
            bands.append(adjust_signal_length(rec, target_len))

        coeffs_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        rec_a = pywt.waverec(coeffs_approx, wavelet)
        bands.append(adjust_signal_length(rec_a, target_len))

        original = adjust_signal_length(signal, target_len)
        all_bands = [original] + bands[::-1]
        return np.stack(all_bands, axis=-1)

    except Exception as e:
        logger.error(f"Error DWT: {e}")
        return None


class SpatioTemporalWaveletDataset(Dataset):
    def __init__(self, data_dict, window_size, stride=1, num_nodes=5):
        self.window_size = window_size
        self.stride = stride
        self.num_nodes = num_nodes

        valid_data = {}
        min_len = float('inf')
        expected_features = -1

        for sid, data in data_dict.items():
            if data is not None and isinstance(data, np.ndarray) and data.ndim == 2:
                if data.shape[0] >= window_size:
                    if expected_features == -1:
                        expected_features = data.shape[1]
                    elif data.shape[1] != expected_features:
                        logger.error(f"S{sid}: features {data.shape[1]} != {expected_features}")
                        continue
                    valid_data[sid] = data
                    min_len = min(min_len, data.shape[0])

        if len(valid_data) != num_nodes:
            missing = set(range(1, num_nodes + 1)) - set(valid_data.keys())
            raise ValueError(f"Faltan sensores: {missing}")

        processed = [valid_data[sid][:min_len] for sid in range(1, num_nodes + 1)]
        self.data = np.stack(processed, axis=1).astype(np.float32)
        self.num_features = self.data.shape[2]
        self.n_samples = (self.data.shape[0] - window_size) // stride + 1

        logger.info(f"Dataset: {self.data.shape}, {self.n_samples} ventanas")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        return torch.from_numpy(window), torch.from_numpy(window)


# ============================================================================
# MODELO
# ============================================================================
class GNNLayer(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch, bias=False)
        self.conv2 = GCNConv(hidden_ch, out_ch, bias=False)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index = edge_index.to(x.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(x.device)
        x = self.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(x, edge_index, edge_weight)


class SpatioTemporalAutoencoder(nn.Module):
    def __init__(self, num_nodes, num_features, window_size, gnn_h, gnn_out, rnn_h, rnn_layers):
        super().__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_features = num_features
        self.gnn_hidden_dim = gnn_h
        self.gnn_encoder_out_dim = gnn_out
        self.rnn_encoder_hidden_dim = rnn_h
        self.rnn_layers = rnn_layers
        self.rnn_decoder_output_dim = gnn_h * num_nodes

        self.gnn_encoder = GNNLayer(num_features, gnn_h, gnn_out)
        self.rnn_encoder = nn.GRU(gnn_out * num_nodes, rnn_h, batch_first=True, num_layers=rnn_layers)
        self.rnn_decoder = nn.GRU(rnn_h, gnn_h * num_nodes, batch_first=True, num_layers=rnn_layers)
        self.gnn_decoder = GNNLayer(gnn_h, gnn_h, num_features)
        self.latent_project_up = nn.Linear(rnn_h, rnn_h)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index, edge_weight=None):
        B, T, N, F = x.shape

        x_flat = x.reshape(B * T, N, F)
        gnn_enc = self.gnn_encoder(x_flat, edge_index, edge_weight)
        gnn_enc = gnn_enc.reshape(B, T, N, self.gnn_encoder_out_dim)
        rnn_in = gnn_enc.reshape(B, T, -1)
        _, h_n = self.rnn_encoder(rnn_in)

        latent_vector_z = self.relu(self.latent_project_up(h_n[-1]))

        rnn_decoder_input = latent_vector_z.unsqueeze(1).repeat(1, T, 1)
        rnn_decoded, _ = self.rnn_decoder(rnn_decoder_input)
        gnn_input_decoder = rnn_decoded.reshape(B * T, N, self.gnn_hidden_dim)
        reconstructed_frames = self.gnn_decoder(gnn_input_decoder, edge_index, edge_weight)

        return reconstructed_frames.reshape(B, T, N, F)


# ============================================================================
# ENTRENAMIENTO
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scaler, device, edge_index, edge_weight):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Train', leave=False)
    for batch_idx, (inputs, _) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(inputs, edge_index, edge_weight)
            loss = criterion(outputs, inputs) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        pbar.set_postfix({'loss': f'{loss.item() * GRAD_ACCUM_STEPS:.6f}'})

    return total_loss / len(loader)


def validate(model, loader, criterion, device, edge_index, edge_weight):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc='Val', leave=False):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs, edge_index, edge_weight)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 90)
    logger.info("CONTINUACIÓN ENTRENAMIENTO PI-STG-AE: ÉPOCA 50 → 100")
    logger.info("=" * 90)
    logger.info(f"👤 Usuario: EmanuelAncco")
    logger.info(f"⚙️  Hardware: i9-14900 + RTX 4060 8GB")
    logger.info(f"📅 Fecha: {datetime.now()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🎮 Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE, f"CONTINUE_50to100_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📁 Output: {output_dir}\n")

    # Cargar checkpoint
    logger.info("📂 Cargando checkpoint de época 50...")
    hp_path = os.path.join(RESUME_DIR, 'hyperparameters_stgae_physics.json')
    scaler_path = os.path.join(RESUME_DIR, 'scaler_stgae_physics.gz')
    model_path = os.path.join(RESUME_DIR, 'best_model_stgae_physics.pth')
    history_path = os.path.join(RESUME_DIR, 'loss_history_stgae_physics.json')

    with open(hp_path, 'r') as f:
        hp_orig = json.load(f)
    with open(history_path, 'r') as f:
        history = json.load(f)

    scaler = joblib.load(scaler_path)
    best_val_loss_50 = hp_orig.get('best_val_loss', float('inf'))
    start_epoch = len(history['train_loss'])

    logger.info(f"   ✓ Época inicial: {start_epoch}")
    logger.info(f"   ✓ Best Val Loss (época 50): {best_val_loss_50:.6f}\n")

    hp_orig.update(HP)
    hp = hp_orig

    # Cargar datos
    logger.info("📂 Cargando datos...")
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

    sensor_data_raw = {i: [] for i in range(1, 6)}
    for fpath in tqdm(all_files, desc="Archivos"):
        try:
            sid = int(os.path.basename(fpath).split('_')[0])
            if sid in sensor_data_raw:
                data = pd.read_csv(fpath, sep='\s+', header=None, usecols=[1], engine='python',
                                   on_bad_lines='warn').values
                if data.size > 0:
                    sensor_data_raw[sid].append(data)
        except:
            pass

    sensor_concat = {}
    min_len = float('inf')
    for sid, dlist in sensor_data_raw.items():
        if dlist:
            valid = [d.flatten() for d in dlist if d.size > 0]
            if valid:
                concat = np.concatenate(valid)
                sensor_concat[sid] = concat
                min_len = min(min_len, len(concat))

    if len(sensor_concat) != 5:
        logger.error("Faltan sensores")
        return

    logger.info(f"   ✓ Longitud: {min_len:,} muestras")

    # Wavelet
    logger.info("🌊 Aplicando DWT...")
    sensor_features = {}
    for sid in tqdm(range(1, 6), desc="Wavelet"):
        feats = apply_dwt_features(sensor_concat[sid], hp['wavelet_name'], hp['wavelet_level'], min_len)
        if feats is None:
            logger.error(f"Error DWT S{sid}")
            return
        sensor_features[sid] = feats

    num_features = sensor_features[1].shape[1]
    logger.info(f"   ✓ Features: {num_features}")

    # Escalar
    logger.info("📊 Escalando...")
    sensor_scaled = {sid: scaler.transform(data) for sid, data in sensor_features.items()}

    del sensor_data_raw, sensor_concat, sensor_features
    gc.collect()

    # Dataset
    logger.info("📦 Creando dataset...")
    dataset = SpatioTemporalWaveletDataset(sensor_scaled, hp['window_size'], hp['stride'], 5)

    val_len = int(0.15 * len(dataset))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # DataLoaders SIN workers (Windows)
    train_loader = DataLoader(
        train_ds,
        batch_size=hp['batch_size'],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=hp['batch_size'],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    logger.info(f"   ✓ Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    logger.info(f"   ✓ Batches: {len(train_loader)} / {len(val_loader)}\n")

    # Modelo
    logger.info("🧠 Cargando modelo...")
    model = SpatioTemporalAutoencoder(5, num_features, hp['window_size'], hp['gnn_hidden'],
                                      hp['gnn_out'], hp['rnn_hidden'], hp['rnn_layers']).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    logger.info("   ✓ Checkpoint cargado")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   ✓ Parámetros: {total_params:,}\n")

    # Grafo
    logger.info("🌉 Creando grafo físico...")
    physics_graph = create_physics_informed_graph(5)
    edge_index, edge_weight = define_bridge_graph(5, physics_graph)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    logger.info(f"   ✓ Aristas: {edge_index.shape[1]}\n")

    # Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hp['scheduler_patience'],
                                                           factor=hp['scheduler_factor'])
    scaler_amp = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # Training
    logger.info("=" * 90)
    logger.info(f"🚀 ENTRENANDO: ÉPOCA {start_epoch + 1} → {start_epoch + hp['epochs']}")
    logger.info("=" * 90 + "\n")

    best_val_loss = best_val_loss_50
    patience_counter = 0

    for epoch in range(start_epoch, start_epoch + hp['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler_amp, device, edge_index, edge_weight)
        val_loss = validate(model, val_loader, criterion, device, edge_index, edge_weight)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(lr)

        improvement = ((best_val_loss - val_loss) / best_val_loss * 100) if val_loss < best_val_loss else 0

        logger.info(f"Epoch {epoch + 1:3d}/{start_epoch + hp['epochs']} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e} | "
                    f"{'✓ +' + f'{improvement:.2f}%' if improvement > 0 else ''}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_stgae_physics.pth'))
            joblib.dump(scaler, os.path.join(output_dir, 'scaler_stgae_physics.gz'))

            hp['best_val_loss'] = best_val_loss
            with open(os.path.join(output_dir, 'hyperparameters_stgae_physics.json'), 'w') as f:
                json.dump(hp, f, indent=4)
            with open(os.path.join(output_dir, 'loss_history_stgae_physics.json'), 'w') as f:
                json.dump(history, f, indent=4)

            logger.info(f"   💾 MEJOR MODELO GUARDADO")
        else:
            patience_counter += 1

        if patience_counter >= hp['patience']:
            logger.info(f"\n⚠ Early stopping")
            break

    # Plot
    logger.info("\n📊 Generando gráficos...")
    plt.figure(figsize=(14, 7))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train', marker='.')
    plt.plot(epochs, history['val_loss'], label='Val', marker='.')
    plt.axvline(start_epoch + 0.5, color='r', linestyle='--', label=f'Resumed at {start_epoch + 1}')
    plt.axhline(best_val_loss_50, color='orange', linestyle=':', label=f'Epoch 50: {best_val_loss_50:.6f}')
    plt.axhline(best_val_loss, color='g', linestyle=':', label=f'Best: {best_val_loss:.6f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training: Epoch 50 → 100 (PI-STG-AE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_curve_50to100.png'), dpi=300)
    plt.close()

    improvement_total = ((best_val_loss_50 - best_val_loss) / best_val_loss_50 * 100)
    logger.info("\n" + "=" * 90)
    logger.info("✅ ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 90)
    logger.info(f"📊 Época 50:  {best_val_loss_50:.6f}")
    logger.info(f"📊 Época 100: {best_val_loss:.6f}")
    logger.info(f"📈 Mejora:    {improvement_total:.2f}%")
    logger.info(f"📁 Modelos:   {output_dir}")
    logger.info("=" * 90 + "\n")


if __name__ == '__main__':
    # CRÍTICO para Windows multiprocessing
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrumpido")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)