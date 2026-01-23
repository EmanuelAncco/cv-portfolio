# train_physwave_gat.py
"""
Script de entrenamiento para PhysWave-GAT

Autor: Emanuel Ancco (EmanuelAncco)
Fecha: 2025-11-13 19:35:03 UTC
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
import gc
import warnings

warnings.filterwarnings('ignore')

# Importar modelo
from physwave_gat_model_FAST import FastPhysWaveGAT, create_physical_graph, augment_signal

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
# CONFIGURACIÓN
# ============================================================================
CONFIG = {
    # Directorios
    'data_dir': r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar",
    'output_base': r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm",

    # Modelo
    'num_nodes': 5,
    'window_size': 64,
    'stride': 32,

    # Entrenamiento
    'batch_size': 20,  # Optimizado para RTX 4060
    'learning_rate': 0.0003,
    'weight_decay': 1e-5,
    'epochs': 80,
    'patience': 20,

    # Scheduler
    'scheduler_patience': 10,
    'scheduler_factor': 0.7,
    'min_lr': 1e-8,

    # Contrastive learning
    'lambda_contrastive': 0.1,  # Peso del contrastive loss
    'use_contrastive': True,

    # Optimización
    'use_amp': True,
    'gradient_clip': 0.5,
    'num_workers': 0,  # Windows

    # Física
    'sensor_coords': np.array([
        [13.88, -4.0, -1.0],
        [13.88, 4.0, -1.0],
        [27.76, -4.0, -1.0],
        [27.76, 4.0, -1.0],
        [41.64, 0.0, -1.0],
    ]),
}


# ============================================================================
# DATASET
# ============================================================================

class BridgeAccelerationDataset(Dataset):
    """Dataset simple para aceleraciones del puente."""

    def __init__(self, data, window_size, stride):
        """
        Args:
            data: np.array (timesteps, num_nodes)
            window_size: int
            stride: int
        """
        self.window_size = window_size
        self.stride = stride
        self.data = data.astype(np.float32)

        # Calcular número de ventanas
        self.n_samples = (len(data) - window_size) // stride + 1

        logger.info(f"Dataset: {len(data):,} timesteps → {self.n_samples:,} ventanas")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]  # (window_size, num_nodes)

        # Añadir dimensión de canal
        window = window[..., np.newaxis]  # (window_size, num_nodes, 1)

        return torch.from_numpy(window), torch.from_numpy(window)


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_bridge_data(data_dir):
    """Carga datos de aceleración de los 5 sensores."""
    logger.info("📂 Cargando datos...")

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    sensor_data = {i: [] for i in range(1, 6)}

    for filepath in tqdm(all_files, desc="Archivos"):
        try:
            sid = int(os.path.basename(filepath).split('_')[0])
            if sid in sensor_data:
                data = pd.read_csv(
                    os.path.join(data_dir, filepath),
                    sep='\s+',
                    header=None,
                    usecols=[1],
                    engine='python',
                    on_bad_lines='warn'
                ).values

                if data.size > 0:
                    sensor_data[sid].append(data.flatten())
        except:
            pass

    # Concatenar por sensor
    sensor_concat = {}
    min_len = float('inf')

    for sid, data_list in sensor_data.items():
        if data_list:
            concat = np.concatenate(data_list)
            sensor_concat[sid] = concat
            min_len = min(min_len, len(concat))

    if len(sensor_concat) != 5:
        raise ValueError(f"Faltan sensores: {set(range(1, 6)) - set(sensor_concat.keys())}")

    # Alinear longitudes y stackear
    data_matrix = np.column_stack([sensor_concat[i][:min_len] for i in range(1, 6)])

    logger.info(f"   ✓ Datos cargados: {data_matrix.shape}")
    return data_matrix


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler_amp, device,
                edge_index, edge_weight, config):
    """Entrena una época con optional contrastive learning."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_contrast = 0.0

    pbar = tqdm(loader, desc='Train', leave=False)

    for inputs, _ in pbar:
        inputs = inputs.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config['use_amp']):
            # Reconstruction loss
            outputs = model(inputs, edge_index, edge_weight)
            recon_loss = criterion(outputs, inputs)

            loss = recon_loss



        # Backward
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss += loss.item()
        total_recon += recon_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'recon': f'{recon_loss.item():.6f}'
        })

    n = len(loader)
    return total_loss / n, total_recon / n, total_contrast / n if config['use_contrastive'] else 0.0


def validate(model, loader, criterion, device, edge_index, edge_weight):
    """Validación."""
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
    logger.info("=" * 100)
    logger.info("ENTRENAMIENTO PhysWave-GAT: METODOLOGÍA COMPLETAMENTE NUEVA")
    logger.info("=" * 100)
    logger.info(f"👤 Usuario: EmanuelAncco")
    logger.info(f"⚙️  Hardware: i9-14900 + RTX 4060 8GB")
    logger.info(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🎮 Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Crear directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(CONFIG['output_base'], f"PhysWaveGAT_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"📁 Output: {output_dir}\n")

    # Cargar datos
    data_raw = load_bridge_data(CONFIG['data_dir'])

    # Escalar
    logger.info("📊 Escalando datos...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_raw)
    logger.info("   ✓ Escalado completado\n")

    # Dataset
    logger.info("📦 Creando dataset...")
    dataset = BridgeAccelerationDataset(data_scaled, CONFIG['window_size'], CONFIG['stride'])

    val_len = int(0.15 * len(dataset))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)

    logger.info(f"   ✓ Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    logger.info(f"   ✓ Batches: {len(train_loader)} / {len(val_loader)}\n")

    # Grafo físico
    logger.info("🌉 Creando grafo físico...")
    edge_index, edge_weight = create_physical_graph(CONFIG['sensor_coords'])
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    logger.info(f"   ✓ Aristas: {edge_index.shape[1]}\n")

    # Modelo
    logger.info("🧠 Inicializando PhysWave-GAT...")
    model = FastPhysWaveGAT(CONFIG).to(device)  # ← FastPhysWaveGAT en vez de PhysWaveGAT

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"   ✓ Parámetros: {total_params:,}")
    logger.info(f"   ✓ Componentes innovadores:")
    logger.info(f"      • Wavelet Scattering Transform")
    logger.info(f"      • Dynamic Graph Attention (GAT)")
    logger.info(f"      • Temporal Convolutional Network (TCN)")
    logger.info(f"      • Contrastive Learning Head")
    logger.info(f"      • Adaptive Graph Construction\n")

    # Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=CONFIG['scheduler_patience'],
        factor=CONFIG['scheduler_factor'], min_lr=CONFIG['min_lr']
    )
    scaler_amp = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'])

    # Training loop
    logger.info("=" * 100)
    logger.info(f"🚀 INICIANDO ENTRENAMIENTO ({CONFIG['epochs']} ÉPOCAS)")
    logger.info("=" * 100 + "\n")

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [],
               'train_contrast': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_recon, train_contrast = train_epoch(
            model, train_loader, criterion, optimizer, scaler_amp,
            device, edge_index, edge_weight, CONFIG
        )

        val_loss = validate(model, val_loader, criterion, device, edge_index, edge_weight)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['train_contrast'].append(train_contrast)
        history['lr'].append(lr)

        improvement = ((best_val_loss - val_loss) / best_val_loss * 100) if val_loss < best_val_loss else 0

        log_msg = f"Epoch {epoch:3d}/{CONFIG['epochs']} | " \
                  f"Train: {train_loss:.6f} (R:{train_recon:.6f}"

        if CONFIG['use_contrastive']:
            log_msg += f", C:{train_contrast:.6f}"

        log_msg += f") | Val: {val_loss:.6f} | LR: {lr:.2e}"

        if improvement > 0:
            log_msg += f" | ✓ +{improvement:.2f}%"

        logger.info(log_msg)

        # Guardar mejor
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG,
            }, os.path.join(output_dir, 'best_model_physwave_gat.pth'))

            joblib.dump(scaler, os.path.join(output_dir, 'scaler.gz'))

            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                config_save = CONFIG.copy()
                config_save['sensor_coords'] = config_save['sensor_coords'].tolist()
                json.dump(config_save, f, indent=4)

            with open(os.path.join(output_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=4)

            logger.info(f"   💾 MEJOR MODELO GUARDADO")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            logger.info(f"\n⚠ Early stopping (paciencia agotada)")
            break

    # Plot final
    logger.info("\n📊 Generando gráficos...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    epochs = range(1, len(history['val_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].axhline(best_val_loss, color='g', linestyle='--', label=f'Best: {best_val_loss:.6f}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(epochs, history['train_recon'], 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # Contrastive loss
    if CONFIG['use_contrastive']:
        axes[1, 0].plot(epochs, history['train_contrast'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Contrastive Loss')
        axes[1, 0].set_title('Contrastive Loss')
        axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['lr'], 'green', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('PhysWave-GAT Training History', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()

    # Resumen final
    logger.info("\n" + "=" * 100)
    logger.info("✅ ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 100)
    logger.info(f"📊 Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"📊 Épocas completadas: {epoch}")
    logger.info(f"📁 Modelos guardados en: {output_dir}")
    logger.info("=" * 100 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrumpido por usuario")
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)