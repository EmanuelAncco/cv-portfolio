"""
PIPELINE DE FIGURAS Q1 - MÓDULO 2: MÉTRICAS DE ENTRENAMIENTO (ESTILO "ULTRA-CLEAN")
===================================================================================

Este script genera las figuras de entrenamiento con un estilo visual "Ultra-Clean",
replicando la estética de líneas finas con marcadores discretos preferida por el usuario.

CAMBIOS DE ESTILO:
- Eliminado "Shadow Plotting" (suavizado + sombra).
- Implementado estilo de líneas finas con marcadores en cada época.
- Train Loss: Línea sólida fina.
- Val Loss: Línea discontinua con marcadores.
- Línea vertical roja punteada para la "Best Epoch".

Genera:
1. Table_1_Training_Metrics.md (Markdown)
2. Figure_2_Loss_Comparison.png (Curvas Comparativas Limpias)
3. Figure_3_Performance_Bar.png (Barras de Rendimiento)
4. Figure_4_Individual_Loss_Curves.png (Panel 2x2 detallado, estilo referencia)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import re
import seaborn as sns
from matplotlib.font_manager import findfont, FontProperties

# --- 1. CONFIGURACIÓN GLOBAL ---

BASE_DIR = Path(r"D:\Python_proyectos_2025\GAIATECH")
FIGURES_DIR = BASE_DIR / "paper_figures_Q1_FINAL" / "2_training_metrics"

# Estilos
Q1_FONT_NAME = "Times New Roman"
DPI = 300
OUTPUT_FORMAT = "png"

# Definición de Modelos
MODELS = {
    "M1": {
        "name": "ST-AE (No-GNN)",
        "path": BASE_DIR / r"resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627",
        "type": "json",
        "files": {"hp": "hyperparameters_no_gnn.json", "loss": "loss_history_no_gnn.json"},
        "color": "#555555", # Gris oscuro (Estilo referencia)
        "marker": "o"
    },
    "M2": {
        "name": "GNN-AE (Base)",
        "path": BASE_DIR / r"resultados_entrenamiento\run_gnn_20250910-020756",
        "type": "txt_log",
        "files": {"hp": "hyperparameters.json", "loss": "training_log_gnn.txt"},
        "color": "#6a0dad", # Violeta oscuro
        "marker": "^"
    },
    "M3": {
        "name": "Wavelet-GNN (Ours)",
        "path": BASE_DIR / r"resultados_entrenamiento_wavelet\run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343",
        "resume": BASE_DIR / r"resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547",
        "type": "json_resume",
        "files": {"hp": "hyperparameters_wavelet_gnn.json", "loss": "loss_history_wavelet_gnn.json"},
        "color": "#005b96", # Azul profesional
        "marker": "s"
    },
    "M4": {
        "name": "PI-STG-AE (Physics)",
        "path": BASE_DIR / r"resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920",
        "resume": BASE_DIR / r"resultados_entrenamiento_modelos_shm\RESUME-PHYSICS_run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920_e50_20251031-142347",
        "type": "json_resume",
        "files": {"hp": "hyperparameters_stgae_physics.json", "loss": "loss_history_stgae_physics.json"},
        "color": "#b30000", # Rojo oscuro
        "marker": "D"
    }
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger()

# --- 2. UTILITIES ---

def setup_style():
    try:
        findfont(FontProperties(family=Q1_FONT_NAME))
        font = Q1_FONT_NAME
    except:
        font = 'serif'

    sns.set_style("whitegrid") # Base blanca con grid suave
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [font], 'font.size': 10,
        'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
        'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.format': OUTPUT_FORMAT,
        'axes.edgecolor': '#333333', 'grid.alpha': 0.3,
        'lines.linewidth': 1.0, # Líneas finas por defecto
        'lines.markersize': 3   # Marcadores pequeños por defecto
    })

def add_label(ax, text):
    """Añade etiqueta (a), (b) estilo paper."""
    # Fondo blanco semitransparente para que se lea bien sobre el grid
    props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#dddddd', alpha=0.8)
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=12, fontweight='bold',
            va='top', ha='left', bbox=props)

# --- 3. CARGA DE DATOS ---

def parse_txt_log(file_path):
    history = {'train_loss': [], 'val_loss': []}
    if not file_path.exists(): return history
    pattern = re.compile(r"Train Loss:\s+([\d\.]+).*Val Loss:\s+([\d\.]+)")
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    history['train_loss'].append(float(match.group(1)))
                    history['val_loss'].append(float(match.group(2)))
    except Exception as e:
        log.error(f"Error TXT log: {e}")
    return history

def load_model_data():
    results = {}
    for key, cfg in MODELS.items():
        log.info(f"Procesando {cfg['name']}...")
        data = {'hp': {}, 'history': {'train_loss': [], 'val_loss': []}}

        # HPs
        hp_path = cfg['path'] / cfg['files']['hp']
        if hp_path.exists():
            with open(hp_path, 'r') as f: data['hp'] = json.load(f)
            if key == "M2": # Fix M2 bug
                data['hp'].update({'gnn_hidden': 32, 'gnn_out': 32, 'rnn_hidden': 64})

        # History
        if cfg['type'] == 'json':
            p = cfg['path'] / cfg['files']['loss']
            if p.exists():
                with open(p, 'r') as f: data['history'] = json.load(f)
        elif cfg['type'] == 'txt_log':
            p = cfg['path'] / cfg['files']['loss']
            data['history'] = parse_txt_log(p)
        elif cfg['type'] == 'json_resume':
            p_res = cfg['resume'] / cfg['files']['loss']
            if p_res.exists():
                with open(p_res, 'r') as f: data['history'] = json.load(f)
            else:
                p_base = cfg['path'] / cfg['files']['loss']
                if p_base.exists():
                    with open(p_base, 'r') as f: data['history'] = json.load(f)

        # Metrics
        if data['history']['val_loss']:
            best_idx = np.argmin(data['history']['val_loss'])
            data['metrics'] = {
                'best_val_loss': data['history']['val_loss'][best_idx],
                'best_epoch': best_idx + 1
            }
        else:
             data['metrics'] = {'best_val_loss': np.nan, 'best_epoch': 0}
        results[key] = data
    return results

# --- 4. PLOTTING ESTILO "ULTRA-CLEAN" ---

def plot_individual_loss_curves(results, save_dir):
    """
    Genera Figura 4 (NUEVA): Panel 2x2 con curvas individuales detalladas.
    Replica el estilo de '2_1_All_Models_Loss_Curves.jpg'.
    """
    log.info("Generando Panel 2x2 de Curvas Individuales (Estilo Referencia)...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = 'abcd'

    for i, (key, res) in enumerate(results.items()):
        ax = axes.flatten()[i]
        cfg = MODELS[key]

        train_loss = res['history']['train_loss']
        val_loss = res['history']['val_loss']
        if not val_loss: continue

        epochs = np.arange(1, len(val_loss) + 1)

        # Plot Train Loss (Línea sólida fina grisácea)
        ax.plot(epochs, train_loss, color='#888888', linestyle='-', linewidth=0.8,
                label='Training Loss')

        # Plot Val Loss (Línea discontinua con marcadores, color del modelo)
        ax.plot(epochs, val_loss, color=cfg['color'], linestyle='--', linewidth=1.0,
                marker=cfg['marker'], markersize=3, markeredgewidth=0.5,
                label='Validation Loss')

        # Línea vertical para Best Epoch
        best_ep = res['metrics']['best_epoch']
        best_val = res['metrics']['best_val_loss']
        ax.axvline(x=best_ep, color='red', linestyle=':', linewidth=1.0, alpha=0.8,
                   label=f'Best ({best_ep}): {best_val:.5f}')

        ax.set_yscale('log')
        ax.set_title(cfg['name'])
        if i >= 2: ax.set_xlabel("Epoch")
        if i % 2 == 0: ax.set_ylabel("Loss (MSE, log scale)")

        # Leyenda minimalista
        ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=0.9)
        add_label(ax, f"({labels[i]})")

    plt.tight_layout()
    plt.savefig(save_dir / "Figure_4_Individual_Loss_Curves.png")
    plt.close()

def plot_loss_comparison(results, save_dir):
    """
    Genera gráfico comparativo general con líneas finas.
    """
    log.info("Generando Curvas de Pérdida Comparativas (Líneas Finas)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, res in results.items():
        val_loss = res['history']['val_loss']
        if not val_loss: continue
        epochs = np.arange(1, len(val_loss) + 1)
        cfg = MODELS[key]

        label = f"{cfg['name']} (Best: {res['metrics']['best_val_loss']:.4f})"

        # Estilo limpio: línea sólida fina, marcadores cada 5 o 10 puntos para no saturar
        markevery = max(1, len(epochs) // 15)
        ax.plot(epochs, val_loss, label=label, color=cfg['color'], linewidth=1.2,
                marker=cfg['marker'], markersize=4, markevery=markevery)

    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (MSE) [Log Scale]")
    ax.set_title("Training Convergence Comparison")
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    ax.grid(True, which="both", ls="-", alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_dir / "Figure_2_Loss_Comparison.png")
    plt.close()

def plot_performance_bar(results, save_dir):
    log.info("Generando Barras de Rendimiento...")
    names = [MODELS[k]['name'] for k in results]
    values = [results[k]['metrics']['best_val_loss'] for k in results]
    colors = [MODELS[k]['color'] for k in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8, width=0.6)

    ax.set_yscale('log')
    ax.set_ylabel("Best MSE Loss (Lower is Better)")
    ax.set_title("Model Performance Summary")

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height*1.1, f"{val:.4f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.grid(axis='y', which="major", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "Figure_3_Performance_Bar.png")
    plt.close()

def generate_markdown_table(results, save_dir):
    log.info("Generando Tabla...")
    lines = [
        "# Tabla 1: Comparación de Modelos y Resultados", "",
        "| Parámetro / Métrica | " + " | ".join([MODELS[k]['name'] for k in results]) + " |",
        "| :--- | " + " | ".join([":---:" for _ in results]) + " |"
    ]

    fields = [
        ('Architecture', lambda d: d.get('model_type', 'N/A')),
        ('Window Size', lambda d: d['hp'].get('window_size')),
        ('Latent Dim', lambda d: d['hp'].get('rnn_hidden')),
        ('Learning Rate', lambda d: d['hp'].get('learning_rate')),
        ('**Best Val Loss**', lambda d: f"**{d['metrics']['best_val_loss']:.5f}**"),
        ('Total Epochs', lambda d: len(d['history']['val_loss']))
    ]

    for label, accessor in fields:
        row = f"| {label} |"
        for key in results:
            try: val = accessor(results[key])
            except: val = "-"
            row += f" {val} |"
        lines.append(row)

    with open(save_dir / "Table_1_Training_Metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --- MAIN ---

def main():
    log.info("--- INICIO MÓDULO 2 (ESTILO ULTRA-CLEAN) ---")
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = load_model_data()

    plot_individual_loss_curves(results, FIGURES_DIR) # Nueva figura estilo referencia
    plot_loss_comparison(results, FIGURES_DIR)
    plot_performance_bar(results, FIGURES_DIR)
    generate_markdown_table(results, FIGURES_DIR)

    log.info(f"--- LISTO. Revisa: {FIGURES_DIR} ---")

if __name__ == "__main__":
    main()