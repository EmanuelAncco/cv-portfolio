"""
PIPELINE DE FIGURAS Q1 - MÓDULO 1: METODOLOGÍA (ESTILO "HIGH IMPACT")
=====================================================================

Este script genera figuras de calidad de publicación (Q1) replicando el estilo
visual avanzado de las referencias proporcionadas (anotaciones, picos, pesos en grafos).

FIGURAS GENERADAS:
1. Figure_1_Spectral_Dashboard.png: Panel 2x2 (Tiempo, FFT con picos, PSD, Energía por Banda).
2. Figure_2_Wavelet_Decomposition.png: Stack vertical con colores específicos y rangos de frecuencia.
3. Figure_3_Graph_Comparison.png: Comparación visual "Binario vs Físico" con pesos explícitos.
4. Figure_4_Scalogram_CWT.png: Scalograma continuo (Extra).

Autor: GAIATECH Architecture Team
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pywt
import networkx as nx
from scipy.signal import welch, find_peaks
from tqdm import tqdm
import seaborn as sns
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.patches import Rectangle

# --- 1. CONFIGURACIÓN GLOBAL ---

BASE_DIR = Path(r"D:\Python_proyectos_2025\GAIATECH")
DATA_DIR_HEALTHY = Path(r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar")
FIGURES_DIR = BASE_DIR / "paper_figures_Q1_FINAL" / "1_methodology"

# Parámetros Físicos
SAMPLING_RATE_HZ = 333
NUM_SENSORS = 5
TARGET_SENSOR_ID = 3  # Sensor representativo para análisis detallado

# Estilos
Q1_FONT_NAME = "Times New Roman"
DPI = 300
COLORS_WAVELET = {
    'Original': 'black',
    'A5': '#2ca02c',  # Green
    'D5': '#1f77b4',  # Blue
    'D4': '#9467bd',  # Purple
    'D3': '#d62728',  # Red
    'D2': '#ff7f0e',  # Orange
    'D1': '#7f7f7f'  # Gray
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger()


# --- 2. UTILITIES ---

def setup_style():
    """Configura el estilo visual Q1."""
    try:
        findfont(FontProperties(family=Q1_FONT_NAME))
        font = Q1_FONT_NAME
    except:
        font = 'serif'
        log.warning(f"Fuente {Q1_FONT_NAME} no encontrada. Usando default.")

    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [font], 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': DPI, 'axes.edgecolor': '#333333', 'grid.alpha': 0.3
    })


def add_label(ax, text):
    """Añade etiqueta (a), (b) estilo paper."""
    ax.text(-0.05, 1.05, text, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')


# --- 3. CARGA DE DATOS ---

def load_single_sensor(directory, sensor_id, limit=50):
    """Carga datos concatenados de un solo sensor."""
    files = sorted(list(directory.glob(f"{sensor_id}_*.txt")))[:limit]
    if not files: return None
    data = []
    for f in tqdm(files, desc=f"Cargando S{sensor_id}", leave=False):
        try:
            d = pd.read_csv(f, sep='\s+', header=None, usecols=[1], engine='python', on_bad_lines='skip').values
            data.append(d)
        except:
            pass
    return np.concatenate(data).flatten() if data else None


# --- 4. FIGURA 1: DASHBOARD ESPECTRAL (MAGIA TIPO 1) ---

def plot_spectral_dashboard(signal, fs, save_dir):
    """
    Genera un dashboard 2x2 replicando Fig1-2_spectral_analysis_clear.jpg
    Incluye: Tiempo, FFT (con picos), PSD, Energía por Banda.
    """
    log.info("Generando Figura 1: Dashboard Espectral Completo...")

    # Recorte de señal (8 segundos como en la referencia)
    n_samples = int(8 * fs)
    sig = signal[10000: 10000 + n_samples]  # Offset arbitrario para evitar inicio
    t = np.arange(len(sig)) / fs

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- (a) Time Domain ---
    ax = axes[0, 0]
    ax.plot(t, sig, color='#333333', linewidth=1)
    ax.set_title("Time Domain Signal (8 seconds)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_xlim(0, 8)
    add_label(ax, "(a)")

    # --- (b) Frequency Domain (FFT) con Picos ---
    ax = axes[0, 1]
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fft_mag = np.abs(np.fft.rfft(sig)) / n * 2  # Normalizado

    # Detectar picos significativos
    peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag) * 0.1, distance=fs)  # distance aprox 1Hz

    ax.plot(freqs, fft_mag, color='blue', linewidth=1.5)
    ax.set_title("Frequency Domain (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("FFT Magnitude")
    ax.set_xlim(0, 15)  # Zoom en frecuencias bajas (modos)

    # Anotar los 3 picos principales (como en la imagen de referencia)
    top_peaks = sorted(peaks, key=lambda x: fft_mag[x], reverse=True)[:3]
    for p in top_peaks:
        f_val = freqs[p]
        mag_val = fft_mag[p]
        ax.axvline(x=f_val, color='red', linestyle='--', alpha=0.7)
        # Caja amarilla
        ax.text(f_val, mag_val * 1.05, f"{f_val:.1f} Hz", ha='center', fontsize=9,
                bbox=dict(facecolor='#ffffcc', edgecolor='orange', boxstyle='round,pad=0.2'))
    add_label(ax, "(b)")

    # --- (c) PSD (Welch) ---
    ax = axes[1, 0]
    f_welch, psd = welch(sig, fs, nperseg=fs * 2)
    ax.plot(f_welch, psd, color='#2ca02c', linewidth=1.5)  # Green
    ax.set_title("PSD (Welch Method)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_yscale('log')
    ax.set_xlim(0, 15)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    add_label(ax, "(c)")

    # --- (d) Energy Distribution by Band ---
    ax = axes[1, 1]
    coeffs = pywt.wavedec(sig, 'db4', level=5)
    # Orden: A5, D5, D4, D3, D2, D1
    energies = [np.sum(c ** 2) for c in coeffs]
    total_E = sum(energies)
    percentages = [e / total_E * 100 for e in energies]

    bands = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
    # Calcular rangos de frecuencia (Nyquist = fs/2 = 166.5)
    # D1: 83-166, D2: 41-83, D3: 20-41, D4: 10-20, D5: 5-10, A5: 0-5 (Aprox)
    nyq = fs / 2
    ranges = [
        f"0-{nyq / 32:.1f}Hz",  # A5
        f"{nyq / 32:.1f}-{nyq / 16:.1f}Hz",  # D5
        f"{nyq / 16:.1f}-{nyq / 8:.1f}Hz",  # D4
        f"{nyq / 8:.1f}-{nyq / 4:.1f}Hz",  # D3
        f"{nyq / 4:.1f}-{nyq / 2:.1f}Hz",  # D2
        f"{nyq / 2:.1f}-{nyq:.0f}Hz"  # D1
    ]
    labels = [f"{b}\n{r}" for b, r in zip(bands, ranges)]
    colors = [COLORS_WAVELET[b] for b in bands]

    bars = ax.bar(bands, percentages, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title("Energy Distribution by Frequency Band")
    ax.set_ylabel("Energy (%)")
    ax.set_xticklabels(labels, fontsize=9)

    # Etiquetas de porcentaje encima de las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    add_label(ax, "(d)")

    plt.tight_layout()
    plt.savefig(save_dir / "Figure_1_Spectral_Dashboard.png")
    log.info("Figura 1 generada.")


# --- 5. FIGURA 2: WAVELET DECOMPOSITION (MAGIA TIPO 2) ---

def plot_wavelet_decomposition(signal, fs, save_dir):
    """
    Genera stack vertical replicando Fig1-1_wavelet_decomposition_visible.jpg
    Colores específicos y títulos con rango de frecuencia.
    """
    log.info("Generando Figura 2: Descomposición Wavelet Detallada...")

    # Segmento corto para ver detalles (5 segundos)
    sig = signal[10000: 10000 + int(5 * fs)]
    t = np.arange(len(sig)) / fs

    coeffs = pywt.wavedec(sig, 'db4', level=5)
    # coeffs: [cA5, cD5, cD4, cD3, cD2, cD1]

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)

    # (a) Original
    ax = axes[0]
    ax.plot(t, sig, color=COLORS_WAVELET['Original'], linewidth=1.5)
    ax.set_title("Original Bridge Acceleration Signal", fontweight='bold')
    ax.set_ylabel("Accel (m/s²)")
    add_label(ax, "(a)")

    # Bandas
    band_names = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
    nyq = fs / 2
    # Frecuencias aproximadas para títulos
    freqs = [
        (0, nyq / 32), (nyq / 32, nyq / 16), (nyq / 16, nyq / 8),
        (nyq / 8, nyq / 4), (nyq / 4, nyq / 2), (nyq / 2, nyq)
    ]

    for i, (coeff, name, freq_range) in enumerate(zip(coeffs, band_names, freqs)):
        ax = axes[i + 1]

        # Reconstrucción de la banda (Up-sampling) para que tenga misma longitud temporal
        # Creamos lista de ceros excepto el nivel actual
        c_rec = [np.zeros_like(c) for c in coeffs]
        c_rec[i] = coeff
        rec_sig = pywt.waverec(c_rec, 'db4')[:len(t)]  # Trim to match

        color = COLORS_WAVELET[name]
        label_title = f"{'Approximation' if 'A' in name else 'Detail'} {name}"
        freq_text = f"({freq_range[0]:.2f}-{freq_range[1]:.2f} Hz)"

        ax.plot(t, rec_sig, color=color, linewidth=1.2)

        # Título coloreado dentro del gráfico o encima
        ax.set_title(f"{label_title} {freq_text}", color=color, fontsize=11)
        ax.set_ylabel("Amp")
        ax.grid(True, alpha=0.2)

        # Letra (b), (c), etc.
        add_label(ax, f"({'bcdefg'[i]})")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_dir / "Figure_2_Wavelet_Decomposition.png")
    log.info("Figura 2 generada.")


# --- 6. FIGURA 3: GRAPH COMPARISON (MAGIA TIPO 3) ---

def plot_graph_comparison(save_dir):
    """
    Genera comparación visual replicando Fig1-3_graph_comparison_NOTORIO.jpg
    Grafo Binario vs Físico con PESOS VISIBLES.
    """
    log.info("Generando Figura 3: Comparación de Grafos con Pesos...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Definir posiciones tipo "Puente" (Vista en planta esquemática)
    # S1(0), S2(1) izquierda. S3(2), S4(3) centro. S5(4) derecha extrema.
    pos = {
        0: (0, 0),  # S1
        1: (0, 4),  # S2
        2: (3, 0),  # S3
        3: (3, 4),  # S4
        4: (6, 2)  # S5
    }

    # Nombres de nodos
    labels = {i: f"S{i + 1}" for i in range(5)}

    # --- (a) BINARY GRAPH ---
    ax = axes[0]
    G_bin = nx.Graph()
    # Conectividad básica (Topología del puente)
    edges_bin = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    G_bin.add_edges_from(edges_bin)

    # Dibujar nodos
    nx.draw_networkx_nodes(G_bin, pos, ax=ax, node_color='#3498db', node_size=2500, edgecolors='black')
    # Dibujar aristas
    nx.draw_networkx_edges(G_bin, pos, ax=ax, edge_color='#2c3e50', width=3)
    # Etiquetas
    nx.draw_networkx_labels(G_bin, pos, ax=ax, labels=labels, font_size=14, font_weight='bold', font_color='white')

    # Decoración "1" en las aristas (como la referencia)
    edge_labels = {e: "1" for e in G_bin.edges()}
    nx.draw_networkx_edge_labels(G_bin, pos, ax=ax, edge_labels=edge_labels,
                                 font_size=10, font_weight='bold',
                                 bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='circle'))

    ax.set_title("BINARY GRAPH\n(All edges = 1, No spatial info)", fontsize=14, color='#2980b9', fontweight='bold')
    ax.axis('off')  # Opcional: dejar ejes para referencia de metros
    # Simular ejes de coordenadas como en la referencia
    ax.set_xlim(-1, 7);
    ax.set_ylim(-1, 5)
    ax.text(3, -1.5, "X Position (m)", ha='center');
    ax.text(-1.2, 2, "Y Position (m)", va='center', rotation=90)
    add_label(ax, "(a)")

    # --- (b) PHYSICS-INFORMED GRAPH ---
    ax = axes[1]
    G_phys = nx.Graph()
    # Grafo totalmente conectado pero ponderado por distancia
    for i in range(5):
        for j in range(i + 1, 5):
            p1 = np.array(pos[i])
            p2 = np.array(pos[j])
            dist = np.linalg.norm(p1 - p2)
            weight = 1.0 / (dist + 0.1)  # Evitar div zero
            # Solo añadimos aristas significativas para no ensuciar el gráfico
            if weight > 0.2:
                G_phys.add_edge(i, j, weight=weight)

    # Dibujar
    edges = G_phys.edges()
    weights = [G_phys[u][v]['weight'] for u, v in edges]
    # Normalizar anchos para visualización
    widths = [w * 5 for w in weights]

    nx.draw_networkx_nodes(G_phys, pos, ax=ax, node_color='#c0392b', node_size=2500, edgecolors='black')
    nx.draw_networkx_edges(G_phys, pos, ax=ax, edge_color='#e74c3c', width=widths, alpha=0.8)
    nx.draw_networkx_labels(G_phys, pos, ax=ax, labels=labels, font_size=14, font_weight='bold', font_color='white')

    # Etiquetas de peso en CAJAS AMARILLAS (como referencia)
    edge_labels_phys = {e: f"{G_phys[e[0]][e[1]]['weight']:.3f}" for e in G_phys.edges()}
    nx.draw_networkx_edge_labels(G_phys, pos, ax=ax, edge_labels=edge_labels_phys,
                                 font_size=9, font_weight='bold',
                                 bbox=dict(facecolor='#ffffcc', edgecolor='#e67e22', boxstyle='round,pad=0.2'))

    ax.set_title("PHYSICS-INFORMED GRAPH\n(w_ij = 1/distance, Spatial correlation)", fontsize=14, color='#c0392b',
                 fontweight='bold')
    ax.set_xlim(-1, 7);
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.text(3, -1.5, "X Position (m)", ha='center');
    ax.text(-1.2, 2, "Y Position (m)", va='center', rotation=90)
    add_label(ax, "(b)")

    plt.tight_layout()
    plt.savefig(save_dir / "Figure_3_Graph_Comparison.png")
    log.info("Figura 3 generada.")


# --- 7. FIGURA 4: SCALOGRAM (EXTRA) ---

def plot_scalogram(signal, fs, save_dir):
    log.info("Generando Figura 4: Scalograma CWT...")
    sig = signal[10000: 10000 + int(5 * fs)]  # 5 segundos
    t = np.arange(len(sig)) / fs

    scales = np.logspace(np.log10(2), np.log10(128), 100)
    coefs, freqs = pywt.cwt(sig, scales, 'morl', sampling_period=1 / fs)

    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(coefs), extent=[0, 5, freqs[-1], freqs[0]], aspect='auto', cmap='jet')
    plt.yscale('log')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Continuous Wavelet Transform (Scalogram)')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.savefig(save_dir / "Figure_4_Scalogram_CWT.png")
    plt.close()
    log.info("Figura 4 generada.")


# --- MAIN ---

def main():
    log.info("--- INICIO PIPELINE MÓDULO 1 (HIGH IMPACT) ---")
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    signal = load_single_sensor(DATA_DIR_HEALTHY, TARGET_SENSOR_ID)
    if signal is None:
        log.error("No se encontraron datos. Verifica la ruta.")
        return

    # Generar
    plot_spectral_dashboard(signal, SAMPLING_RATE_HZ, FIGURES_DIR)
    plot_wavelet_decomposition(signal, SAMPLING_RATE_HZ, FIGURES_DIR)
    plot_graph_comparison(FIGURES_DIR)
    plot_scalogram(signal, SAMPLING_RATE_HZ, FIGURES_DIR)

    log.info(f"--- COMPLETADO. Figuras guardadas en: {FIGURES_DIR} ---")


if __name__ == "__main__":
    main()