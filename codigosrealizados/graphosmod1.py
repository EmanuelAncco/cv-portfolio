"""
PIPELINE DE FIGURAS Q1 - MÓDULO 1: METODOLOGÍA (AUTOCONTENIDO v4)
==================================================================

Este script es un ejecutable autocontenido para la Sección 1 del paper.
Genera un conjunto de figuras de alta calidad (Q1) para la metodología,
utilizando datos reales de los sensores y paletas de colores vibrantes.

CORRECCIONES v4:
- Corregido error 'ValueError: Received invalid argument(s): edges' en Figura 8.
- Uso correcto de 'edgelist' en nx.draw.

Figuras Generadas:
1. (5x2) Análisis de Señal Cruda (Tiempo/PSD) para los 5 sensores.
2. (1x1) Topología del Grafo del Puente (Adyacencia).
3. (2x3) Comparación de Familias Wavelet (phi y psi).
4. (7x1) Descomposición DWT Multi-Nivel (Datos Reales, con color).
5. (1x1) PSD Comparativo (Superposición de los 5 sensores).
6. (5x1) Scalograma CWT Comparativo (para los 5 sensores).
7. (1x1) NUEVO: Análisis de Energía DWT (Justificación Wavelet).
8. (1x3) NUEVO: Comparación de Topologías de Grafo (Justificación GNN).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pywt  # Para Wavelets
import networkx as nx  # Para topología de grafos
from scipy.signal import welch  # Para Power Spectral Density (PSD)
from tqdm import tqdm
import seaborn as sns
from matplotlib.font_manager import findfont, FontProperties

# --- 1. CONFIGURACIÓN GLOBAL (Todo en un solo lugar) ---

# --- RUTAS BASE (Configurar según tu máquina) ---
BASE_DIR = Path(r"D:\Python_proyectos_2025\GAIATECH")
DATA_DIR_HEALTHY = Path(r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar")
FIGURES_DIR = BASE_DIR / "paper_figures_Q1_FINAL" / "1_methodology"

# --- PARÁMETROS GLOBALES DE SHM ---
SAMPLING_RATE_HZ = 333
NUM_SENSORS = 5

# --- PARÁMETROS DE PLOTTING ---
Q1_FONT_NAME = "Times New Roman"
FIGURE_DPI = 300
OUTPUT_IMAGE_FORMAT = "png"
# Paleta de colores para los 5 sensores (perceptualmente uniforme)
SENSOR_COLORS = plt.cm.viridis(np.linspace(0, 1, NUM_SENSORS))

# --- CONSTANTES PARA ESTE MÓDULO ---
TARGET_SENSOR_ID = 3
SEGMENT_DURATION_SEC = 5
WAVELET_FAMILY_DWT = 'db4'
WAVELET_LEVELS_DWT = 5
WAVELET_FAMILY_CWT = 'morl' # Morlet es mejor para CWT

# Configurar el logging raíz
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-7s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger()


# --- 2. FUNCIONES DE UTILIDAD (UTILITIES) ---

_font_found = None

def setup_q1_visuals():
    """Aplica la configuración global de Matplotlib/Seaborn."""
    global _font_found
    if _font_found is None:
        try:
            findfont(FontProperties(family=Q1_FONT_NAME))
            _font_found = True
            log.info(f"Fuente Q1 '{Q1_FONT_NAME}' encontrada y configurada.")
        except Exception:
            _font_found = False
            log.warning(f"ADVERTENCIA: Fuente Q1 '{Q1_FONT_NAME}' no encontrada. Usando serif por defecto.")

    sns.set_style("whitegrid")
    font_family = Q1_FONT_NAME if _font_found else 'serif'

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [font_family],
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.dpi': FIGURE_DPI, 'savefig.dpi': FIGURE_DPI,
        'savefig.format': OUTPUT_IMAGE_FORMAT, 'axes.edgecolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.labelcolor': 'black', 'text.color': 'black',
        'grid.color': '#DDDDDD', 'grid.linestyle': '--'
    })

def add_subplot_label(ax, label, x=0.02, y=0.95, **kwargs):
    """Añade una etiqueta de subfigura (ej. '(a)') a un eje (ax)."""
    props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7)
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontweight='bold',
            fontsize=11, verticalalignment='top', bbox=props, **kwargs)

# --- 3. CARGADOR DE DATOS (DATA LOADER) ---

def load_sensor_data(directory: Path, num_sensors: int, file_limit: int = 20) -> dict:
    """Carga y concatena datos de sensores desde un directorio."""
    log.info(f"Iniciando carga de datos de sensores desde: {directory}")
    if not directory.exists():
        log.error(f"¡Directorio de datos no encontrado! {directory}")
        return {}
    if not directory.is_dir():
        log.error(f"La ruta de datos no es un directorio: {directory}")
        return {}

    sensor_files = {i: [] for i in range(1, num_sensors + 1)}
    all_files = list(directory.glob("*.txt"))
    log.info(f"Encontrados {len(all_files)} archivos .txt en total.")

    for f_path in all_files:
        try:
            sid = int(f_path.name.split('_')[0])
            if sid in sensor_files:
                sensor_files[sid].append(f_path)
        except (ValueError, IndexError):
            log.warning(f"No se pudo extraer ID de sensor numérico de '{f_path.name}'.")

    sensor_data_concat = {}
    for sid, files in sensor_files.items():
        if not files:
            log.warning(f"No se encontraron archivos para el Sensor {sid}.")
            continue

        log.info(f"Procesando Sensor {sid} ({len(files)} archivos)...")
        data_list = []
        files_to_load = files[:file_limit] if file_limit else files

        for f_path in tqdm(files_to_load, desc=f"Sensor {sid}", leave=False, unit="file"):
            try:
                data = pd.read_csv(
                    f_path, sep='\s+', header=None, usecols=[1],
                    engine='python', on_bad_lines='skip'
                ).values
                if data is not None and data.size > 0:
                    data_list.append(data)
            except Exception as e:
                log.error(f"Error cargando {f_path.name}: {e}")

        if not data_list:
            log.error(f"No se pudieron cargar datos válidos para el Sensor {sid}.")
            continue

        try:
            concatenated_data = np.concatenate(data_list, axis=0).squeeze()
            sensor_data_concat[sid] = concatenated_data
            log.info(f"Sensor {sid} cargado. Total puntos: {len(concatenated_data):,}")
        except Exception as e:
            log.error(f"Error concatenando datos para Sensor {sid}: {e}")

    log.info("Carga de datos de sensores completada.")
    return sensor_data_concat

# --- 4. FUNCIONES DE PLOTTING (MÓDULO 1) ---

def get_signal_segment(signal_data: np.ndarray, fs: float, start_sec: int = 10):
    """Extrae un segmento de señal de duración definida."""
    start_index = int(start_sec * fs)
    num_points = int(SEGMENT_DURATION_SEC * fs)

    if len(signal_data) < start_index + num_points:
        log.warning(f"Datos insuficientes para segmento, usando datos disponibles desde {start_index}.")
        segment = signal_data[start_index:]
        if len(segment) < fs:
            log.error(f"Segmento de datos demasiado corto ({len(segment)} puntos) para análisis.")
            return None
    else:
        segment = signal_data[start_index : start_index + num_points]

    return segment

def plot_all_sensors_time_psd(sensor_data_map: dict, fs: float, save_dir: Path):
    """
    Genera Figura 1 (MEJORADA): Panel 5x2 de Tiempo y PSD para CADA sensor.
    """
    log.info("Generando Figura 1: Análisis de Señal Cruda (Tiempo/PSD) para los 5 sensores...")
    try:
        fig, axes = plt.subplots(NUM_SENSORS, 2, figsize=(12, 16))
        fig.suptitle("Raw Signal Analysis (Healthy State) - All Sensors", fontsize=16)

        labels = 'abcdefghij' # Etiquetas para 10 subplots (5x2)

        for i in range(1, NUM_SENSORS + 1):
            ax_time = axes[i-1, 0]
            ax_psd = axes[i-1, 1]
            color = SENSOR_COLORS[i-1]

            signal_data = sensor_data_map.get(i)
            if signal_data is None:
                log.warning(f"No hay datos para Sensor {i}, plot omitido.")
                ax_time.text(0.5, 0.5, f"Sensor {i}\nData not found", ha='center', va='center', color='red')
                ax_psd.text(0.5, 0.5, f"Sensor {i}\nData not found", ha='center', va='center', color='red')
                continue

            segment = get_signal_segment(signal_data, fs, start_sec=10*i)
            if segment is None: continue

            time_axis = np.arange(len(segment)) / fs

            # --- (a) Dominio del Tiempo ---
            ax_time.plot(time_axis, segment, color=color, linewidth=0.5)
            ax_time.set_ylabel(f"Sensor {i}\nAmplitude")
            ax_time.set_xlim(0, time_axis[-1])
            add_subplot_label(ax_time, labels[2*(i-1)]) # (a), (c), (e), ...
            if i == NUM_SENSORS: ax_time.set_xlabel("Time (s)")
            if i == 1: ax_time.set_title("Time-Domain Signal")

            # --- (b) Dominio de la Frecuencia (PSD) ---
            f, Pxx = welch(segment, fs, nperseg=int(fs*2))
            ax_psd.plot(f, Pxx, color=color, linewidth=1.0)
            ax_psd.set_ylabel(f"Sensor {i}\nPSD (log scale)")
            ax_psd.set_yscale('log')
            ax_psd.set_xscale('log')
            ax_psd.set_xlim(0.1, fs / 2)
            add_subplot_label(ax_psd, labels[2*(i-1) + 1]) # (b), (d), (f), ...
            if i == NUM_SENSORS: ax_psd.set_xlabel("Frequency (Hz)")
            if i == 1: ax_psd.set_title("Power Spectral Density (PSD)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = save_dir / f"figure_1_all_sensors_time_psd.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 1 guardada en: {save_path}")
        plt.close(fig)

    except Exception as e:
        log.error(f"Error en plot_all_sensors_time_psd: {e}", exc_info=True)


def plot_bridge_graph_topology(save_dir: Path):
    """
    Genera Figura 2: Visualiza la topología del grafo de 5 nodos.
    """
    log.info("Generando Figura 2: Topología del Grafo del Puente (Propuesta)...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        G = nx.Graph()
        edge_list = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        G.add_edges_from(edge_list)
        pos = {0: (0.0, 0.5), 1: (0.0, -0.5), 2: (1.0, 0.5), 3: (1.0, -0.5), 4: (2.0, 0.0)}
        labels = {i: f"Sensor {i+1}\n(Node {i})" for i in range(5)}

        nx.draw(G, pos, ax=ax, with_labels=True, labels=labels,
                node_color='#1f77b4', node_size=3000, font_size=11,
                font_weight='bold', font_color='white',
                edge_color='black', width=2.0)

        ax.set_title("Bridge Sensor Graph Topology (N=5)")
        ax.text(0.5, 1.05, "Physical Adjacency-Based Graph (M2 / M3)",
                transform=ax.transAxes, ha='center', fontsize=10, style='italic')
        plt.tight_layout()
        save_path = save_dir / f"figure_2_graph_topology.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 2 guardada en: {save_path}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Error en plot_bridge_graph_topology: {e}", exc_info=True)


def plot_wavelet_families(save_dir: Path):
    """
    Genera Figura 3 (MEJORADA): Compara familias wavelet (phi y psi).
    """
    log.info("Generando Figura 3: Comparación de Familias Wavelet...")
    try:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle("Comparison of Wavelet Families (Scaling $\phi$ and Wavelet $\psi$ functions)", fontsize=14)

        wavelet_names = ['db4', 'db8', 'sym4', 'sym8', 'coif2', 'coif5']
        label_chars = 'abcdef'[0:len(wavelet_names)]

        for ax, name, label in zip(axes.flat, wavelet_names, label_chars):
            try:
                wavelet = pywt.Wavelet(name)
                phi, psi, x = wavelet.wavefun(level=8)

                ax.plot(x, psi, color='#d62728', linewidth=1.5, label='Wavelet ($\psi$)')
                ax.plot(x, phi, color='#1f77b4', linewidth=1.0, linestyle='--', alpha=0.7, label='Scaling ($\phi$)')

                ax.set_title(f"Family: {name}")
                ax.set_xlabel("Time (samples)")
                ax.set_ylabel("Amplitude")
                ax.legend(loc='upper right', fontsize=8)
                add_subplot_label(ax, label)

            except Exception as e:
                log.warning(f"No se pudo plotear la familia wavelet {name}: {e}")
                ax.text(0.5, 0.5, f"Error al cargar {name}", ha='center', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = save_dir / f"figure_3_wavelet_families.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 3 guardada en: {save_path}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Error en plot_wavelet_families: {e}", exc_info=True)


def plot_dwt_decomposition(signal_segment: np.ndarray, wavelet: str, level: int, fs: float, save_dir: Path):
    """
    Genera Figura 4 (MEJORADA): Descomposición DWT con bandas de color.
    """
    log.info(f"Generando Figura 4: Descomposición DWT ({wavelet}, L{level})...")
    try:
        coeffs = pywt.wavedec(signal_segment, wavelet, level=level)
        fig, axes = plt.subplots(level + 2, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"DWT Multi-Resolution Analysis (Wavelet: {wavelet} on Sensor {TARGET_SENSOR_ID})", fontsize=14)

        time_axis = np.arange(len(signal_segment)) / fs
        label_chars = 'abcdefghijklm'[0:level + 2]
        band_colors = plt.cm.plasma(np.linspace(0, 0.9, level + 1))

        # --- Plot (a): Señal Original ---
        ax = axes[0]
        ax.plot(time_axis, signal_segment, color='black', linewidth=0.5)
        ax.set_title(f"({label_chars[0]}) Original Time-Domain Signal")
        ax.set_ylabel("Amplitude")
        add_subplot_label(ax, label_chars[0])

        # --- Plots (b) a (g): Bandas Reconstruidas ---
        for i, coeff in enumerate(coeffs):
            ax = axes[i + 1]
            label_char = label_chars[i+1]
            color = band_colors[i]

            rec_coeffs = [np.zeros_like(c) if j != i else coeff for j, c in enumerate(coeffs)]
            rec_signal = pywt.waverec(rec_coeffs, wavelet)
            rec_signal = rec_signal[:len(time_axis)]

            nyquist = fs / 2.0
            if i == 0:
                 label = f"A{level}"
                 f_high = nyquist / (2**level)
                 freq_range = f"(0 - {f_high:.2f} Hz)"
                 title_label = f"Approximation {label}"
            else:
                 detail_level = level - i + 1
                 label = f"D{detail_level}"
                 f_low = nyquist / (2**detail_level)
                 f_high = nyquist / (2**(detail_level - 1))
                 freq_range = f"({f_low:.2f} - {f_high:.2f} Hz)"
                 title_label = f"Detail {label}"

            ax.plot(time_axis, rec_signal, color=color, linewidth=0.7)
            ax.set_title(f"({label_char}) {title_label} - Band {freq_range}")
            ax.set_ylabel("Amplitude")
            add_subplot_label(ax, label_char)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = save_dir / f"figure_4_dwt_decomposition.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 4 guardada en: {save_path}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Error en plot_dwt_decomposition: {e}", exc_info=True)

def plot_comparative_psd(sensor_data_map: dict, fs: float, save_dir: Path):
    """
    Genera Figura 5 (NUEVA): Superposición de PSD de los 5 sensores.
    """
    log.info("Generando Figura 5: PSD Comparativo (Superpuesto)...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for i in range(1, NUM_SENSORS + 1):
            signal_data = sensor_data_map.get(i)
            if signal_data is None: continue

            f, Pxx = welch(signal_data, fs, nperseg=int(fs*4))
            ax.plot(f, Pxx, color=SENSOR_COLORS[i-1], linewidth=1.0, alpha=0.8, label=f'Sensor {i}')

        ax.set_title("Comparative Power Spectral Density (All Sensors)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD ($Units^2/Hz$, log scale)")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(0.1, fs / 2)
        ax.legend()

        plt.tight_layout()
        save_path = save_dir / f"figure_5_comparative_psd.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 5 guardada en: {save_path}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Error en plot_comparative_psd: {e}", exc_info=True)

def plot_all_sensors_cwt(sensor_data_map: dict, wavelet: str, fs: float, save_dir: Path):
    """
    Genera Figura 6 (NUEVA): Panel 5x1 de Scalogramas CWT para CADA sensor.
    """
    log.info(f"Generando Figura 6: Scalograma CWT Comparativo ({wavelet})...")
    try:
        fig, axes = plt.subplots(NUM_SENSORS, 1, figsize=(10, 15), sharex=True, sharey=True)
        fig.suptitle(f"Continuous Wavelet Transform (CWT) Scalogram (Wavelet: {wavelet})", fontsize=16)

        min_freq = 0.5
        max_freq = 50.0
        scales = np.logspace(
            np.log10(pywt.frequency2scale(wavelet, max_freq) * fs),
            np.log10(pywt.frequency2scale(wavelet, min_freq) * fs),
            num=200
        )

        im = None
        label_chars = 'abcde'

        for i in range(1, NUM_SENSORS + 1):
            ax = axes[i-1]
            signal_data = sensor_data_map.get(i)
            if signal_data is None:
                ax.text(0.5, 0.5, f"Sensor {i}\nData not found", ha='center', va='center', color='red')
                continue

            segment = get_signal_segment(signal_data, fs, start_sec=10*i)
            if segment is None: continue

            time_axis = np.arange(len(segment)) / fs
            coefficients, frequencies = pywt.cwt(segment, scales, wavelet, sampling_period=1.0/fs)

            power = (np.abs(coefficients)) ** 2
            im = ax.pcolormesh(time_axis, frequencies, np.log10(power),
                               cmap='jet', shading='gouraud', vmin=np.log10(power).min(), vmax=np.log10(power).max())

            ax.set_ylabel(f"Sensor {i}\nFreq. (Hz)")
            ax.set_yscale('log')
            ax.set_ylim(min_freq, max_freq)
            add_subplot_label(ax, label_chars[i-1])

        axes[-1].set_xlabel("Time (s)")

        fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', pad=0.01, aspect=40, label="log$_{10}$(Power)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = save_dir / f"figure_6_all_sensors_cwt.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 6 guardada en: {save_path}")
        plt.close(fig)
    except Exception as e:
        log.error(f"Error en plot_all_sensors_cwt: {e}", exc_info=True)

# --- INICIO DE LA "MAGIA" (NUEVAS FIGURAS 7 Y 8) ---

def plot_dwt_energy_distribution(sensor_data_map: dict, wavelet: str, level: int, save_dir: Path):
    """
    Genera Figura 7 (NUEVA): Análisis de Energía DWT para justificar
    la extracción de características.
    """
    log.info(f"Generando Figura 7: Análisis de Distribución de Energía DWT...")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Etiquetas de las bandas: [A5, D5, D4, D3, D2, D1]
        band_labels = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
        num_bands = len(band_labels)
        sensor_ids = list(range(1, NUM_SENSORS + 1))

        energy_data = pd.DataFrame(index=band_labels, columns=[f"Sensor {i}" for i in sensor_ids])

        for sid in sensor_ids:
            signal_data = sensor_data_map.get(sid)
            if signal_data is None: continue

            # Usar un segmento largo para un análisis de energía estable
            segment = get_signal_segment(signal_data, SAMPLING_RATE_HZ, start_sec=100)
            if segment is None: continue

            coeffs = pywt.wavedec(segment, wavelet, level=level)

            # Calcular energía (suma de cuadrados)
            energies = [np.sum(np.square(c)) for c in coeffs]
            total_energy = np.sum(energies)
            energy_percent = (energies / total_energy) * 100

            # Guardar en el DataFrame (el orden de coeffs es [A_level, D_level, ..., D1])
            energy_data[f"Sensor {sid}"] = energy_percent

        # Convertir a numérico para plotear
        energy_data = energy_data.astype(float)

        # Crear el stacked bar chart
        energy_data.transpose().plot(
            kind='bar',
            stacked=True,
            ax=ax,
            cmap='plasma', # Paleta de colores vibrante
            figsize=(10, 6)
        )

        ax.set_title("DWT Energy Distribution per Band (All Sensors)")
        ax.set_ylabel("Energy Percentage (%)")
        ax.set_xlabel("Sensor")
        ax.legend(title="DWT Band", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]) # Espacio para la leyenda
        save_path = save_dir / f"figure_7_dwt_energy_analysis.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 7 guardada en: {save_path}")
        plt.close(fig)

    except Exception as e:
        log.error(f"Error en plot_dwt_energy_distribution: {e}", exc_info=True)


def create_physics_informed_graph_definition(num_nodes=5):
    """
    Crea el grafo ponderado (M4) basado en la geometría 3D del puente.
    """
    coords = {
        0: np.array([13.88, -4.0, -1.0]),  # Sensor 1
        1: np.array([13.88, 4.0, -1.0]),  # Sensor 2
        2: np.array([27.76, -4.0, -1.0]),  # Sensor 3
        3: np.array([27.76, 4.0, -1.0]),  # Sensor 4
        4: np.array([41.64, 0.0, -1.0])   # Sensor 5
    }

    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])
            weight = 1.0 / (dist + 1e-6)
            G.add_edge(i, j, weight=weight)
    return G, {i: (coords[i][0], coords[i][1]) for i in range(num_nodes)} # Posiciones 2D

def plot_graph_topology_comparison(save_dir: Path):
    """
    Genera Figura 8 (NUEVA): Compara las 3 topologías de grafo.
    """
    log.info("Generando Figura 8: Comparación de Topologías de Grafo...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Graph Topology Comparison (Methodology Justification)", fontsize=16)

        labels = {i: f"S{i+1}" for i in range(5)}
        node_pos = {0: (0, 0.5), 1: (0, -0.5), 2: (1, 0.5), 3: (1, -0.5), 4: (2, 0.0)}
        label_chars = 'abc'

        # --- (a) Grafo de Adyacencia (M2 / M3) ---
        ax = axes[0]
        G_adj = nx.Graph()
        G_adj.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
        nx.draw(G_adj, node_pos, ax=ax, with_labels=True, labels=labels,
                node_color='#1f77b4', node_size=2000, font_size=10,
                font_weight='bold', font_color='white',
                edge_color='black', width=2.0)
        ax.set_title("(a) Proposed: Adjacency Graph\n(M2 & M3)")
        add_subplot_label(ax, 'a')

        # --- (b) Grafo Completo (Ingenuo) ---
        ax = axes[1]
        G_full = nx.complete_graph(NUM_SENSORS)
        nx.draw(G_full, node_pos, ax=ax, with_labels=True, labels=labels,
                node_color='#ff7f0e', node_size=2000, font_size=10,
                font_weight='bold', font_color='white',
                edge_color='gray', width=1.0, style='--')
        ax.set_title("(b) Alternative: Fully Connected Graph\n(Naïve approach)")
        add_subplot_label(ax, 'b')

        # --- (c) Grafo Físico-Ponderado (M4) ---
        ax = axes[2]
        G_phys, pos_phys = create_physics_informed_graph_definition()
        edges_phys = list(G_phys.edges())
        weights = [G_phys[u][v]['weight'] * 5 for u,v in edges_phys]

        # CORRECCIÓN CRÍTICA: Usar 'edgelist' en lugar de 'edges'
        nx.draw(G_phys, pos_phys, ax=ax, with_labels=True, labels=labels,
                node_color='#d62728', node_size=2000, font_size=10,
                font_weight='bold', font_color='white',
                edgelist=edges_phys, edge_color=weights, width=weights,
                edge_cmap=plt.cm.Reds)
        ax.set_title("(c) Advanced: Physics-Informed Graph\n(M4 - Weighted by 1/distance)")
        add_subplot_label(ax, 'c')

        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        save_path = save_dir / f"figure_8_graph_topology_comparison.{OUTPUT_IMAGE_FORMAT}"
        plt.savefig(save_path)
        log.info(f"Figura 8 guardada en: {save_path}")
        plt.close(fig)

    except Exception as e:
        log.error(f"Error en plot_graph_topology_comparison: {e}", exc_info=True)


# --- 5. BLOQUE DE EJECUCIÓN PRINCIPAL ---

def main():
    """
    Función principal para ejecutar el pipeline del Módulo 1.
    """
    log.info("--- INICIANDO PIPELINE DE FIGURAS [MÓDULO 1: METODOLOGÍA] ---")

    setup_q1_visuals()

    output_directory = FIGURES_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    log.info(f"Directorio de salida de figuras: {output_directory}")

    try:
        # Cargar TODOS los datos de los 5 sensores
        all_healthy_data = load_sensor_data(
            DATA_DIR_HEALTHY,
            num_sensors=NUM_SENSORS,
            file_limit=20 # Límite para carga rápida
        )
        if not all_healthy_data:
            log.critical(f"No se pudieron cargar datos. Verifica la ruta: {DATA_DIR_HEALTHY}")
            return

        target_signal = all_healthy_data.get(TARGET_SENSOR_ID)
        if target_signal is None:
            log.error(f"No se encontraron datos para el sensor objetivo {TARGET_SENSOR_ID} para Fig 4.")
            return

        segment_fig4 = get_signal_segment(target_signal, SAMPLING_RATE_HZ, start_sec=10)
        if segment_fig4 is None:
            log.error("No se pudo crear el segmento para la Fig 4 (DWT).")
            return

        log.info("Datos cargados. Iniciando generación de figuras...")

    except Exception as e:
        log.critical(f"Fallo catastrófico durante la carga de datos: {e}", exc_info=True)
        return

    # Generar todas las figuras
    try:
        plot_all_sensors_time_psd(all_healthy_data, SAMPLING_RATE_HZ, output_directory)
        plot_bridge_graph_topology(output_directory)
        plot_wavelet_families(output_directory)
        plot_dwt_decomposition(segment_fig4, WAVELET_FAMILY_DWT, WAVELET_LEVELS_DWT, SAMPLING_RATE_HZ, output_directory)
        plot_comparative_psd(all_healthy_data, SAMPLING_RATE_HZ, output_directory)
        plot_all_sensors_cwt(all_healthy_data, WAVELET_FAMILY_CWT, SAMPLING_RATE_HZ, output_directory)

        # --- NUEVAS FIGURAS "MÁGICAS" ---
        plot_dwt_energy_distribution(all_healthy_data, WAVELET_FAMILY_DWT, WAVELET_LEVELS_DWT, output_directory)
        plot_graph_topology_comparison(output_directory)

    except Exception as e:
        log.critical(f"Error durante la generación de figuras: {e}", exc_info=True)
        return

    log.info("--- PIPELINE DE FIGURAS [MÓDULO 1] COMPLETADO ---")

if __name__ == "__main__":
    main()