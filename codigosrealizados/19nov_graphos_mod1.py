#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT DE GENERACIÓN DE FIGURAS Q1 (Versión REAL v11 - Módulo 1 AVANZADO)
========================================================================

Descripción:
    Este script se enfoca ÚNICAMENTE en generar las figuras para
    la Carpeta 1 (Metodología y Wavelets), usando un estilo visual
    Q1 (académico, Times New Roman, agrupado) y cargando los
    datos y helpers REALES.

    v11: Reemplaza los gráficos "simples" por un análisis Q1 profundo:
        1.  Análisis Espectral Multidominio (Tiempo, FFT, PSD, Energía).
        2.  Heatmap de Sensibilidad de Familias Wavelet (Justificación de db4).
        3.  Simulación Estocástica de Robustez al Ruido (Monte Carlo).
        4.  Comparación de Escalogramas CWT (Sano vs Daño).

Instrucciones (Emanuel):
    1.  Asegúrate de que tus discos 'D:\' estén accesibles.
    2.  Asegúrate de que los 4 scripts de entrenamiento (.py) estén
        en la misma carpeta que este script.
    3.  Ejecuta el script: `python 1_generate_methodology_ADVANCED.py`
"""

import os
import logging
import json
import shutil
import random
import re
import joblib
from pathlib import Path
import importlib.util
import gc
from tqdm import tqdm
import glob

# --- Dependencias Científicas ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import pywt  # (pip install PyWavelets)
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch
from matplotlib.gridspec import GridSpec

# --- Dependencias de ML ---
from sklearn.preprocessing import StandardScaler

# --- Dependencias de PyTorch (necesarias para cargar modelos) ---
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("ADVERTENCIA: torch_geometric no está instalado.")
    GCNConv = None  # Placeholder


# ###########################################################################
# --- MÓDULO DE IMPORTACIÓN DINÁMICA DE MODELOS (SECCIÓN 1) ---
# ###########################################################################

def import_model_class_from_file(file_path: Path, class_name: str, make_unique: bool = False):
    """Importa dinámicamente una clase de un archivo .py."""
    try:
        # Eliminar _Version2 si existe
        clean_path = file_path.with_name(file_path.name.replace("_Version2", ""))

        if not clean_path.exists():
            # No usar logging aquí, porque aún no está configurado
            print(f"ERROR CRÍTICO: El script .py no se encuentra en {clean_path}")
            return None

        module_name = clean_path.stem
        if make_unique:  # Evitar colisiones de nombres
            module_name = f"{clean_path.stem}_{class_name}"

        spec = importlib.util.spec_from_file_location(module_name, clean_path)
        if spec is None:
            raise ImportError(f"No se pudo crear spec desde {clean_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except Exception as e:
        print(f"ERROR CRÍTICO al importar la clase '{class_name}' desde {file_path}: {e}")
        return None


# ###########################################################################
# --- CONFIGURACIÓN (SECCIÓN 2) ---
# ###########################################################################

class Config:
    """Configuración centralizada para el pipeline de generación de figuras."""

    # 1. Semilla para Reproducibilidad
    RANDOM_SEED: int = 42

    # 2. Constantes del Modelo (basadas en tus archivos)
    WINDOW_SIZE: int = 64  # De hparams
    N_SENSORS: int = 5  # De hparams

    # 3. Rutas del Proyecto
    # USA LA RUTA DEL SCRIPT (`1_generate_methodology_ADVANCED.py`) COMO BASE
    BASE_DIR: Path = Path(__file__).parent.resolve()

    # Ruta a tus datos de entrenamiento (ESTADO SANO)
    DATA_DIR: Path = Path(r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar")

    # Ruta a tus datos de ANOMALÍA (DAÑO)
    ANOMALY_DATA_DIR: Path = Path(r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones")

    # Rutas a los SCRIPTS .PY (asumiendo que están en la misma carpeta que este script)
    MODEL_SCRIPT_FILES: dict[str, Path] = {
        "gnn_base": BASE_DIR / "train_autoencoder.py",
        "stgae_physics": BASE_DIR / "31_oct_newmodel.py",
        "wavelet_gnn": BASE_DIR / "train_wavelet_v3.py",
    }

    # Carpeta de salida principal para todas las figuras
    OUTPUT_DIR: Path = BASE_DIR / "paper_figures_Q1_FINAL"

    # 5. Diccionario de Directorios de Salida
    FIGURE_DIRS: dict[str, Path] = {
        "methodology": OUTPUT_DIR / "1_methodology_wavelets",
    }

    # 6. Configuración de Trazado
    PLOT_DPI: int = 300
    PLOT_FORMAT: str = "png"
    FONT_SIZE: int = 12


# --- Instancia de Configuración Global (Solo para importación) ---
_temp_config = Config()

# ###########################################################################
# --- IMPORTACIÓN GLOBAL DE CLASES Y HELPERS (SECCIÓN 3) ---
# ###########################################################################

IMPORTED_MODEL_HELPERS = {
    "GnnBaseGraphDef": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["gnn_base"], "define_bridge_graph"),
    "PhysicsGraphDef": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["stgae_physics"],
                                                    "create_physics_informed_graph"),
    "DWT_helper_function": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["wavelet_gnn"],
                                                        "apply_dwt_features"),
    "DWT_adjust_length": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["wavelet_gnn"],
                                                      "adjust_signal_length")
}

# Limpiar la config temporal
del _temp_config


# ###########################################################################
# --- SETUP DEL ENTORNO Y LOGGING (SECCIÓN 4) ---
# ###########################################################################

def setup_logging(output_dir: Path) -> None:
    """Configura el logging para consola y archivo."""
    log_file = output_dir / "figure_generation_MOD_1.log"
    # Configurar el logger raíz
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Limpiar handlers existentes si se re-ejecuta
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler de archivo
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler de consola
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Silenciar loggers ruidosos
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def setup_environment(config: Config) -> None:
    """Prepara el entorno: configura seeds, estilos de plot y crea directorios."""
    if not config.OUTPUT_DIR.exists():
        config.OUTPUT_DIR.mkdir(parents=True)

    # Crear solo el directorio de metodología
    methodology_dir = config.FIGURE_DIRS["methodology"]
    if methodology_dir.exists():
        try:
            shutil.rmtree(methodology_dir)
        except PermissionError:
            print(f"Error de permisos. No se pudo eliminar {methodology_dir}. ¿Archivos abiertos?")
            return
    methodology_dir.mkdir(parents=True)

    setup_logging(config.OUTPUT_DIR)
    logging.info(
        f"Entorno de logging configurado. Log guardado en: {config.OUTPUT_DIR / 'figure_generation_MOD_1.log'}")

    os.environ['PYTHONHASHSEED'] = str(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    logging.info(f"Seeds de aleatoriedad fijadas en: {config.RANDOM_SEED}")

    # --- Configuración de estilo Q1 (del script 'ultimate') ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.3)
        plt.rcParams['figure.dpi'] = config.PLOT_DPI
        plt.rcParams['savefig.dpi'] = config.PLOT_DPI
        plt.rcParams['font.size'] = config.FONT_SIZE
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['grid.alpha'] = 0.3
        logging.info("Estilos de Matplotlib y Seaborn (Q1) configurados.")
    except Exception as e:
        logging.warning(f"No se pudo cargar 'seaborn-v0_8-whitegrid' o 'Times New Roman'. Usando defaults. Error: {e}")

    logging.info(f"Creado directorio de figuras: {methodology_dir}")


def add_subfigure_label(ax: plt.Axes | Axes3D, label: str, x: float = -0.1, y: float = 1.05) -> None:
    """Añade etiqueta (a), (b), etc. - Compatible con 3D."""
    # Coordenadas ajustadas para el estilo de bbox
    if hasattr(ax, 'zaxis'):  # Es 3D
        ax.text2D(x, y, f'({label})', transform=ax.transAxes,
                  fontsize=18, weight='bold', va='top', ha='left',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='black', linewidth=1.5))
    else:
        ax.text(x, y, f'({label})', transform=ax.transAxes,
                fontsize=18, weight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='black', linewidth=1.5))


def save_plot(
        fig: plt.Figure,
        module_key: str,
        filename: str,
        config: Config
) -> None:
    """Función helper para guardar y cerrar figuras de forma robusta."""
    try:
        # tight_layout a veces falla con gridspec o 3D, lo llamamos con cuidado
        try:
            # Ajuste para suptitle
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        except Exception:
            logging.warning(f"tight_layout falló para {filename}. Guardando de todas formas.")

        filepath = config.FIGURE_DIRS[module_key] / f"{filename}.{config.PLOT_FORMAT}"
        fig.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
        logging.info(f"Figura guardada: {filepath}")
    except Exception as e:
        logging.error(f"Error al guardar la figura {filename}: {e}", exc_info=True)
    finally:
        plt.close(fig)


# ###########################################################################
# --- MÓDULO 0: UTILIDADES Y CARGA (LÓGICA REAL) ---
# ###########################################################################

def _load_data_from_dir_real(directory: Path, num_nodes: int, max_len: int = None) -> (np.ndarray, int):
    """
    Lógica de carga de datos REAL, basada en tu script de inferencia.
    Carga datos de sensores (columna 1) desde un directorio.
    """
    local_logger = logging.getLogger(f"{__name__}._load_data_from_dir_real")
    local_logger.info(f"Buscando archivos en: {directory} con patrón '<id>_*'")
    all_sensor_data = {}
    loaded_files_count = 0
    min_length = float('inf')

    sensor_ids_to_find = list(range(1, num_nodes + 1))

    for i in sensor_ids_to_find:
        search_pattern = str(directory / f"{i}_*")
        file_list = glob.glob(search_pattern)

        if not file_list:
            local_logger.warning(f"No se encontraron archivos para el sensor {i} con patrón: {search_pattern}")
            continue

        file_list.sort()
        sensor_df_list = []
        for filepath in file_list:
            try:
                # Esta es la corrección clave: usecols=[1] (aceleración)
                df = pd.read_csv(filepath, header=None, sep=r'\s+', usecols=[1], engine='python')
                if df.empty:
                    local_logger.warning(f"Archivo vacío omitido: {filepath}")
                    continue
                sensor_df_list.append(df)
            except Exception as e:
                local_logger.error(f"Error leyendo o procesando archivo {filepath}: {e}")

        if not sensor_df_list:
            local_logger.warning(f"No se pudo leer data válida para el sensor {i}.")
            continue

        full_sensor_df = pd.concat(sensor_df_list, ignore_index=True)
        all_sensor_data[i] = full_sensor_df.iloc[:, 0].values
        loaded_files_count += len(file_list)

        if len(all_sensor_data[i]) < min_length:
            min_length = len(all_sensor_data[i])

    if not all_sensor_data:
        local_logger.error("No se cargaron datos de ningún sensor.")
        return None, 0

    if len(all_sensor_data) < num_nodes:
        local_logger.warning(
            f"Se esperaban datos para {num_nodes} sensores, pero solo se cargaron {len(all_sensor_data)}.")
        for i in sensor_ids_to_find:
            if i not in all_sensor_data:
                local_logger.warning(f"Rellenando datos faltantes para el sensor {i} con ceros.")
                fill_length = min_length if min_length != float('inf') else 1
                all_sensor_data[i] = np.zeros(fill_length)

    if min_length == float('inf'):
        min_length = max_len if max_len else 1
        local_logger.warning(f"No se pudo determinar la longitud mínima, usando: {min_length}")

    # Truncar todos los arrays a la longitud mínima o max_len
    if max_len is None:
        max_len = min_length
    local_logger.info(f"Normalizando todos los sensores a la longitud: {max_len}")

    processed_data = np.zeros((max_len, num_nodes))
    for i in sensor_ids_to_find:
        sensor_data = all_sensor_data.get(i)
        if sensor_data is None:
            continue  # Ya está en ceros

        if len(sensor_data) >= max_len:
            processed_data[:, i - 1] = sensor_data[:max_len]
        else:
            pad_value = sensor_data[-1] if len(sensor_data) > 0 else 0
            padding = np.full(max_len - len(sensor_data), pad_value)
            processed_data[:, i - 1] = np.concatenate((sensor_data, padding))

    local_logger.info(f"Datos procesados con shape final: {processed_data.shape}.")
    return processed_data, max_len


def _helper_windowing(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Función helper para aplicar ventaneo a datos [T, N, F]."""
    T, N, F = data.shape
    n_samples = max(0, (T - window_size) // stride + 1)

    if n_samples == 0:
        logging.warning(f"No se pueden crear ventanas. Datos T={T}, window_size={window_size}, stride={stride}")
        return np.array([]).reshape(0, window_size, N, F)

    windows = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_samples, window_size, N, F),
        strides=(data.strides[0] * stride, data.strides[0], data.strides[1], data.strides[2])
    )
    return windows.copy()  # .copy() es crucial


def load_shm_data(config: Config) -> dict[str, np.ndarray]:
    """
    Carga y preprocesa los datos de SHM (healthy/anomaly).
    *** LÓGICA DE CARGA REAL ***
    """
    logging.info(f"--- Iniciando Carga de Datos REAL ---")

    # --- 1. Cargar Datos 'Healthy' ---
    logging.info(f"Cargando datos 'Healthy' desde: {config.DATA_DIR}")
    if not config.DATA_DIR.exists():
        logging.critical(f"Directorio de datos 'healthy' {config.DATA_DIR} no encontrado. Abortando.")
        raise FileNotFoundError(f"El directorio de datos {config.DATA_DIR} no existe.")

    healthy_data_raw, max_len_healthy = _load_data_from_dir_real(config.DATA_DIR, config.N_SENSORS)
    if healthy_data_raw is None:
        raise ValueError("Error fatal al cargar datos 'healthy'.")

    # Reshape a [T, N, F=1]
    healthy_data_full = healthy_data_raw[:, :, np.newaxis]
    logging.info(f"Datos 'healthy' cargados. Shape: {healthy_data_full.shape}")

    # Liberar memoria
    del healthy_data_raw
    gc.collect()

    # --- 2. Cargar Datos 'Anomaly' ---
    logging.info(f"Cargando datos 'Anomaly' desde: {config.ANOMALY_DATA_DIR}")
    if not config.ANOMALY_DATA_DIR.exists():
        logging.error(f"Directorio de anomalía {config.ANOMALY_DATA_DIR} no encontrado. Abortando.")
        raise FileNotFoundError(f"El directorio de datos de anomalía {config.ANOMALY_DATA_DIR} no existe.")

    # Cargar y truncar a la misma longitud que 'healthy' para consistencia
    anomaly_data_raw, _ = _load_data_from_dir_real(config.ANOMALY_DATA_DIR, config.N_SENSORS, max_len=max_len_healthy)
    if anomaly_data_raw is None:
        raise ValueError("Error fatal al cargar datos 'anomaly'.")

    # Reshape a [T, N, F=1]
    anomaly_data_full = anomaly_data_raw[:, :, np.newaxis]
    logging.info(f"Datos 'anomaly' cargados. Shape: {anomaly_data_full.shape}")

    # Liberar memoria
    del anomaly_data_raw
    gc.collect()

    # --- 3. Aplicar Ventaneo ---
    # Dividir 'healthy' en train/test (85/15 split)
    test_split_idx = int(len(healthy_data_full) * 0.85)
    train_data = healthy_data_full[:test_split_idx]
    test_data = healthy_data_full[test_split_idx:]

    # Usar un stride igual a window_size para ventanas no solapadas
    stride = config.WINDOW_SIZE

    data_dict = {
        'healthy_train': _helper_windowing(train_data, config.WINDOW_SIZE, stride),
        'healthy_test': _helper_windowing(test_data, config.WINDOW_SIZE, stride),
        'anomaly_test': _helper_windowing(anomaly_data_full, config.WINDOW_SIZE, stride)
    }

    logging.info(
        f"Datos ventaneados. healthy_train: {data_dict['healthy_train'].shape}, healthy_test: {data_dict['healthy_test'].shape}, anomaly_test: {data_dict['anomaly_test'].shape}")

    # Liberar memoria
    del healthy_data_full, train_data, test_data, anomaly_data_full
    gc.collect()

    return data_dict


# ###########################################################################
# --- MÓDULO 1: GRÁFICOS DE METODOLOGÍA Y WAVELETS (AVANZADO) ---
# ###########################################################################

def generate_advanced_module_1_plots(data: dict[str, np.ndarray], config: Config) -> None:
    """Genera figuras avanzadas (Q1) para metodología."""
    logging.info("--- Iniciando Módulo 1: Gráficos AVANZADOS de Metodología ---")
    module_key = "methodology"

    signal_healthy = data['healthy_test'][0, :, 0, 0]
    signal_anomaly = data['anomaly_test'][0, :, 0, 0]
    fs = 100  # Asumiendo 100Hz

    try:
        # --- Figura 1.1: Análisis Multidominio (Replica Fig1-2_spectral_analysis_clear.jpg) ---
        logging.info("Generando Figura 1.1: Multi-domain Signal Analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-domain Signal Analysis: Time, Frequency, and Energy', fontsize=18, weight='bold')

        # (a) Señal Temporal
        time = np.arange(len(signal_healthy)) / fs
        axes[0, 0].plot(time, signal_healthy, color='black', linewidth=1, alpha=0.9)
        axes[0, 0].set_title('Time Domain Signal (8 seconds)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Acceleration (m/s²)')
        axes[0, 0].set_xlim(0, 8)
        add_subfigure_label(axes[0, 0], 'a')

        # (b) FFT (Frequency Domain)
        fft_vals = np.abs(np.fft.fft(signal_healthy))
        freqs = np.fft.fftfreq(len(signal_healthy), 1 / fs)
        idx_pos = np.where(freqs >= 0)
        freqs_pos = freqs[idx_pos]
        fft_pos = fft_vals[idx_pos]

        axes[0, 1].plot(freqs_pos, fft_pos, color='blue', linewidth=1.5)
        axes[0, 1].set_title('Frequency Domain (FFT)')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('FFT Magnitude')
        axes[0, 1].set_xlim(0, 15)
        # Marcar picos (simulados para estilo, o detectados)
        peaks, _ =
        from scipy.signal import find_peaks
        (fft_pos, height=np.max(fft_pos) * 0.1)
        for p in peaks[:3]:  # Top 3 picos
            axes[0, 1].axvline(freqs_pos[p], color='red', linestyle='--', alpha=0.7)
            axes[0, 1].text(freqs_pos[p], fft_pos[p], f"{freqs_pos[p]:.1f} Hz",
                            ha='center', va='bottom', bbox=dict(facecolor='yellow', alpha=0.5))
        add_subfigure_label(axes[0, 1], 'b')

        # (c) PSD (Welch)
        f, Pxx = welch(signal_healthy, fs, nperseg=256)
        axes[1, 0].semilogy(f, Pxx, color='green', linewidth=1.5)
        axes[1, 0].set_title('PSD (Welch Method)')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].set_xlim(0, 15)
        add_subfigure_label(axes[1, 0], 'c')

        # (d) Energy Distribution (Bar Chart)
        # Calcular energía por banda wavelet (db4, level 5)
        coeffs = pywt.wavedec(signal_healthy, 'db4', level=5)
        energies = [np.sum(c ** 2) for c in coeffs]
        total_energy = sum(energies)
        percentages = [e / total_energy * 100 for e in energies]
        labels = ['A5\n0-1.56Hz', 'D5\n1.56-3.1Hz', 'D4\n3.1-6.2Hz', 'D3\n6.2-12.5Hz', 'D2\n12.5-25Hz', 'D1\n25-50Hz']
        # Colores estilo 'Paired' o similar
        colors = sns.color_palette("Paired", len(labels))

        bars = axes[1, 1].bar(labels, percentages, color=colors, edgecolor='black')
        axes[1, 1].set_title('Energy Distribution by Frequency Band')
        axes[1, 1].set_ylabel('Energy (%)')
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.1f}%', ha='center', va='bottom')
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, "1_1_Multidomain_Signal_Analysis", config)

        # --- Figura 1.2: Heatmap de Sensibilidad de Wavelets (NUEVO) ---
        logging.info("Generando Figura 1.2: Wavelet Sensitivity Heatmap...")

        families = ['db2', 'db4', 'db8', 'sym2', 'sym4', 'sym8', 'coif1', 'coif3', 'bior3.3', 'dmey']
        levels = ['D1', 'D2', 'D3', 'D4', 'D5']
        sensitivity_matrix = np.zeros((len(families), len(levels)))

        # Calcular sensibilidad: (Energía_Daño - Energía_Sano) / Energía_Sano
        for i, fam in enumerate(families):
            try:
                coeffs_h = pywt.wavedec(signal_healthy, fam, level=5)
                coeffs_a = pywt.wavedec(signal_anomaly, fam, level=5)
                # Detalles están en índices 1 a 5 (D5 a D1). Invertimos para D1 a D5
                details_h = coeffs_h[1:][::-1]
                details_a = coeffs_a[1:][::-1]

                for j in range(len(levels)):
                    e_h = np.sum(details_h[j] ** 2)
                    e_a = np.sum(details_a[j] ** 2)
                    sensitivity = (e_a - e_h) / e_h * 100  # % Cambio
                    sensitivity_matrix[i, j] = sensitivity
            except Exception as e:
                logging.warning(f"Wavelet {fam} falló: {e}")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sensitivity_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                    xticklabels=levels, yticklabels=families, ax=ax)
        ax.set_title("Wavelet Sensitivity Analysis (% Energy Increase on Damage)", fontsize=16, weight='bold')
        ax.set_xlabel("Decomposition Level")
        ax.set_ylabel("Wavelet Family")
        save_plot(fig, module_key, "1_2_Wavelet_Sensitivity_Heatmap", config)

        # --- Figura 1.3: Simulación Estocástica (Robustez al Ruido) (NUEVO) ---
        logging.info("Generando Figura 1.3: Stochastic Noise Robustness Simulation...")

        noise_levels = np.linspace(0, 0.5, 20)  # Niveles de ruido sigma
        n_iter = 50

        snr_results = []

        for noise in noise_levels:
            level_snrs = []
            for _ in range(n_iter):
                # Añadir ruido a la señal sana
                noisy_signal = signal_healthy + np.random.normal(0, noise, len(signal_healthy))

                # Reconstruir con db4 (la elegida)
                coeffs = pywt.wavedec(noisy_signal, 'db4', level=5)
                # Thresholding simple para denoising
                threshold = noise * np.sqrt(2 * np.log(len(noisy_signal))) if noise > 0 else 0
                coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
                rec_signal = pywt.waverec(coeffs_thresh, 'db4')
                rec_signal = rec_signal[:len(signal_healthy)]

                # Calcular SNR de reconstrucción
                mse = np.mean((signal_healthy - rec_signal) ** 2)
                if mse == 0: mse = 1e-10
                snr = 10 * np.log10(np.mean(signal_healthy ** 2) / mse)
                level_snrs.append(snr)
            snr_results.append(level_snrs)

        snr_results = np.array(snr_results)
        mean_snr = np.mean(snr_results, axis=1)
        std_snr = np.std(snr_results, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(noise_levels, mean_snr, 'b-', label='Mean Reconstruction SNR')
        ax.fill_between(noise_levels, mean_snr - std_snr, mean_snr + std_snr, color='blue', alpha=0.2,
                        label='SNR Variance (±1 std)')

        ax.set_title('Wavelet Denoising Robustness (Stochastic Simulation)', fontsize=16, weight='bold')
        ax.set_xlabel('Input Noise Level (Sigma)')
        ax.set_ylabel('Reconstruction SNR (dB)')
        ax.legend()
        ax.grid(True, linestyle=':')
        save_plot(fig, module_key, "1_3_Stochastic_Robustness", config)

        # --- Figura 1.4: Comparación de Escalogramas CWT (Replica Fig1-4) ---
        logging.info("Generando Figura 1.4: CWT Scalogram Comparison...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Continuous Wavelet Transform (CWT) Analysis: Healthy vs. Damaged', fontsize=18, weight='bold')

        t = np.arange(len(signal_healthy)) / fs
        scales = np.arange(1, 128)

        # CWT
        coeffs_h, freqs_h = pywt.cwt(signal_healthy, scales, 'morl', 1 / fs)
        coeffs_a, freqs_a = pywt.cwt(signal_anomaly, scales, 'morl', 1 / fs)

        # (a) Healthy Signal
        axes[0, 0].plot(t, signal_healthy, 'g-', linewidth=1.5)
        axes[0, 0].set_title('HEALTHY Signal', color='green', weight='bold')
        axes[0, 0].set_ylabel('Acceleration')
        axes[0, 0].set_xlabel('Time (s)')
        add_subfigure_label(axes[0, 0], 'a')

        # (b) Healthy Scalogram
        im1 = axes[0, 1].imshow(np.abs(coeffs_h), extent=[t[0], t[-1], freqs_h[-1], freqs_h[0]],
                                cmap='jet', aspect='auto', vmax=np.max(np.abs(coeffs_h)) * 0.9)
        axes[0, 1].set_title('HEALTHY Scalogram', color='green', weight='bold')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylim(0, 15)  # Zoom en frecuencias bajas
        plt.colorbar(im1, ax=axes[0, 1])
        add_subfigure_label(axes[0, 1], 'b')

        # (c) Damaged Signal
        axes[1, 0].plot(t, signal_anomaly, 'r-', linewidth=1.5)
        axes[1, 0].set_title('DAMAGED Signal', color='red', weight='bold')
        axes[1, 0].set_ylabel('Acceleration')
        axes[1, 0].set_xlabel('Time (s)')
        add_subfigure_label(axes[1, 0], 'c')

        # (d) Damaged Scalogram
        im2 = axes[1, 1].imshow(np.abs(coeffs_a), extent=[t[0], t[-1], freqs_a[-1], freqs_a[0]],
                                cmap='hot', aspect='auto', vmax=np.max(np.abs(coeffs_a)) * 0.9)
        axes[1, 1].set_title('DAMAGED Scalogram', color='red', weight='bold')
        axes[1, 1].set_ylabel('Frequency (Hz)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylim(0, 15)
        plt.colorbar(im2, ax=axes[1, 1])
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, "1_4_Scalogram_Comparison_Advanced", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos avanzados del Módulo 1: {e}", exc_info=True)


# ###########################################################################
# --- ORQUESTADOR PRINCIPAL (MAIN) ---
# ###########################################################################

def main():
    """Función principal que orquesta el pipeline."""

    config = Config()
    setup_environment(config)

    logging.info("=" * 80)
    logging.info("INICIANDO PIPELINE DE GENERACIÓN DE FIGURAS Q1 (MÓDULO 1 - AVANZADO)")
    logging.info(f"Proyecto: {config.BASE_DIR.name}")
    logging.info(f"Resultados se guardarán en: {config.FIGURE_DIRS['methodology']}")
    logging.info("=" * 80)

    try:
        # --- PASO 1: Cargar Datos (REAL) ---
        logging.info("--- Cargando y Preprocesando Datos (REAL) ---")
        data = load_shm_data(config)
        logging.info("Datos (REALES) cargados exitosamente.")
        gc.collect()

        # --- PASO 2: Generar Gráficos Avanzados ---
        generate_advanced_module_1_plots(data, config)

        logging.info("=" * 80)
        logging.info("GENERACIÓN DEL MÓDULO 1 (AVANZADO) COMPLETADA.")
        logging.info("=" * 80)

    except Exception as e:
        logging.critical(f"Error fatal en el pipeline: {e}", exc_info=True)
    finally:
        if logging.getLogger().hasHandlers():
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                    logging.getLogger().removeHandler(handler)
                except Exception:
                    pass


if __name__ == "__main__":
    main()