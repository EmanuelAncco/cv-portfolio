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
        1.  Heatmap de Sensibilidad de Familias Wavelet.
        2.  Simulación Estocástica de Robustez al Ruido (Monte Carlo).
        3.  Análisis Estadístico de Distribución de Coeficientes.

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
# --- MÓDULO 1: GRÁFICOS DE METODOLOGÍA Y WAVELETS ---
# ###########################################################################

def generate_signal_analysis_plot(data: dict[str, np.ndarray], config: Config) -> None:
    """Genera Figura 1.1: Análisis de Señal y Espectro (2x2)."""
    module_key = "methodology"
    try:
        logging.info("Generando Figura 1.1: Signal and Spectral Analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Signal and Spectral Analysis', fontsize=18, weight='bold')

        # (a) Señal Sana
        signal_healthy = data['healthy_test'][0, :, 0, 0]  # Sensor 0, sano
        axes[0, 0].plot(signal_healthy, label="Healthy Signal (Sensor 1)", color="C0", linewidth=1.5)
        axes[0, 0].set_title("Sample Healthy Signal (Time Domain)")
        axes[0, 0].set_xlabel(f"Time Samples (Window Size = {config.WINDOW_SIZE})")
        axes[0, 0].set_ylabel("Raw Acceleration")
        axes[0, 0].legend()
        add_subfigure_label(axes[0, 0], 'a')

        # (b) Señal Anómala
        signal_anomaly = data['anomaly_test'][0, :, 0, 0]  # Sensor 0, anómalo
        axes[0, 1].plot(signal_anomaly, label="Damaged Signal (Sensor 1)", color="C3", linewidth=1.5)
        axes[0, 1].set_title("Sample Damaged Signal (Time Domain)")
        axes[0, 1].set_xlabel(f"Time Samples (Window Size = {config.WINDOW_SIZE})")
        axes[0, 1].set_ylabel("Raw Acceleration")
        axes[0, 1].legend()
        add_subfigure_label(axes[0, 1], 'b')

        # (c) FFT de ambas
        fft_healthy = np.abs(np.fft.fft(signal_healthy))
        fft_anomaly = np.abs(np.fft.fft(signal_anomaly))
        freq = np.fft.fftfreq(len(signal_healthy))
        n = len(freq) // 2

        axes[1, 0].plot(freq[:n], fft_healthy[:n], label="Healthy FFT", color="C0", alpha=0.8)
        axes[1, 0].plot(freq[:n], fft_anomaly[:n], label="Damaged FFT", color="C3", alpha=0.8, linestyle='--')
        axes[1, 0].set_title("Frequency Spectrum (FFT)")
        axes[1, 0].set_xlabel("Normalized Frequency")
        axes[1, 0].set_ylabel("FFT Magnitude")
        axes[1, 0].legend()
        add_subfigure_label(axes[1, 0], 'c')

        # (d) PSD de ambas
        fs = 100  # Asumiendo 100Hz
        f_h, pxx_h = welch(signal_healthy, fs, nperseg=config.WINDOW_SIZE // 2)
        f_a, pxx_a = welch(signal_anomaly, fs, nperseg=config.WINDOW_SIZE // 2)

        axes[1, 1].semilogy(f_h, pxx_h, label="Healthy PSD", color="C0", alpha=0.8)
        axes[1, 1].semilogy(f_a, pxx_a, label="Damaged PSD", color="C3", alpha=0.8, linestyle='--')
        axes[1, 1].set_title("Power Spectral Density (Welch)")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("PSD (log scale)")
        axes[1, 1].legend()
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, "1_1_Signal_and_Spectral_Analysis", config)
    except Exception as e:
        logging.error(f"Error al generar Figura 1.1: {e}", exc_info=True)


def generate_wavelet_decomposition_plot(data: dict[str, np.ndarray], config: Config) -> None:
    """Genera Figura 1.2: Descomposición Wavelet (4x2)."""
    module_key = "methodology"
    try:
        logging.info("Generando Figura 1.2: Wavelet Decomposition...")
        signal_healthy = data['healthy_test'][0, :, 0, 0]  # Sensor 0, sano
        wavelet = 'db4'
        level = 5  # Coincide con tu hparam `wavelet_level`

        coeffs = pywt.wavedec(signal_healthy, wavelet, level=level)

        # --- CORRECCIÓN v10: Usar GridSpec(4, 2) ---
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Panel original (ocupa toda la fila 0)
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(signal_healthy, color="black", linewidth=1.5)
        ax0.set_title(f"Original Signal and DWT ('{wavelet}', L{level})", fontsize=16, weight='bold')
        ax0.set_ylabel("Amplitude")
        ax0.set_xticklabels([])
        add_subfigure_label(ax0, 'a')

        # Componentes wavelet
        # [A5, D5, D4, D3, D2, D1]
        plot_coeffs = [coeffs[0]] + list(coeffs[1:])
        plot_titles = [
            (f'Approximation A{level}', f'0-{100 / (2 ** (level + 1)):.2f} Hz'),
            (f'Detail D{level}', f'{100 / (2 ** (level + 1)):.2f}-{100 / (2 ** level):.2f} Hz'),
            (f'Detail D{level - 1}', f'{100 / (2 ** level):.2f}-{100 / (2 ** (level - 1)):.2f} Hz'),
            (f'Detail D{level - 2}', f'{100 / (2 ** (level - 1)):.2f}-{100 / (2 ** (level - 2)):.2f} Hz'),
            (f'Detail D{level - 3}', f'{100 / (2 ** (level - 2)):.2f}-{100 / (2 ** (level - 3)):.2f} Hz'),
            (f'Detail D{level - 4}', f'{100 / (2 ** (level - 3)):.2f}-{100 / (2 ** (level - 4)):.2f} Hz')
        ]
        colors_wavelet = ['#27ae60', '#2980b9', '#8e44ad', '#c0392b', '#d35400', '#7f8c8d']
        labels_wavelet = ['b', 'c', 'd', 'e', 'f', 'g']

        for i, (coeff, (title, freq_range), color, label) in enumerate(
                zip(plot_coeffs, plot_titles, colors_wavelet, labels_wavelet)):
            # --- CORRECCIÓN v10: Lógica de indexación para GridSpec(4, 2) ---
            row = (i // 2) + 1  # Empieza desde la fila 1 (la fila 0 es ax0)
            col = i % 2
            ax = fig.add_subplot(gs[row, col], sharex=ax0)

            # Reconstruir componente para que tenga la misma longitud
            coeff_list = [np.zeros_like(c) for c in coeffs]
            coeff_list[i] = coeff
            rec = pywt.waverec(coeff_list, wavelet)

            if IMPORTED_MODEL_HELPERS["DWT_adjust_length"]:
                rec = IMPORTED_MODEL_HELPERS["DWT_adjust_length"](rec, len(signal_healthy))
            else:
                logging.error("DWT_adjust_length no está disponible.")
                if len(rec) > len(signal_healthy): rec = rec[:len(signal_healthy)]

            ax.plot(rec, color=color, linewidth=1.5)
            ax.set_title(f"{title}\n({freq_range})", fontsize=14, weight='bold', color=color)
            ax.set_ylabel("Amplitude", fontsize=12, weight='bold')
            if row == 3:  # Última fila de plots
                ax.set_xlabel("Time Samples")
            else:
                ax.set_xticklabels([])
            add_subfigure_label(ax, label)

        save_plot(fig, module_key, "1_2_Wavelet_Decomposition", config)

    except Exception as e:
        logging.error(f"Error al generar Figura 1.2: {e}", exc_info=True)


def generate_graph_topology_plot(config: Config) -> None:
    """Genera Figura 1.3: Comparación de Topologías de Grafo (1x2)."""
    module_key = "methodology"
    try:
        logging.info("Generando Figura 1.3: Graph Topology Comparison...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('Graph Topology Comparison', fontsize=18, weight='bold')

        # (a) Topología del Grafo Binario
        ax_bin = axes[0]
        if IMPORTED_MODEL_HELPERS["GnnBaseGraphDef"] is None:
            logging.warning("No se pudo cargar `define_bridge_graph`. Omitiendo gráfico 3.2.")
            ax_bin.set_title("Binary Graph (Load Failed)")
        else:
            edge_index_base = IMPORTED_MODEL_HELPERS["GnnBaseGraphDef"]().cpu().numpy().T
            G_base = nx.Graph()
            G_base.add_edges_from(edge_index_base)
            G_base.add_nodes_from(range(config.N_SENSORS))
            pos_base = {0: (0, 0), 1: (1, 0.1), 2: (1, -0.1), 3: (2, 0.1), 4: (2, -0.1)}
            labels = {i: f"S{i + 1}" for i in G_base.nodes()}

            nx.draw(G_base, pos_base, with_labels=True, labels=labels, node_color='#8e44ad',  # Color M2
                    node_size=2500, font_size=15, font_weight='bold', edge_color='gray', width=3, ax=ax_bin,
                    font_color='white')
            ax_bin.set_title("Binary Adjacency Graph (M2: GNN Original)")
            add_subfigure_label(ax_bin, 'a')

        # (b) Topología del Grafo Físico
        ax_phys = axes[1]
        if IMPORTED_MODEL_HELPERS["PhysicsGraphDef"] is None:
            logging.warning("No se pudo cargar `create_physics_informed_graph`. Omitiendo gráfico 3.3.")
            ax_phys.set_title("Physics-Informed Graph (Load Failed)")
        else:
            physics_graph = IMPORTED_MODEL_HELPERS["PhysicsGraphDef"](num_nodes=config.N_SENSORS)
            edge_index = physics_graph['edge_index'].t().cpu().numpy()
            edge_weights = physics_graph['edge_weight'].cpu().numpy()

            G_phys = nx.Graph()
            G_phys.add_nodes_from(range(config.N_SENSORS))
            coords = {
                0: [13.88, -4.0], 1: [13.88, 4.0], 2: [27.76, -4.0],
                3: [27.76, 4.0], 4: [41.64, 0.0]
            }
            pos_phys = {i: (coords[i][0], coords[i][1]) for i in range(config.N_SENSORS)}
            labels_phys = {i: f"S{i + 1}" for i in G_phys.nodes()}

            unique_edges = {}
            for (u, v), w in zip(edge_index, edge_weights):
                if (v, u) not in unique_edges:
                    G_phys.add_edge(u, v, weight=w)
                    unique_edges[(u, v)] = w

            weights = [G_phys[u][v]['weight'] for u, v in G_phys.edges()]
            max_w, min_w = max(weights), min(weights)
            norm_weights = [(w - min_w) / (max_w - min_w + 1e-6) * 8 + 2 for w in weights]  # Grosor 2 a 10

            nx.draw(
                G_phys, pos_phys, labels=labels_phys, with_labels=True,
                node_color='#c0392b',  # Color M4
                node_size=2500, font_size=15, font_weight='bold',
                edge_color=weights, edge_cmap=plt.cm.viridis, width=norm_weights,
                ax=ax_phys, font_color='white'
            )

            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=min_w, vmax=max_w))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_phys, shrink=0.8)
            cbar.set_label('Edge Weight (1/distance)')
            ax_phys.set_title("Physics-Informed Graph (M4: PI-STG-AE)")
            add_subfigure_label(ax_phys, 'b')

        save_plot(fig, module_key, "1_3_Graph_Comparison", config)
    except Exception as e:
        logging.error(f"Error al generar Figura 1.3: {e}", exc_info=True)


def generate_scalogram_plot(data: dict[str, np.ndarray], config: Config) -> None:
    """Genera Figura 1.4: Comparación de Escalogramas CWT (2x2)."""
    module_key = "methodology"
    try:
        logging.info("Generando Figura 1.4: CWT Scalogram Comparison...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Scalogram Comparison (Healthy vs. Damaged)', fontsize=18, weight='bold')

        signal_healthy = data['healthy_test'][0, :, 0, 0]
        signal_damaged = data['anomaly_test'][0, :, 0, 0]
        t = np.arange(config.WINDOW_SIZE) / 100.0  # Asumir 100 Hz

        scales = np.arange(1, 128)
        coeffs_h, freqs_h = pywt.cwt(signal_healthy, scales, 'morl', 1 / 100)
        coeffs_d, freqs_d = pywt.cwt(signal_damaged, scales, 'morl', 1 / 100)

        # (a) Healthy Signal
        axes[0, 0].plot(t, signal_healthy, 'g-', linewidth=1.5, alpha=0.9)
        axes[0, 0].set_xlabel('Time (s)', fontsize=13, weight='bold')
        axes[0, 0].set_ylabel('Acceleration (m/s²)', fontsize=13, weight='bold')
        axes[0, 0].set_title('HEALTHY Signal', fontsize=15, weight='bold', color='green')
        axes[0, 0].grid(True, alpha=0.4, linewidth=1.5)
        add_subfigure_label(axes[0, 0], 'a')

        # (b) Healthy Scalogram
        im1 = axes[0, 1].imshow(np.abs(coeffs_h), extent=[t[0], t[-1], freqs_h[-1], freqs_h[0]],
                                cmap='jet', aspect='auto', vmax=np.max(np.abs(coeffs_h)) * 0.9)
        axes[0, 1].set_xlabel('Time (s)', fontsize=13, weight='bold')
        axes[0, 1].set_ylabel('Frequency (Hz)', fontsize=13, weight='bold')
        axes[0, 1].set_title('HEALTHY Scalogram', fontsize=15, weight='bold', color='green')
        axes[0, 1].set_ylim(0, 15)
        plt.colorbar(im1, ax=axes[0, 1])
        add_subfigure_label(axes[0, 1], 'b')

        # (c) Damaged Signal
        axes[1, 0].plot(t, signal_damaged, 'r-', linewidth=1.5, alpha=0.9)
        axes[1, 0].set_xlabel('Time (s)', fontsize=13, weight='bold')
        axes[1, 0].set_ylabel('Acceleration (m/s²)', fontsize=13, weight='bold')
        axes[1, 0].set_title('DAMAGED Signal', fontsize=15, weight='bold', color='red')
        axes[1, 0].grid(True, alpha=0.4, linewidth=1.5)
        add_subfigure_label(axes[1, 0], 'c')

        # (d) Damaged Scalogram
        im2 = axes[1, 1].imshow(np.abs(coeffs_d), extent=[t[0], t[-1], freqs_d[-1], freqs_d[0]],
                                cmap='hot', aspect='auto', vmax=np.max(np.abs(coeffs_d)) * 0.9)
        axes[1, 1].set_xlabel('Time (s)', fontsize=13, weight='bold')
        axes[1, 1].set_ylabel('Frequency (Hz)', fontsize=13, weight='bold')
        axes[1, 1].set_title('DAMAGED Scalogram', fontsize=15, weight='bold', color='red')
        axes[1, 1].set_ylim(0, 15)
        plt.colorbar(im2, ax=axes[1, 1])
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, "1_4_Scalogram_Comparison", config)

    except Exception as e:
        logging.error(f"Error al generar Figura 1.4: {e}", exc_info=True)


# ###########################################################################
# --- ORQUESTADOR PRINCIPAL (MAIN) ---
# ###########################################################################

def main():
    """Función principal que orquesta el pipeline de generación de figuras."""

    config = Config()
    setup_environment(config)

    logging.info("=" * 80)
    logging.info("INICIANDO PIPELINE DE GENERACIÓN DE FIGURAS Q1 (MÓDULO 1)")
    logging.info(f"Proyecto: {config.BASE_DIR.name}")
    logging.info(f"Resultados se guardarán en: {config.FIGURE_DIRS['methodology']}")
    logging.info("=" * 80)

    try:
        # --- PASO 1: Cargar Clases de Modelo (REAL) ---
        logging.info("--- PASO 1/3: Verificando Clases de Modelo .py ---")
        if not all(IMPORTED_MODEL_HELPERS.values()):
            logging.critical(
                "Una o más clases helper (DWT, Graph) no se pudieron importar. Revisa los logs. Abortando.")
            return
        logging.info("Todas las clases helper importadas exitosamente.")

        # --- PASO 2: Cargar Datos (REAL) ---
        logging.info("--- PASO 2/3: Cargando y Preprocesando Datos (REAL) ---")
        data = load_shm_data(config)
        logging.info("Datos (REALES) cargados exitosamente.")
        gc.collect()

        # --- PASO 3: Generar Gráficos por Módulo ---
        logging.info("--- PASO 3/3: Iniciando Generación de Gráficos (Módulo 1) ---")

        generate_signal_analysis_plot(data, config)
        generate_wavelet_decomposition_plot(data, config)
        generate_graph_topology_plot(config)
        generate_scalogram_plot(data, config)

        logging.info("=" * 80)
        logging.info("GENERACIÓN DEL MÓDULO 1 COMPLETADA.")
        logging.info(f"Todas las figuras se han guardado en: {config.FIGURE_DIRS['methodology']}")
        logging.info(f"El log de esta ejecución está en: {config.OUTPUT_DIR / 'figure_generation_MOD_1.log'}")
        logging.info("=" * 80)

    except Exception as e:
        logging.critical(f"Error fatal en el pipeline principal (Módulo 1): {e}", exc_info=True)
    finally:
        if logging.getLogger().hasHandlers():
            # Asegurarse de que todos los handlers se cierren
            for handler in logging.getLogger().handlers[:]:
                try:
                    handler.close()
                    logging.getLogger().removeHandler(handler)
                except Exception:
                    pass  # Ignorar errores al cerrar el log


if __name__ == "__main__":
    main()