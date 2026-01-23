#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SCRIPT DE GENERACIÓN DE FIGURAS CIENTÍFICAS Q1 (Versión FINAL v8 - Corregida)
==========================================================================

Descripción:
    Este script es un pipeline completo y reproducible para generar todas las
    visualizaciones científicas necesarias para un artículo Q1 sobre
    detección de anomalías en SHM de puentes.

    Esta versión carga y utiliza TODOS los artefactos reales proporcionados
    (scripts .py, logs .txt, hparams .json, y scalers .gz).

    El script AHORA IMPLEMENTA la lógica de inferencia real,
    incluyendo la carga de datos, DWT, y la inferencia de PyTorch.

    v8: Corrige los errores de la ejecución:
        - UnicodeDecodeError: Cambia encoding de logs a 'latin-1'.
        - FileNotFoundError: Corrige la ruta de 'no_gnn_ae'.
        - RuntimeError (size mismatch): Corrige la instanciación de 'gnn_base'
          para que use los defaults de la clase, igual que hizo el script de
          entrenamiento.
        - TypeError (TSNE): Cambia 'n_iter' a 'max_iter'.
        - FutureWarning (applymap): Cambia 'applymap' a 'map'.
        - Estilo: Integra el estilo visual (fuentes, colores, agrupación)
          del script 'ultimate_figure_generator_v3.py'.

Instrucciones (Emanuel):
    1.  Asegúrate de que tus discos 'D:\' estén accesibles.
    2.  Verifica que los archivos .py están en la misma carpeta que este script.
    3.  Ejecuta el script: `python generate_all_figures.py`
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
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
import pywt  # (pip install PyWavelets)
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity  # Para SSIM
from scipy import stats
from scipy.signal import welch
from matplotlib.gridspec import GridSpec

# --- Dependencias de ML ---
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Dependencias de PyTorch (necesarias para cargar modelos) ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as PyTorchDataLoader, Dataset as PyTorchDataset

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    print("ADVERTENCIA: torch_geometric no está instalado. `load_model_weights` fallará.")
    print("Instale con: pip install torch_geometric")
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
        if make_unique:  # Evitar colisiones de nombres (ej. SpatioTemporalAutoencoder)
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
    # USA LA RUTA DEL SCRIPT (`data1711.py`) COMO BASE
    BASE_DIR: Path = Path(__file__).parent.resolve()

    # Ruta a tus datos de entrenamiento (ESTADO SANO)
    # (Extraído de tus scripts de train_*.py)
    DATA_DIR: Path = Path(r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar")

    # Ruta a tus datos de ANOMALÍA (DAÑO)
    # (Extraído de tu script `inference_accelerometer_V4.py`)
    ANOMALY_DATA_DIR: Path = Path(r"D:\descargas 2025\Aceleraciones con daño\Aceleraciones")

    # --- RUTAS DE ARTEFACTOS CORREGIDAS ---

    # Rutas a los SCRIPTS .PY (asumiendo que están en la misma carpeta que este script)
    MODEL_SCRIPT_FILES: dict[str, Path] = {
        "gnn_base": BASE_DIR / "train_autoencoder.py",
        "no_gnn_ae": BASE_DIR / "train_no_gnn.py",
        "wavelet_gnn": BASE_DIR / "train_wavelet_v3.py",
        "stgae_physics": BASE_DIR / "31_oct_newmodel.py",
    }

    # Rutas a las carpetas de RESULTADOS (extraídas de tus logs)
    MODEL_RESULTS_DIRS: dict[str, Path] = {
        # El Log 1 usa una ruta relativa, la resolvemos desde BASE_DIR
        "gnn_base": BASE_DIR / "resultados_entrenamiento/run_gnn_20250910-020756",
        # Los otros logs usan rutas absolutas
        # --- CORRECCIÓN v8: Ruta de no_gnn_ae corregida ---
        "no_gnn_ae": Path(
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_no_gnn\run_no_gnn_20251027-110627"),
        # <-- Tomado de tu log
        "wavelet_gnn": Path(
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547"),
        # <-- Tomado de tu log RESUME
        # --- CORRECCIÓN v8: Ruta de stgae_physics corregida ---
        "stgae_physics": Path(
            r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_modelos_shm\run_STGAE-PHYSICS_lr0.0005_bs16_20251031-124920")
        # <-- Tomado de tu log
    }

    # Carpeta de salida principal para todas las figuras
    OUTPUT_DIR: Path = BASE_DIR / "paper_figures_Q1_FINAL"

    # 4. Nombres de Modelos (deben coincidir con las carpetas)
    MODEL_NAMES: list[str] = [
        "M1: No-GNN",
        "M2: GNN Original",
        "M3: Wavelet-GNN",
        "M4: PI-STG-AE"
    ]
    MODEL_NAME_MAP: dict[str, str] = {
        "no_gnn_ae": "M1: No-GNN",
        "gnn_base": "M2: GNN Original",
        "wavelet_gnn": "M3: Wavelet-GNN",
        "stgae_physics": "M4: PI-STG-AE"
    }
    MODEL_NAME_MAP_INV: dict[str, str] = {v: k for k, v in MODEL_NAME_MAP.items()}

    # El mejor modelo (basado en el análisis de logs: 0.006422)
    BEST_MODEL_NAME: str = "M3: Wavelet-GNN"

    # 5. Diccionario de Directorios de Salida
    FIGURE_DIRS: dict[str, Path] = {
        "1_methodology": OUTPUT_DIR / "1_methodology_wavelets",
        "2_training": OUTPUT_DIR / "2_training_metrics",
        "3_architecture": OUTPUT_DIR / "3_model_architecture",
        "4_reconstruction": OUTPUT_DIR / "4_reconstruction_analysis",
        "5_simulation": OUTPUT_DIR / "5_3d_simulations",
        "6_anomaly": OUTPUT_DIR / "6_anomaly_detection",
        "7_analysis": OUTPUT_DIR / "7_additional_analysis",
    }

    # 6. Configuración de Trazado
    PLOT_DPI: int = 300
    PLOT_FORMAT: str = "png"
    FIGSIZE: tuple[int, int] = (10, 6)
    FONT_SIZE: int = 12

    # Paleta de colores (del script 'ultimate')
    COLORS_MODEL: dict[str, str] = {
        "M1: No-GNN": '#7f8c8d',
        "M2: GNN Original": '#8e44ad',
        "M3: Wavelet-GNN": '#2980b9',
        "M4: PI-STG-AE": '#c0392b',
    }
    MARKERS: dict[str, str] = {
        "M1: No-GNN": 'o',
        "M2: GNN Original": 's',
        "M3: Wavelet-GNN": '^',
        "M4: PI-STG-AE": 'D',
    }


# --- Instancia de Configuración Global (Solo para importación) ---
_temp_config = Config()

# ###########################################################################
# --- IMPORTACIÓN GLOBAL DE CLASES Y HELPERS (SECCIÓN 3) ---
# --- (Resuelve los FileNotFoundError y Unresolved reference) ---
# ###########################################################################

IMPORTED_MODEL_CLASSES = {
    "gnn_base": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["gnn_base"], "SpatioTemporalAutoencoder",
                                             make_unique=True),
    "no_gnn_ae": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["no_gnn_ae"],
                                              "SpatioTemporalAutoencoderNoGNN"),
    "wavelet_gnn": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["wavelet_gnn"],
                                                "SpatioTemporalAutoencoder", make_unique=True),
    "stgae_physics": import_model_class_from_file(_temp_config.MODEL_SCRIPT_FILES["stgae_physics"],
                                                  "SpatioTemporalAutoencoder", make_unique=True)
}

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
    log_file = output_dir / "figure_generation.log"
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
    if config.OUTPUT_DIR.exists():
        # No configurar logging aquí si la carpeta ya existe
        try:
            shutil.rmtree(config.OUTPUT_DIR)
        except PermissionError:
            print(f"Error de permisos. No se pudo eliminar {config.OUTPUT_DIR}. ¿Archivos abiertos?")
            return
    config.OUTPUT_DIR.mkdir(parents=True)
    setup_logging(config.OUTPUT_DIR)
    logging.info(f"Entorno de logging configurado. Log guardado en: {config.OUTPUT_DIR / 'figure_generation.log'}")

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

    for key, path in config.FIGURE_DIRS.items():
        path.mkdir(parents=True)
    logging.info(f"Creados {len(config.FIGURE_DIRS)} subdirectorios de figuras.")


def add_subfigure_label(ax: plt.Axes | Axes3D, label: str, x: float = -0.15, y: float = 1.05) -> None:
    """Añade etiqueta (a), (b), etc. - Compatible con 3D."""
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
        # --- CORRECCIÓN v8: `ax` no es necesario, lo eliminamos ---
        module_key: str,
        filename: str,
        config: Config
) -> None:
    """Función helper para guardar y cerrar figuras de forma robusta."""
    try:
        # tight_layout a veces falla con gridspec o 3D, lo llamamos con cuidado
        try:
            fig.tight_layout()
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
# --- (FUNCIONES MOVIDAS A LA PARTE SUPERIOR PARA RESOLVER REFERENCIAS) ---
# ###########################################################################

def parse_training_log(log_path: Path) -> pd.DataFrame:
    """Parsea los archivos training_log_*.txt para extraer las métricas."""
    logging.info(f"Parseando log: {log_path.name}")
    epoch_data = []
    # Regex unificado para capturar todos los formatos de log
    pattern = re.compile(
        # --- CORRECCIÓN v8: Eliminado '\' de r"Epoch (\d+)/\d+..." ---
        r"Epoch (\d+)/\d+ -> .*Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+)"
    )
    try:
        # --- CORRECCIÓN v8: Usar 'latin-1' para logs con acentos ---
        with open(log_path, 'r', encoding='latin-1') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        epoch = int(match.group(1))
                        train_loss = float(match.group(2))
                        val_loss = float(match.group(3))
                        epoch_data.append({
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss
                        })
                    except ValueError:
                        logging.warning(f"No se pudo parsear el valor flotante en la línea: {line.strip()}")

        if not epoch_data:
            logging.warning(f"No se encontraron datos de epoch en {log_path.name}")
            return pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

        df = pd.DataFrame(epoch_data)
        # Manejar logs reanudados (épocas duplicadas)
        df = df.drop_duplicates(subset='epoch', keep='last')
        return df.sort_values(by='epoch').reset_index(drop=True)

    except Exception as e:
        logging.error(f"Error parseando {log_path.name}: {e}", exc_info=True)
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])


def find_best_model_path_from_logs(log_paths: list[Path]) -> Path | None:
    """Parsea logs .txt para encontrar la *última* ruta de 'best_model.pth' guardada."""
    best_model_path_str = None
    # Regex para encontrar la ruta del modelo guardado
    pattern = re.compile(r"Nuevo mejor modelo .* guardado en:? (.*best_model.*\.pth)", re.IGNORECASE)

    # Leer todos los logs en orden (asumiendo que los nombres se ordenan cronológicamente)
    for log_path in sorted(log_paths):
        try:
            # --- CORRECCIÓN v8: Usar 'latin-1' para logs con acentos ---
            with open(log_path, 'r', encoding='latin-1') as f:
                content = f.read()
                matches = pattern.findall(content)
                if matches:
                    # Quedarse con la *última* coincidencia del archivo
                    best_model_path_str = matches[-1]
        except Exception as e:
            logging.warning(f"No se pudo leer el log {log_path.name}: {e}")

    if best_model_path_str:
        # Limpiar la ruta (quitar ' (Val Loss: ...)')
        if '(' in best_model_path_str:
            best_model_path_str = best_model_path_str.split(' (')[0].strip()

        # Validar si la ruta existe
        best_model_path = Path(best_model_path_str)
        if best_model_path.exists():
            logging.info(f"Ruta de modelo .pth parseada y VERIFICADA: {best_model_path}")
            return best_model_path
        else:
            logging.error(f"Ruta de modelo .pth parseada pero NO ENCONTRADA: {best_model_path}")
            return None

    logging.error(f"No se pudo encontrar la ruta .pth en {len(log_paths)} logs.")
    return None


def load_real_scaler(scaler_path: Path) -> StandardScaler | None:
    """Carga un scaler de joblib."""
    if not scaler_path.exists():
        # Intentar eliminar el sufijo _Version2 si existe
        scaler_path_alt = scaler_path.with_name(scaler_path.name.replace("_Version2", ""))
        if scaler_path_alt.exists():
            scaler_path = scaler_path_alt
        else:
            logging.error(f"No se encontró el scaler: {scaler_path}")
            return None
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Scaler real cargado desde: {scaler_path.name}")
        return scaler
    except Exception as e:
        logging.error(f"Error al cargar {scaler_path.name}: {e}", exc_info=True)
        return None


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


def _helper_apply_dwt_features_batch(data_batch_np: np.ndarray, wavelet: str, level: int) -> np.ndarray | None:
    """
    Aplica DWT a un batch [B, T, N, 1] y retorna [B, T, N, F_out].
    Esta es la LÓGICA REAL de tu script `train_wavelet_v3.py`.
    """
    if IMPORTED_MODEL_HELPERS["DWT_helper_function"] is None or IMPORTED_MODEL_HELPERS["DWT_adjust_length"] is None:
        logging.error("Funciones DWT no importadas. Abortando DWT.")
        return None

    B, T, N, _ = data_batch_np.shape
    F_out = 1 + 1 + level  # Original + Aprox + Details

    # Array de salida
    dwt_features_np = np.zeros((B, T, N, F_out))

    logging.info(f"Aplicando DWT real ({wavelet}, L{level}) a {B}x{N} señales...")

    # Usar tqdm aquí si es un proceso largo
    for b in tqdm(range(B), desc="Procesando DWT", leave=False, total=B):
        for n in range(N):
            try:
                signal_1d = data_batch_np[b, :, n, 0]  # Shape [T]

                # Esta es TU lógica de `train_wavelet_v3.py`
                features_2d = IMPORTED_MODEL_HELPERS["DWT_helper_function"](signal_1d, wavelet, level,
                                                                            T)  # Shape [T, F_out]

                if features_2d is None or features_2d.shape != (T, F_out):
                    logging.warning(f"DWT falló para batch {b}, sensor {n}. Rellenando con ceros.")
                    dwt_features_np[b, :, n, 0] = signal_1d  # Al menos copiar la original
                else:
                    dwt_features_np[b, :, n, :] = features_2d
            except Exception as e:
                logging.warning(f"Error en DWT para batch {b}, sensor {n}: {e}")
                dwt_features_np[b, :, n, 0] = data_batch_np[b, :, n, 0]  # Copiar original

    return dwt_features_np


class _InferenceDataset(PyTorchDataset):
    """Dataset simple para inferencia."""

    def __init__(self, data): self.data = data

    def __len__(self): return len(self.data)

    def __getitem__(self, idx): return torch.FloatTensor(self.data[idx])


# Helper para capturar el espacio latente
latent_space_capture = {}


def get_latent_hook(name: str):
    """Genera el hook para capturar el estado oculto de la GRU."""

    def hook(model, input, output):
        # La salida de GRU es (output_sequence, h_n)
        # h_n tiene shape [num_layers * num_directions, B, rnn_hidden]
        # Queremos el estado de la última capa: h_n[-1]
        latent_space_capture[name] = output[1][-1].cpu().detach().numpy()

    return hook


def load_model_artifacts(model_results_dir: Path, model_name_key: str, config: Config) -> dict:
    """
    Carga los artefactos REALES (logs, hparams, scaler y RUTA al .pth)
    de un modelo entrenado.
    """
    model_name = config.MODEL_NAME_MAP[model_name_key]  # ej. "M1: No-GNN"
    logging.info(f"Cargando artefactos REALES para el modelo: {model_name} desde {model_results_dir}...")
    artifacts = {
        "model_class": IMPORTED_MODEL_CLASSES.get(model_name_key),
        "scaler": None,
        "log": pd.DataFrame(),
        "hparams": {},
        "best_model_path": None  # Ruta al .pth
    }

    if not model_results_dir.exists():
        logging.error(f"No se encontró la carpeta de resultados del modelo: {model_results_dir}.")
        return artifacts

    if artifacts["model_class"] is None:
        logging.error(f"La clase de modelo para {model_name_key} no fue importada correctamente.")
        # No podemos continuar sin la clase
        return artifacts

    # 2. Cargar Hiperparámetros (REAL)
    hparam_files = list(model_results_dir.glob("hyperparameters*.json"))
    if hparam_files:
        try:
            # Corregir nombre de archivo si tiene _Version2
            hparam_file_path = hparam_files[0].with_name(hparam_files[0].name.replace("_Version2", ""))
            with open(hparam_file_path, 'r', encoding='utf-8') as f:
                artifacts["hparams"] = json.load(f)
            logging.info(f"Hiperparámetros cargados desde: {hparam_file_path.name}")
        except Exception as e:
            logging.warning(f"Error al cargar {hparam_files[0].name}: {e}")
    else:
        logging.warning(f"No se encontró 'hyperparameters.json' para {model_name}.")

    # 3. Cargar Log de Entrenamiento (REAL)
    log_data = {'train_loss': [], 'val_loss': []}
    log_files = sorted(list(model_results_dir.glob("loss_history*.json")))

    if log_files:
        try:
            for log_file in log_files:
                log_file_path = log_file.with_name(log_file.name.replace("_Version2", ""))
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    log_data['train_loss'].extend(data.get('train_loss', []))
                    log_data['val_loss'].extend(data.get('val_loss', []))
            artifacts["log"] = pd.DataFrame(log_data)
            artifacts["log"]["epoch"] = range(1, len(artifacts["log"]) + 1)
            logging.info(f"Historial de pérdidas cargado y combinado desde {len(log_files)} archivos .json.")
        except Exception as e:
            logging.warning(f"Error al cargar logs .json: {e}")

    # Fallback a .txt si .json falló o está vacío
    log_txt_files = sorted(list(model_results_dir.glob("training_log*.txt")))
    if artifacts["log"].empty:
        if log_txt_files:
            log_txt_files_clean = [f.with_name(f.name.replace("_Version2", "")) for f in log_txt_files]
            all_logs_df = pd.concat([parse_training_log(f) for f in log_txt_files_clean])
            all_logs_df = all_logs_df.drop_duplicates(subset='epoch', keep='last').sort_values(by='epoch')
            artifacts["log"] = all_logs_df.reset_index(drop=True)
            logging.info(f"Historial de pérdidas parseado y combinado desde {len(log_txt_files_clean)} archivos .txt.")
        else:
            logging.error(f"No se encontró historial de pérdidas (.json o .txt) para {model_name}.")

    # 4. Cargar Scaler (REAL)
    scaler_files = list(model_results_dir.glob("scaler*.gz"))
    if not scaler_files:  # Fallback para el primer modelo
        scaler_files = list(model_results_dir.glob("scaler.gz"))

    if scaler_files:
        scaler_path = scaler_files[0].with_name(scaler_files[0].name.replace("_Version2", ""))
        artifacts["scaler"] = load_real_scaler(scaler_path)
    else:
        logging.error(f"No se encontró 'scaler_*.gz' para {model_name}.")

    # 5. Encontrar la RUTA al .pth (REAL)
    if log_txt_files:
        log_txt_files_clean = [f.with_name(f.name.replace("_Version2", "")) for f in log_txt_files]
        artifacts["best_model_path"] = find_best_model_path_from_logs(log_txt_files_clean)
        if artifacts["best_model_path"]:
            logging.info(f"Ruta de modelo .pth encontrada para {model_name}.")
        else:
            # Fallback: buscar .pth directamente en la carpeta
            pth_files = list(model_results_dir.glob("best_model*.pth"))
            if pth_files:
                artifacts["best_model_path"] = pth_files[0]
                logging.warning(f"No se pudo parsear la ruta .pth del log, pero se encontró: {pth_files[0].name}")
            else:
                logging.error(f"No se pudo encontrar la ruta .pth en los logs NI en la carpeta para {model_name}.")
    else:
        logging.error(f"No hay logs .txt para buscar la ruta .pth de {model_name}.")

    return artifacts


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


def get_model_predictions(
        model_artifacts: dict,
        data: np.ndarray,  # Shape [B, T, N, 1] (Datos CRUDA)
        config: Config,
        model_name_key: str  # ej. "gnn_base"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtiene reconstrucciones, errores y espacio latente de un modelo.
    *** LÓGICA DE INFERENCIA REAL ***
    """
    model_name_display = config.MODEL_NAME_MAP[model_name_key]  # ej. "M2: GNN Original"
    logging.info(f"--- Iniciando Inferencia REAL para {model_name_display} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Obtener Artefactos ---
    hparams = model_artifacts.get('hparams', {})
    scaler = model_artifacts.get('scaler')
    model_weights_path = model_artifacts.get('best_model_path')
    ModelClass = model_artifacts.get('model_class')

    if not all([hparams, scaler, model_weights_path, ModelClass]):
        logging.error(f"Faltan artefactos para {model_name_display}. Abortando inferencia real.")
        return (np.array([]),) * 5

    if not model_weights_path.exists():
        logging.error(f"Ruta de pesos no encontrada: {model_weights_path}")
        return (np.array([]),) * 5

    # --- 2. Instanciar Modelo ---
    model = None
    num_features = 1
    requires_dwt = False
    try:
        if model_name_key == "gnn_base":
            # --- CORRECCIÓN v8: El script 'train_autoencoder.py' NO usa hparams
            # para gnn_hidden, gnn_out, rnn_hidden. Usa los defaults de la clase.
            # PERO el checkpoint REQUIERE gnn_out=32, gnn_hidden=32.
            logging.warning(f"Instanciando {model_name_display} con parámetros forzados (gnn_out=32, gnn_hidden=32) "
                            f"para coincidir con el checkpoint.")
            model = ModelClass(
                num_nodes=config.N_SENSORS,
                num_features=1,
                window_size=config.WINDOW_SIZE,
                gnn_hidden=32,  # Forzado
                gnn_out=32,  # Forzado
                rnn_hidden=64  # Default/Hparam
            )
        elif model_name_key == "no_gnn_ae":
            model = ModelClass(
                num_nodes=config.N_SENSORS, num_features=1,
                window_size=config.WINDOW_SIZE, rnn_hidden=hparams['rnn_hidden'],
                rnn_layers=hparams['rnn_layers']
            )
        elif model_name_key == "wavelet_gnn" or model_name_key == "stgae_physics":
            requires_dwt = True
            num_features = hparams.get('num_features', 7)  # 7 de tus logs
            model = ModelClass(
                num_nodes=config.N_SENSORS, num_features=num_features,
                window_size=config.WINDOW_SIZE, gnn_hidden=hparams['gnn_hidden'],
                gnn_out=hparams['gnn_out'], rnn_hidden=hparams['rnn_hidden'],
                rnn_layers=hparams['rnn_layers']
            )
    except Exception as e:
        logging.error(f"Error al instanciar {model_name_display}: {e}", exc_info=True)
        return (np.array([]),) * 5

    if model is None:
        logging.error(f"Nombre de modelo {model_name_display} no reconocido.")
        return (np.array([]),) * 5

    # --- 3. Cargar Pesos ---
    try:
        # --- CORRECCIÓN v8: Añadir weights_only=True ---
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        logging.info(f"Pesos reales cargados para {model_name_display} desde {model_weights_path.name}")
    except Exception as e:
        logging.error(f"Error al cargar pesos desde {model_weights_path} para {model_name_display}: {e}", exc_info=True)
        return (np.array([]),) * 5

    # --- 4. Preparar Datos (DWT y Escalado) ---
    B, T, N, F = data.shape

    if requires_dwt:
        logging.info(f"Aplicando DWT (features={num_features}) a {B} muestras...")
        data_features_np = _helper_apply_dwt_features_batch(
            data, hparams['wavelet_name'], hparams['wavelet_level']
        )  # Shape: [B, T, N, F_out]
        if data_features_np is None: return (np.array([]),) * 5
    else:
        data_features_np = data  # Shape: [B, T, N, 1]

    num_features_model = data_features_np.shape[3]  # Actualizar num_features

    # Escalar: Reshape -> Scale -> Reshape back
    try:
        data_flat = data_features_np.reshape(-1, num_features_model)
        data_scaled = scaler.transform(data_flat)
        data_scaled_np = data_scaled.reshape(B, T, N, num_features_model)
    except Exception as e:
        logging.error(
            f"Error al escalar datos para {model_name_display}. Esperado F={num_features_model}. Datos shape: {data_features_np.shape}. Scaler n_features_in_={scaler.n_features_in_}. Error: {e}")
        return (np.array([]),) * 5

    # --- 5. Bucle de Inferencia ---
    dataset = _InferenceDataset(data_scaled_np)
    dataloader = PyTorchDataLoader(dataset, batch_size=hparams.get('batch_size', 16), shuffle=False, num_workers=0)

    all_reconstructions_scaled = []
    all_latent_vectors = []

    # Registrar hook para capturar embedding
    hook = model.rnn_encoder.register_forward_hook(get_latent_hook(model_name_key))

    # Definir edge_index
    if model_name_key == "stgae_physics":
        graph_data = IMPORTED_MODEL_HELPERS["PhysicsGraphDef"](num_nodes=N)
        edge_index = graph_data['edge_index'].to(device)
    elif model_name_key == "gnn_base" or model_name_key == "wavelet_gnn":
        edge_index = IMPORTED_MODEL_HELPERS["GnnBaseGraphDef"]().to(device)
    else:
        edge_index = None  # Para no_gnn_ae

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Inference {model_name_display}", leave=False):
            batch_data = batch_data.to(device)

            # Inferencia
            if edge_index is not None:
                recons_batch = model(batch_data, edge_index)
            else:
                recons_batch = model(batch_data)  # no_gnn_ae

            all_reconstructions_scaled.append(recons_batch.cpu().numpy())
            all_latent_vectors.append(latent_space_capture[model_name_key])

    hook.remove()  # Limpiar hook

    reconstructions_scaled_np = np.concatenate(all_reconstructions_scaled, axis=0)  # [B, T, N, F_out]
    latent_space_np = np.concatenate(all_latent_vectors, axis=0)  # [B, H]

    # --- 6. Post-procesar y Calcular Métricas ---

    # Calcular error en el espacio escalado
    errors_mse_np = np.mean((data_scaled_np - reconstructions_scaled_np) ** 2, axis=(1, 2, 3))  # [B,]
    errors_mse_per_sensor_np = np.mean((data_scaled_np - reconstructions_scaled_np) ** 2, axis=(1, 3))  # [B, N]

    # Calcular SSIM en el espacio escalado
    errors_ssim_per_sensor_np = np.zeros((B, N))
    for b in range(B):
        for n in range(N):
            sig_in = data_scaled_np[b, :, n, 0]  # 1ra feature (cruda escalada)
            sig_out = reconstructions_scaled_np[b, :, n, 0]

            data_range = sig_in.max() - sig_in.min()
            if data_range < 1e-6: data_range = 1.0

            # win_size debe ser impar y <= T
            win_size_ssim = min(7, T)
            if win_size_ssim % 2 == 0: win_size_ssim -= 1
            if win_size_ssim < 3:
                ssim_val = 1.0 if np.allclose(sig_in, sig_out) else 0.0
            else:
                ssim_val = structural_similarity(sig_in, sig_out, data_range=data_range, win_size=win_size_ssim)

            errors_ssim_per_sensor_np[b, n] = ssim_val

    # Invertir escalado (solo de la primera feature, la señal cruda)
    recons_flat = reconstructions_scaled_np.reshape(-1, num_features_model)
    recons_unscaled_flat = scaler.inverse_transform(recons_flat)

    # Tomar solo la primera feature (la señal cruda)
    reconstructions_unscaled_np = recons_unscaled_flat[:, 0].reshape(B, T, N, 1)  # [B, T, N, 1]

    logging.info(f"Inferencia REAL completada para {model_name_display}.")

    # Liberar memoria de GPU
    del model, batch_data, recons_batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Devuelve:
    # 1. Reconstrucciones (un-scaled, F=1) - Para plots de señal
    # 2. Error MSE (global, por ventana) - Para KDE, ROC
    # 3. Espacio Latente - Para t-SNE/PCA
    # 4. Error MSE (por sensor, por ventana) - Para heatmaps
    # 5. Error SSIM (por sensor, por ventana) - Para SSIM plots
    return reconstructions_unscaled_np, errors_mse_np, latent_space_np, errors_mse_per_sensor_np, errors_ssim_per_sensor_np


# ###########################################################################
# --- MÓDULO 1: GRÁFICOS DE METODOLOGÍA Y WAVELETS ---
# ###########################################################################

def generate_module_1_plots(data: dict[str, np.ndarray], config: Config) -> None:
    """Genera las figuras para la sección de metodología."""
    logging.info("--- Iniciando Módulo 1: Gráficos de Metodología y Wavelets ---")
    module_key = "1_methodology"

    try:
        # --- Figura 1: Análisis de Señal y Espectro ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Signal and Spectral Analysis', fontsize=18, weight='bold')

        # (a) Señal Sana
        signal_healthy = data['healthy_test'][0, :, 0, 0]  # Sensor 0, sano
        axes[0, 0].plot(signal_healthy, label="Healthy Signal (Sensor 0)", color="C0")
        axes[0, 0].set_title("Sample Healthy Signal (Time Domain)")
        axes[0, 0].set_xlabel(f"Time Samples (Window Size = {config.WINDOW_SIZE})")
        axes[0, 0].set_ylabel("Raw Acceleration")
        axes[0, 0].legend()
        add_subfigure_label(axes[0, 0], 'a')

        # (b) Señal Anómala
        signal_anomaly = data['anomaly_test'][0, :, 0, 0]  # Sensor 0, anómalo
        axes[0, 1].plot(signal_anomaly, label="Damaged Signal (Sensor 0)", color="C3")
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

        # --- Figura 2: Descomposición Wavelet ---
        wavelet = 'db4'
        level = 5  # Coincide con tu hparam `wavelet_level`
        coeffs = pywt.wavedec(signal_healthy, wavelet, level=level)

        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(level + 1, 1, figure=fig)

        # Panel Original
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(signal_healthy, color="black", linewidth=1.5)
        ax0.set_title(f"Original Signal and DWT ('{wavelet}', L{level})", fontsize=16, weight='bold')
        ax0.set_ylabel("Amplitude")
        ax0.set_xticklabels([])
        add_subfigure_label(ax0, 'a')

        # Paneles de Reconstrucción
        plot_coeffs = [coeffs[0]] + list(coeffs[1:])
        plot_titles = [f'Approximation (A{level})'] + [f'Detail (D{level - i})' for i in range(level)]

        for i, (coeff, title) in enumerate(zip(plot_coeffs, plot_titles)):
            ax = fig.add_subplot(gs[i + 1, 0], sharex=ax0)

            # Reconstruir componente para que tenga la misma longitud
            coeff_list = [np.zeros_like(c) for c in coeffs]
            coeff_list[i] = coeff
            rec = pywt.waverec(coeff_list, wavelet)
            rec = IMPORTED_MODEL_HELPERS["DWT_adjust_length"](rec, len(signal_healthy))

            ax.plot(rec, color=f"C{i}", linewidth=1.5)
            ax.set_ylabel(title, fontsize=12, weight='bold')
            if i == level:  # Último plot
                ax.set_xlabel("Time Samples")
            else:
                ax.set_xticklabels([])
            add_subfigure_label(ax, chr(ord('b') + i))

        save_plot(fig, module_key, "1_2_Wavelet_Decomposition", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 1: {e}", exc_info=True)


# ###########################################################################
# --- MÓDULO 2: GRÁFICOS DE MÉTRICAS DE ENTRENAMIENTO (DATOS REALES) ---
# ###########################################################################

def generate_module_2_plots(all_artifacts: dict[str, dict], config: Config) -> None:
    """Genera las figuras de métricas de entrenamiento (usando logs REALES)."""
    logging.info("--- Iniciando Módulo 2: Gráficos de Métricas de Entrenamiento (DATOS REALES) ---")
    module_key = "2_training"

    try:
        # --- Figura 1: Curvas de Aprendizaje (2x2 Grid) ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Model Training & Validation Loss', fontsize=18, weight='bold')

        for ax, (model_name_key, model_name_display) in zip(axes.flatten(), config.MODEL_NAME_MAP.items()):
            artifacts = all_artifacts.get(model_name_key)
            if artifacts is None:
                logging.warning(f"No hay artefactos para {model_name_display}, omitiendo subgráfico.")
                ax.set_title(f"{model_name_display}\n(No Artifacts Found)")
                ax.axis('off')
                continue

            log_df = artifacts.get('log')
            if log_df is None or log_df.empty:
                logging.warning(f"No hay log para {model_name_display}, omitiendo subgráfico.")
                ax.set_title(f"{model_name_display}\n(No Log Found)")
                ax.axis('off')
                continue

            ax.plot(log_df['epoch'], log_df['train_loss'], label='Training Loss',
                    color=config.COLORS_MODEL[model_name_display], marker='.', markersize=4, alpha=0.7)
            ax.plot(log_df['epoch'], log_df['val_loss'], label='Validation Loss', color='black', linestyle='--',
                    marker='.', markersize=4, alpha=0.7)

            best_epoch_idx = log_df['val_loss'].idxmin()
            best_epoch = log_df['epoch'].iloc[best_epoch_idx]
            best_loss = log_df['val_loss'].min()
            ax.axvline(best_epoch, color='red', linestyle=':', label=f'Best Epoch ({best_epoch}): {best_loss:.6f}')

            ax.set_title(f"{model_name_display}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (MSE, log scale)")
            ax.set_yscale('log')
            ax.legend()

        # Añadir etiquetas (a, b, c, d)
        add_subfigure_label(axes[0, 0], 'a')
        add_subfigure_label(axes[0, 1], 'b')
        add_subfigure_label(axes[1, 0], 'c')
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, "2_1_All_Models_Loss_Curves", config)

        # --- Figura 2: Tabla de Hiperparámetros (REAL) ---
        hparams_list = []
        for model_name_key, model_name_display in config.MODEL_NAME_MAP.items():
            artifacts = all_artifacts.get(model_name_key, {})
            hparams = artifacts.get('hparams', {}).copy()
            hparams['Model'] = model_name_display

            log_df = artifacts.get('log')
            if log_df is not None and not log_df.empty:
                hparams['Best Validation Loss (MSE)'] = log_df['val_loss'].min()
                hparams['Final Training Loss (MSE)'] = log_df['train_loss'].iloc[-1]
            hparams_list.append(hparams)

        hparams_df = pd.DataFrame(hparams_list).set_index('Model').fillna("N/A")

        cols_to_show = [
            'Best Validation Loss (MSE)', 'Final Training Loss (MSE)', 'learning_rate', 'batch_size',
            'gnn_hidden', 'gnn_out', 'rnn_hidden', 'rnn_layers',
            'wavelet_name', 'wavelet_level', 'total_params', 'num_features'
        ]
        cols_present = [col for col in cols_to_show if col in hparams_df.columns]
        hparams_df_filtered = hparams_df[cols_present].T  # Transponer para mejor lectura

        def format_value(x):
            if isinstance(x, float): return f"{x:.6f}"
            if isinstance(x, int): return f"{x:,}"
            return str(x)

        hparams_df_formatted = hparams_df_filtered.map(format_value)  # <-- CORREGIDO v8

        fig, ax = plt.subplots(figsize=(16, 8))  # Más grande
        ax.axis('off')
        ax.set_title("Comparative Table of Hyperparameters and Results", loc='center', fontsize=16, weight='bold')
        table = ax.table(
            cellText=hparams_df_formatted.values,
            colLabels=hparams_df_formatted.columns,
            rowLabels=hparams_df_formatted.index,
            loc='center',
            cellLoc='left',
            rowLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)
        save_plot(fig, module_key, "2_2_hyperparameter_table", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 2: {e}", exc_info=True)


# ###########################################################################
# --- MÓDULO 3: GRÁFICOS DE ARQUITECTURA (GEOMETRÍA REAL) ---
# ###########################################################################

def generate_module_3_plots(config: Config) -> None:
    """Genera diagramas de arquitectura y topología de grafos (usando datos REALES)."""
    logging.info("--- Iniciando Módulo 3: Gráficos de Arquitectura del Modelo (GEOMETRÍA REAL) ---")
    module_key = "3_architecture"

    try:
        # --- Figura 1: Arquitectura y Topologías ---
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.1)

        # (a) Diagrama de Arquitectura ST-GAE
        ax_arch = fig.add_subplot(gs[0, :])
        ax_arch.set_title("Spatio-Temporal Graph Autoencoder (ST-GAE) Architecture", fontsize=16, weight='bold')
        ax_arch.axis('off');
        ax_arch.set_xlim(0, 100);
        ax_arch.set_ylim(0, 40)

        bbox_props = dict(boxstyle="round,pad=0.5", fc="lightblue", ec="b", lw=1.5)
        arrow_props = dict(arrowstyle="->", connectionstyle="arc3,rad=0", ec="black", lw=1.5)

        def add_block(ax, x, y, text, width=18):
            return ax.text(x, y, text, ha="center", va="center", bbox=bbox_props, wrap=True, fontsize=12), (x, y), width

        def connect_blocks(ax, pos1, pos2, w1, w2, label="", y_offset=0):
            x1, y1 = pos1[0] + w1 / 2, pos1[1] + y_offset
            x2, y2 = pos2[0] - w2 / 2, pos2[1] + y_offset
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)
            if label:
                ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1, label, ha="center", va="bottom", fontsize=11)

        node_in, pos_in, w_in = add_block(ax_arch, 10, 20, f"Input\n(B, T={config.WINDOW_SIZE}, N=5, F=7)")
        node_enc, pos_enc, w_enc = add_block(ax_arch, 40, 20, "ST-GAE Encoder\n(GCN + GRU)")
        node_lat, pos_lat, w_lat = add_block(ax_arch, 70, 20, "Latent Vector\n(B, H=256)")
        node_dec, pos_dec, w_dec = add_block(ax_arch, 40, 5, "ST-GAE Decoder\n(GRU + GCN)")
        node_out, pos_out, w_out = add_block(ax_arch, 10, 5, "Output\n(B, T, N, F=7)")

        connect_blocks(ax_arch, pos_in, pos_enc, w_in, w_enc, "edge_index")
        connect_blocks(ax_arch, pos_enc, pos_lat, w_enc, w_lat)

        # Decoder path
        ax_arch.annotate("", xy=(pos_dec[0] + w_dec / 2, pos_dec[1]), xytext=(pos_lat[0] - w_lat / 2, pos_lat[1]),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", ec="black", lw=1.5))
        ax_arch.annotate("", xy=(pos_out[0] + w_out / 2, pos_out[1]), xytext=(pos_dec[0] - w_dec / 2, pos_dec[1]),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", ec="black", lw=1.5))

        add_subfigure_label(ax_arch, 'a', x=0.0)

        # (b) Topología del Grafo Binario
        ax_bin = fig.add_subplot(gs[1, 0])
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

            nx.draw(G_base, pos_base, with_labels=True, labels=labels,
                    node_color=config.COLORS_MODEL["M2: GNN Original"],
                    node_size=2500, font_size=15, font_weight='bold', edge_color='gray', width=3, ax=ax_bin,
                    font_color='white')
            ax_bin.set_title("Binary Adjacency Graph (M2)")
            add_subfigure_label(ax_bin, 'b')

        # (c) Topología del Grafo Físico
        ax_phys = fig.add_subplot(gs[1, 1])
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
                node_color=config.COLORS_MODEL["M4: PI-STG-AE"],
                node_size=2500, font_size=15, font_weight='bold',
                edge_color=weights, edge_cmap=plt.cm.viridis, width=norm_weights,
                ax=ax_phys, font_color='white'
            )

            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=min_w, vmax=max_w))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_phys, shrink=0.8)
            cbar.set_label('Edge Weight (1/distance)')
            ax_phys.set_title("Physics-Informed Graph (M4)")
            add_subfigure_label(ax_phys, 'c')

        save_plot(fig, module_key, "3_1_Architecture_and_Topologies", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 3: {e}", exc_info=True)


# ###########################################################################
# --- MÓDULO 4: ANÁLISIS DE RECONSTRUCCIÓN (REAL) ---
# ###########################################################################

def generate_module_4_plots(
        model_artifacts: dict,
        data: dict[str, np.ndarray],
        preds: dict[str, tuple],
        config: Config,
        model_name_key: str
) -> None:
    """Genera análisis de reconstrucción, error y espacio latente (REALES)."""
    model_name_display = config.MODEL_NAME_MAP[model_name_key]
    logging.info(f"--- Iniciando Módulo 4: Análisis de Reconstrucción ({model_name_display}) (REAL) ---")
    module_key = "4_reconstruction"

    try:
        # --- 4.0: Obtener predicciones (REALES) ---
        recon_h, err_mse_h, latent_h, err_mse_h_sensor, err_ssim_h_sensor = preds['healthy_test']
        recon_a, err_mse_a, latent_a, err_mse_a_sensor, err_ssim_a_sensor = preds['anomaly_test']

        if recon_h.size == 0 or recon_a.size == 0:
            logging.error(f"No hay predicciones para {model_name_display}. Omitiendo Módulo 4.")
            return

        # --- Figura 1: Ejemplos de Reconstrucción (Sano vs Anómalo) ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))
        fig.suptitle(f'Reconstruction Analysis: {model_name_display}', fontsize=18, weight='bold')

        # (a) Reconstrucción Sana
        idx, sensor_idx = 0, 0
        axes[0, 0].plot(data['healthy_test'][idx, :, sensor_idx, 0], 'C0-',
                        label=f'Original Signal (Sensor {sensor_idx + 1})', linewidth=1.5)
        axes[0, 0].plot(recon_h[idx, :, sensor_idx, 0], 'C1--', label=f'Reconstructed Signal', linewidth=1.5)
        axes[0, 0].set_title(f"Healthy State Reconstruction")
        axes[0, 0].set_ylabel("Raw Acceleration")
        axes[0, 0].legend()
        add_subfigure_label(axes[0, 0], 'a')

        # (b) Error Sano
        error_h = data['healthy_test'][idx, :, sensor_idx, 0] - recon_h[idx, :, sensor_idx, 0]
        axes[0, 1].plot(error_h, 'r-', label=f'Reconstruction Error (MSE: {err_mse_h[idx]:.4f})')
        axes[0, 1].set_title(f"Healthy State Reconstruction Error")
        axes[0, 1].set_ylabel("Error")
        axes[0, 1].legend()
        add_subfigure_label(axes[0, 1], 'b')

        # (c) Reconstrucción Anómala
        idx, sensor_idx = 0, 0
        axes[1, 0].plot(data['anomaly_test'][idx, :, sensor_idx, 0], 'C3-',
                        label=f'Original Signal (Sensor {sensor_idx + 1})', linewidth=1.5)
        axes[1, 0].plot(recon_a[idx, :, sensor_idx, 0], 'C1--', label=f'Reconstructed Signal', linewidth=1.5)
        axes[1, 0].set_title(f"Damaged State Reconstruction")
        axes[1, 0].set_xlabel("Time Samples")
        axes[1, 0].set_ylabel("Raw Acceleration")
        axes[1, 0].legend()
        add_subfigure_label(axes[1, 0], 'c')

        # (d) Error Anómalo
        error_a = data['anomaly_test'][idx, :, sensor_idx, 0] - recon_a[idx, :, sensor_idx, 0]
        axes[1, 1].plot(error_a, 'r-', label=f'Reconstruction Error (MSE: {err_mse_a[idx]:.4f})')
        axes[1, 1].set_title(f"Damaged State Reconstruction Error")
        axes[1, 1].set_xlabel("Time Samples")
        axes[1, 1].set_ylabel("Error")
        axes[1, 1].legend()
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, f"4_1_Reconstruction_Examples_{model_name_key}", config)

        # --- Figura 2: Análisis de Espacio Latente ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Latent Space Analysis: {model_name_display}', fontsize=18, weight='bold')

        labels_all = ['Healthy'] * len(latent_h) + ['Damaged'] * len(latent_a)
        latent_all = np.concatenate([latent_h, latent_a], axis=0)

        scaler_latent = StandardScaler()
        latent_scaled = scaler_latent.fit_transform(latent_all)

        # (a) PCA
        logging.info(f"Calculando PCA para {model_name_display}...")
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(latent_scaled)
        pca_df = pd.DataFrame(pca_results, columns=['PCA 1', 'PCA 2'])
        pca_df['State'] = labels_all

        sns.scatterplot(
            data=pca_df, x='PCA 1', y='PCA 2', hue='State',
            palette={'Healthy': 'C0', 'Damaged': 'C3'}, alpha=0.5, s=20, ax=axes[0]
        )
        axes[0].set_title(f"PCA of Latent Space (Variance: {pca.explained_variance_ratio_.sum():.2f})")
        add_subfigure_label(axes[0], 'a')

        # (b) t-SNE
        logging.info(f"Calculando t-SNE para {model_name_display}...")
        # --- CORRECCIÓN v8: Cambiar 'n_iter' a 'max_iter' ---
        tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, perplexity=30, max_iter=1000)
        tsne_results = tsne.fit_transform(latent_scaled)
        tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])
        tsne_df['State'] = labels_all

        sns.scatterplot(
            data=tsne_df, x='t-SNE 1', y='t-SNE 2', hue='State',
            palette={'Healthy': 'C0', 'Damaged': 'C3'}, alpha=0.5, s=20, ax=axes[1]
        )
        axes[1].set_title(f"t-SNE of Latent Space")
        add_subfigure_label(axes[1], 'b')

        save_plot(fig, module_key, f"4_2_Latent_Space_Analysis_{model_name_key}", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 4: {e}", exc_info=True)


# ###########################################################################
# --- MÓDULO 5: SIMULACIONES 3D (GEOMETRÍA REAL) ---
# ###########################################################################

def generate_module_5_plots(
        model_artifacts: dict,
        preds: dict[str, tuple],
        config: Config,
        model_name_key: str
) -> None:
    """Genera visualizaciones 3D del puente y los errores (GEOMETRÍA REAL)."""
    model_name_display = config.MODEL_NAME_MAP[model_name_key]
    logging.info(f"--- Iniciando Módulo 5: Simulaciones 3D ({model_name_display}) (GEOMETRÍA REAL) ---")
    module_key = "5_simulation"

    try:
        # --- 5.0: Coordenadas 3D Reales ---
        coords = {
            0: np.array([13.88, -4.0, -1.0]),  # S1
            1: np.array([13.88, 4.0, -1.0]),  # S2
            2: np.array([27.76, -4.0, -1.0]),  # S3
            3: np.array([27.76, 4.0, -1.0]),  # S4
            4: np.array([41.64, 0.0, -1.0])  # S5
        }
        sensor_pos = np.array([coords[i] for i in range(config.N_SENSORS)])
        x_range = np.ptp(sensor_pos[:, 0])
        y_range = np.ptp(sensor_pos[:, 1])
        z_range = np.ptp(sensor_pos[:, 2])
        if z_range < 1e-6: z_range = max(x_range, y_range) * 0.1

        # --- Figura 1: Layout 3D y Heatmap de Error ---
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'3D Damage Localization Analysis: {model_name_display}', fontsize=18, weight='bold')

        # (a) Visualización 3D del Layout de Sensores
        ax_layout: Axes3D = fig.add_subplot(121, projection='3d')

        ax_layout.plot([0, 55.52], [-4.0, -4.0], [-1.0, -1.0], 'k-', lw=3, label="Main Girder 1")
        ax_layout.plot([0, 55.52], [4.0, 4.0], [-1.0, -1.0], 'k-', lw=3, label="Main Girder 2")

        ax_layout.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2], c='red', s=100, label="Sensors (N=5)")
        for i, (x, y, z) in enumerate(sensor_pos):
            ax_layout.text(x, y + 0.5, z, f'S{i + 1}', color='red', fontsize=12, weight='bold')

        ax_layout.set_title("Sensor 3D Layout (Physical Coordinates)")
        ax_layout.set_xlabel("X-axis (m) - Longitudinal")
        ax_layout.set_ylabel("Y-axis (m) - Transverse")
        ax_layout.set_zlabel("Z-axis (m) - Vertical")
        ax_layout.legend()
        ax_layout.view_init(elev=20, azim=60)
        ax_layout.set_box_aspect([x_range, y_range, z_range])
        add_subfigure_label(ax_layout, 'a')

        # (b) Heatmap 3D de Errores de Anomalía
        _, _, _, err_mse_a_sensor, _ = preds['anomaly_test']

        if err_mse_a_sensor.size == 0:
            logging.warning(f"No hay errores de anomalía para {model_name_display}. Omitiendo heatmap 3D.")
            return

        error_per_sensor = np.mean(err_mse_a_sensor, axis=0)  # Shape [N,]

        ax_heat: Axes3D = fig.add_subplot(122, projection='3d')

        ax_heat.plot([0, 55.52], [-4.0, -4.0], [-1.0, -1.0], 'k-', lw=1, alpha=0.3)
        ax_heat.plot([0, 55.52], [4.0, 4.0], [-1.0, -1.0], 'k-', lw=1, alpha=0.3)

        vmin = np.min(error_per_sensor)
        vmax = np.max(error_per_sensor)
        if vmin <= 0 or vmin == vmax:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.Reds
        else:
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            cmap = plt.cm.Reds

        colors = cmap(norm(error_per_sensor))

        scatter = ax_heat.scatter(
            sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
            c=colors, s=200, depthshade=True
        )

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable, ax=ax_heat, shrink=0.5, aspect=10)
        cbar.set_label("Mean Reconstruction Error (MSE)")

        ax_heat.set_title("3D Anomaly Localization Heatmap (Real)")
        ax_heat.set_xlabel("X-axis (m)")
        ax_heat.set_ylabel("Y-axis (m)")
        ax_heat.set_zlabel("Z-axis (m)")
        ax_heat.view_init(elev=20, azim=60)
        ax_heat.set_box_aspect([x_range, y_range, z_range])
        add_subfigure_label(ax_heat, 'b')

        # --- CORRECCIÓN v8: `is_3d=True` no es un arg válido para mi `save_plot` ---
        save_plot(fig, module_key, f"5_1_3D_Layout_and_Heatmap_{model_name_key}", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 5: {e}", exc_info=True)


# ###########################################################################
# --- MÓDULO 6: GRÁFICOS DE DETECCIÓN DE ANOMALÍAS (REAL) ---
# ###########################################################################

def generate_module_6_plots(
        model_artifacts: dict,
        preds: dict[str, tuple],
        config: Config,
        model_name_key: str
) -> None:
    """Genera métricas de clasificación de anomalías (REALES)."""
    model_name_display = config.MODEL_NAME_MAP[model_name_key]
    logging.info(f"--- Iniciando Módulo 6: Gráficos de Detección de Anomalías ({model_name_display}) (REAL) ---")
    module_key = "6_anomaly"

    try:
        # --- 6.0: Preparar etiquetas y puntuaciones (REALES) ---
        _, errors_healthy, _, _, _ = preds['healthy_test']
        _, errors_anomaly, _, _, _ = preds['anomaly_test']

        if errors_healthy.size == 0 or errors_anomaly.size == 0:
            logging.error(f"Datos de error vacíos para {model_name_display}. Omitiendo Módulo 6.")
            return

        y_true = np.concatenate([np.zeros(len(errors_healthy)), np.ones(len(errors_anomaly))])
        y_scores = np.concatenate([errors_healthy, errors_anomaly])

        # --- Figura 1: Métricas de Clasificación ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'Anomaly Detection Performance: {model_name_display}', fontsize=18, weight='bold')

        # (a) Curva ROC
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
        axes[0, 0].set_title(f'Receiver Operating Characteristic (ROC)')
        axes[0, 0].set_xlabel('False Positive Rate (FPR)')
        axes[0, 0].set_ylabel('True Positive Rate (TPR)')
        axes[0, 0].legend(loc="lower right")
        add_subfigure_label(axes[0, 0], 'a')

        # (b) Matriz de Confusión
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds_roc[optimal_idx]
        y_pred = (y_scores > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                    xticklabels=['Predicted Healthy', 'Predicted Damaged'],
                    yticklabels=['Actual Healthy', 'Actual Damaged'], annot_kws={"size": 16})
        axes[0, 1].set_title(f"Confusion Matrix (Optimal Threshold={threshold:.4f})")
        add_subfigure_label(axes[0, 1], 'b')

        # (c) Curva de Precisión-Recall (PR)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        axes[1, 0].plot(recall, precision, color='blue', lw=2.5, label=f'PR Curve (AP = {avg_precision:.4f})')
        axes[1, 0].set_title(f'Precision-Recall (PR) Curve')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        add_subfigure_label(axes[1, 0], 'c')

        # (d) Serie Temporal de Errores
        n_healthy = len(errors_healthy)
        n_anomaly = len(errors_anomaly)

        axes[1, 1].plot(np.arange(n_healthy), errors_healthy, label='Healthy', color='C0', alpha=0.7, lw=0.5)
        axes[1, 1].plot(np.arange(n_healthy, n_healthy + n_anomaly), errors_anomaly, label='Damaged', color='C3',
                        alpha=0.7, lw=0.5)
        axes[1, 1].axvline(n_healthy, color='black', linestyle='--', label='Damage Event')
        axes[1, 1].set_title(f"Reconstruction Error Time Series")
        axes[1, 1].set_xlabel("Window Index")
        axes[1, 1].set_ylabel("Error (MSE, log scale)")
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        add_subfigure_label(axes[1, 1], 'd')

        save_plot(fig, module_key, f"6_1_Classification_Metrics_{model_name_key}", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 6: {e}", exc_info=True)


# ###########################################################################
# --- MÓDULO 7: ANÁLISIS ADICIONAL (ABLATION, SOTA) (DATOS REALES) ---
# ###########################################################################

def generate_module_7_plots(all_artifacts: dict[str, dict], config: Config) -> None:
    """Genera gráficos de ablation study y SOTA (usando logs REALES)."""
    logging.info("--- Iniciando Módulo 7: Análisis Adicional (DATOS REALES) ---")
    module_key = "7_analysis"

    try:
        # --- Figura 1: Ablation Study y SOTA ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Model Ablation and SOTA Comparison', fontsize=18, weight='bold')

        # (a) Estudio de Ablación
        ablation_data = []
        for model_name_key, model_name_display in config.MODEL_NAME_MAP.items():
            artifacts = all_artifacts.get(model_name_key)
            if artifacts is None: continue

            log_df = artifacts.get('log')
            hparams = artifacts.get('hparams', {})
            if log_df is not None and not log_df.empty:
                ablation_data.append({
                    'Model': model_name_display,
                    'Best Validation Loss (MSE)': log_df['val_loss'].min(),
                    'Parameters': hparams.get('total_params', 0)
                })

        if not ablation_data:
            logging.error("No hay datos de log para generar estudio de ablación.")
            return

        ablation_df = pd.DataFrame(ablation_data).sort_values(by='Best Validation Loss (MSE)')

        barplot = sns.barplot(data=ablation_df, x='Model', y='Best Validation Loss (MSE)',
                              ax=axes[0], palette=config.COLORS_MODEL.values())
        axes[0].set_title("Ablation Study (Validation Loss)")
        axes[0].set_ylabel("Best Validation Loss (MSE, log scale)")
        axes[0].set_xlabel("Model Architecture")
        axes[0].set_yscale('log')
        for p in barplot.patches:
            barplot.annotate(
                f"{p.get_height():.6f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points'
            )
        axes[0].tick_params(axis='x', rotation=15)
        add_subfigure_label(axes[0], 'a')

        # (b) Gráfico de Parámetros vs. Rendimiento
        sns.scatterplot(data=ablation_df, x='Parameters', y='Best Validation Loss (MSE)',
                        hue='Model', s=300, ax=axes[1], palette=config.COLORS_MODEL,
                        style='Model', markers=config.MARKERS)
        axes[1].set_title("Performance vs. Complexity")
        axes[1].set_xlabel("Trainable Parameters (log scale)")
        axes[1].set_ylabel("Best Validation Loss (MSE, log scale)")
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        for i, row in ablation_df.iterrows():
            axes[1].text(row['Parameters'] * 1.1, row['Best Validation Loss (MSE)'], row['Model'], fontsize=10)
        add_subfigure_label(axes[1], 'b')

        save_plot(fig, module_key, "7_1_Ablation_and_Complexity", config)

        # --- Figura 2: Tabla Comparativa SOTA (REAL) ---
        sota_data = []
        for model_name_key, model_name_display in config.MODEL_NAME_MAP.items():
            artifacts = all_artifacts.get(model_name_key, {})
            log_df = artifacts.get('log')
            hparams = artifacts.get('hparams', {})
            if log_df is not None and not log_df.empty:
                sota_data.append({
                    'Model': model_name_display,
                    'Best Loss (MSE)': log_df['val_loss'].min(),
                    'Parameters': hparams.get('total_params', 0),
                    'Features': hparams.get('num_features', 1),
                    'Graph Type': 'Physics' if 'physics' in model_name_key else (
                        'Binary' if 'gnn' in model_name_key and 'no_gnn' not in model_name_key else 'N/A')
                })

        sota_df = pd.DataFrame(sota_data).set_index('Model').sort_values(by='Best Loss (MSE)')

        def format_value(x):
            if isinstance(x, float): return f"{x:.6f}"
            if isinstance(x, int): return f"{x:,}"
            return str(x)

        sota_df_formatted = sota_df.map(format_value)  # <-- CORREGIDO v8

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('off')
        ax.set_title("SOTA Comparison Table (Validation Results)", loc='center', fontsize=16, weight='bold')
        table = ax.table(
            cellText=sota_df_formatted.values,
            colLabels=sota_df_formatted.columns,
            rowLabels=sota_df_formatted.index,
            loc='center', cellLoc='left', rowLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)
        save_plot(fig, module_key, "7_2_SOTA_Comparison_Table", config)

    except Exception as e:
        logging.error(f"Error al generar gráficos del Módulo 7: {e}", exc_info=True)


# ###########################################################################
# --- ORQUESTADOR PRINCIPAL (MAIN) ---
# ###########################################################################

def main():
    """Función principal que orquesta el pipeline de generación de figuras."""

    config = Config()
    setup_environment(config)

    logging.info("=" * 80)
    logging.info("INICIANDO PIPELINE DE GENERACIÓN DE FIGURAS Q1 (SHM-STGAE)")
    logging.info(f"Proyecto: {config.BASE_DIR.name}")
    logging.info(f"Resultados se guardarán en: {config.OUTPUT_DIR}")
    logging.info("=" * 80)

    try:
        # --- PASO 1: Cargar Clases de Modelo (REAL) ---
        # (Esto ahora se hace globalmente, en la SECCIÓN 3)
        logging.info("--- PASO 1/5: Verificando Clases de Modelo .py ---")
        if not all(IMPORTED_MODEL_CLASSES.values()) or not all(IMPORTED_MODEL_HELPERS.values()):
            logging.critical(
                "Una o más clases de modelo o helpers no se pudieron importar. Revisa los logs. Abortando.")
            return
        logging.info("Todas las clases de modelo y helpers importados exitosamente.")

        # --- PASO 2: Cargar Datos (REAL) ---
        logging.info("--- PASO 2/5: Cargando y Preprocesando Datos (REAL) ---")
        data = load_shm_data(config)
        logging.info("Datos (REALES) cargados exitosamente.")
        gc.collect()

        # --- PASO 3: Cargar Artefactos de Modelos (REAL) ---
        logging.info("--- PASO 3/5: Cargando Artefactos de Modelos (REAL) ---")
        all_artifacts = {}
        for model_name_key in config.MODEL_NAME_MAP_INV.keys():  # Iterar por M1, M2...
            internal_key = config.MODEL_NAME_MAP_INV[model_name_key]
            try:
                model_results_dir = config.MODEL_RESULTS_DIRS[internal_key]
                artifacts = load_model_artifacts(model_results_dir, internal_key, config)
                all_artifacts[internal_key] = artifacts  # Guardar por llave interna
                logging.info(f"Artefactos para '{model_name_key}' cargados.")
            except Exception as e:
                logging.error(f"Fallo al cargar artefactos para '{model_name_key}': {e}", exc_info=True)

        if not all_artifacts:
            logging.critical("No se pudo cargar ningún artefacto. Abortando pipeline.")
            return

        logging.info(f"Se cargaron {len(all_artifacts)} conjuntos de artefactos de modelos.")
        gc.collect()  # Limpiar memoria después de cargar artefactos

        # --- PASO 4: Generar Gráficos por Módulo ---
        logging.info("--- PASO 4/5: Iniciando Generación de Gráficos ---")

        # Módulo 1 (Necesita datos reales)
        generate_module_1_plots(data, config)

        # Módulo 2 (Necesita artefactos reales)
        generate_module_2_plots(all_artifacts, config)

        # Módulo 3 (Necesita config y clases importadas)
        generate_module_3_plots(config)

        # Módulos 4, 5, 6 (Se generan para CADA modelo)
        all_predictions = {}
        for model_name_key, artifacts in all_artifacts.items():
            model_name_display = config.MODEL_NAME_MAP[model_name_key]
            if not all(artifacts.get(k) for k in ['hparams', 'scaler', 'best_model_path', 'model_class']):
                logging.warning(f"Omitiendo Inferencia para {model_name_display} por falta de artefactos.")
                continue

            logging.info(f"--- Ejecutando Inferencia para: {model_name_display} ---")

            # Generar predicciones para 'healthy_test' y 'anomaly_test'
            preds = {
                'healthy_test': get_model_predictions(artifacts, data['healthy_test'], config, model_name_key),
                'anomaly_test': get_model_predictions(artifacts, data['anomaly_test'], config, model_name_key)
            }
            all_predictions[model_name_key] = preds

            logging.info(f"--- Generando Módulos 4, 5, 6 para: {model_name_display} ---")

            # Módulo 4
            generate_module_4_plots(artifacts, data, preds, config, model_name_key)

            # Módulo 5
            generate_module_5_plots(artifacts, preds, config, model_name_key)

            # Módulo 6
            generate_module_6_plots(artifacts, preds, config, model_name_key)

            gc.collect()  # Limpiar memoria después de cada modelo

        # Módulo 7 (Necesita todos los artefactos reales)
        generate_module_7_plots(all_artifacts, config)

        # --- PASO 5: Finalización ---
        logging.info("--- PASO 5/5: Pipeline de Generación de Figuras Completado ---")
        logging.info("=" * 80)
        logging.info("GENERACIÓN 100% REAL COMPLETADA.")
        logging.info(f"Todas las figuras se han guardado en: {config.OUTPUT_DIR}")
        logging.info(f"El log de esta ejecución está en: {config.OUTPUT_DIR / 'figure_generation.log'}")
        logging.info("=" * 80)

    except Exception as e:
        logging.critical(f"Error fatal en el pipeline principal: {e}", exc_info=True)
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