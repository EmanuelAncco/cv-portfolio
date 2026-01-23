import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import pandas as pd
from datetime import timedelta


# --- Configuración del Logging ---
def setup_logging(output_dir):
    """Configura el logging para guardar en archivo y mostrar en consola."""
    log_file = os.path.join(output_dir, 'comparison_log.txt')

    # Asegurarse de que los handlers no se acumulen si se re-ejecuta
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)

    # Handler para el archivo
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Handler para la consola
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger


# --- Funciones de Parseo de Datos ---

def load_json_data(filepath):
    """Carga de forma segura un archivo JSON."""
    logger = logging.getLogger()
    if not os.path.exists(filepath):
        logger.error(f"Archivo JSON no encontrado: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Datos cargados exitosamente desde: {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error al cargar o parsear JSON {filepath}: {e}")
        return None


def parse_gnn_base_log(log_path):
    """
    Parsea el archivo training_log.txt para extraer el historial de pérdidas
    y la mejor pérdida de validación.
    """
    logger = logging.getLogger()
    if not os.path.exists(log_path):
        logger.error(f"Archivo de log no encontrado: {log_path}")
        return None, float('inf')

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    # Patrón para Epoch Loss
    epoch_pattern = re.compile(r"Epoch \d+/\d+ -> Train Loss: ([\d.eE+-]+), Val Loss: ([\d.eE+-]+)")
    # Patrón para Mejor Loss (para asegurar que tomamos el valor correcto)
    best_loss_pattern = re.compile(r"Nuevo mejor modelo guardado con Val Loss: ([\d.eE+-]+)")

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    history['train_loss'].append(float(epoch_match.group(1)))
                    history['val_loss'].append(float(epoch_match.group(2)))

                best_match = best_loss_pattern.search(line)
                if best_match:
                    best_val_loss = float(best_match.group(1))

        if not history['train_loss']:
            logger.warning(f"No se encontraron datos de épocas en {log_path}")
            return None, best_val_loss  # Retornar best_val_loss aunque el historial falle

        logger.info(f"Historial de GNN-Base parseado. {len(history['train_loss'])} épocas encontradas.")
        logger.info(f"Mejor Val Loss (GNN-Base) extraída del log: {best_val_loss:.6f}")
        return history, best_val_loss

    except Exception as e:
        logger.error(f"Error al parsear el archivo de log {log_path}: {e}")
        return None, float('inf')


def parse_duration_to_minutes(duration_str):
    """Convierte un string de duración 'H:MM:SS.ms' a minutos totales."""
    if not duration_str:
        return 0
    try:
        # Dividir en tiempo y milisegundos
        parts = duration_str.split('.')
        time_parts = parts[0].split(':')

        if len(time_parts) == 3:  # H:MM:SS
            h, m, s = map(int, time_parts)
            total_seconds = timedelta(hours=h, minutes=m, seconds=s).total_seconds()
        elif len(time_parts) == 2:  # MM:SS
            m, s = map(int, time_parts)
            total_seconds = timedelta(minutes=m, seconds=s).total_seconds()
        else:
            return 0  # Formato no reconocido

        return total_seconds / 60.0
    except Exception as e:
        logging.getLogger().error(f"Error al parsear duración '{duration_str}': {e}")
        return 0


# --- Funciones de Ploteo ---

def plot_combined_loss_curves(hist_dict, output_dir):
    """
    Genera y guarda gráficos de pérdida de entrenamiento y validación superpuestos
    para todos los modelos, usando una escala logarítmica.
    """
    logger = logging.getLogger()
    logger.info("Generando gráficos de curvas de pérdida combinadas...")

    # Paleta de colores distintiva
    palette = sns.color_palette("Set1", n_colors=len(hist_dict))

    fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=False)
    fig.suptitle('Comparación de Curvas de Aprendizaje del Modelo', fontsize=16, y=1.02)

    # --- Gráfico de Pérdida de Entrenamiento ---
    ax = axes[0]
    for i, (model_name, history) in enumerate(hist_dict.items()):
        if history and history.get('train_loss'):
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], label=model_name, color=palette[i], linewidth=2)

    ax.set_title('Pérdida de Entrenamiento (Train Loss)')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida (MSE) - Escala Logarítmica')
    ax.set_yscale('log')  # Escala logarítmica es crucial
    ax.legend()
    ax.grid(True, linestyle=':')

    # --- Gráfico de Pérdida de Validación ---
    ax = axes[1]
    for i, (model_name, history) in enumerate(hist_dict.items()):
        if history and history.get('val_loss'):
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], label=model_name, color=palette[i], linewidth=2, linestyle='--')

            # Marcar el punto de mejor pérdida
            best_epoch = np.argmin(history['val_loss'])
            best_loss = history['val_loss'][best_epoch]
            ax.plot(best_epoch + 1, best_loss, 'x', color=palette[i], markersize=10, markeredgewidth=2,
                    label=f'{model_name} (Mejor: {best_loss:.6f})')

    ax.set_title('Pérdida de Validación (Validation Loss)')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida (MSE) - Escala Logarítmica')
    ax.set_yscale('log')  # Escala logarítmica es crucial
    ax.legend()
    ax.grid(True, linestyle=':')

    plt.tight_layout()

    # Guardar gráfico
    try:
        filename = os.path.join(output_dir, "combined_loss_curves.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de pérdidas guardado en: {filename}")
    except Exception as e:
        logger.error(f"Error al guardar gráfico de pérdidas: {e}")
    plt.close(fig)


def plot_performance_barcharts(stats_df, output_dir):
    """
    Genera y guarda gráficos de barras comparando métricas clave:
    Mejor Val Loss, Parámetros Totales, y Duración de Entrenamiento.
    """
    logger = logging.getLogger()
    logger.info("Generando gráficos de barras de rendimiento...")

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Comparación de Métricas Clave de Modelos', fontsize=16, y=1.03)

    palette = sns.color_palette("Set1", n_colors=len(stats_df))

    # --- 1. Mejor Pérdida de Validación (Log Scale) ---
    ax = axes[0]
    sns.barplot(x=stats_df.index, y='Best Val Loss', data=stats_df, ax=ax, palette=palette)
    ax.set_title('Mejor Pérdida de Validación (Menor es Mejor)')
    ax.set_ylabel('Pérdida (MSE) - Escala Logarítmica')
    ax.set_yscale('log')  # Escala logarítmica
    ax.grid(True, linestyle=':', axis='y')
    # Añadir etiquetas de valor
    for i, (model, row) in enumerate(stats_df.iterrows()):
        ax.text(i, row['Best Val Loss'], f"{row['Best Val Loss']:.6f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # --- 2. Parámetros Totales (Escala Log) ---
    ax = axes[1]
    # Filtrar modelos sin datos de parámetros para evitar errores
    plot_data_params = stats_df[stats_df['Total Params'] > 0]
    if not plot_data_params.empty:
        sns.barplot(x=plot_data_params.index, y='Total Params', data=plot_data_params, ax=ax,
                    palette=sns.color_palette("Set1", n_colors=len(plot_data_params)))
        ax.set_title('Complejidad del Modelo (Menor es Más Ligero)')
        ax.set_ylabel('Parámetros Totales - Escala Logarítmica')
        ax.set_yscale('log')  # Escala logarítmica
        ax.grid(True, linestyle=':', axis='y')
        # Añadir etiquetas de valor
        for i, (model, row) in enumerate(plot_data_params.iterrows()):
            ax.text(i, row['Total Params'], f"{int(row['Total Params']):,}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.set_title('Complejidad del Modelo (Datos no disponibles)')

    # --- 3. Duración de Entrenamiento (Minutos) ---
    ax = axes[2]
    sns.barplot(x=stats_df.index, y='Training Time (min)', data=stats_df, ax=ax, palette=palette)
    ax.set_title('Costo Computacional (Menor es Más Rápido)')
    ax.set_ylabel('Duración de Entrenamiento (Minutos)')
    ax.grid(True, linestyle=':', axis='y')
    # Añadir etiquetas de valor
    for i, (model, row) in enumerate(stats_df.iterrows()):
        ax.text(i, row['Training Time (min)'], f"{row['Training Time (min)']:.1f} min",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Guardar gráfico
    try:
        filename = os.path.join(output_dir, "model_performance_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de barras de rendimiento guardado en: {filename}")
    except Exception as e:
        logger.error(f"Error al guardar gráfico de barras: {e}")
    plt.close(fig)


# --- Función Principal ---

def main():
    # --- Define las rutas a tus archivos de logs e hiperparámetros ---
    # Asegúrate de que estas rutas sean correctas
    base_dir = "D:\\Rutas\\Donde\\Estan\\Tus\\Archivos"  # <--- CAMBIA ESTO

    # Asignar rutas a archivos específicos
    log_paths = {
        'gnn_base_log': os.path.join(base_dir, 'training_log.txt'),
        'wavelet_gnn_log': os.path.join(base_dir, 'loss_history_wavelet_gnn.json'),
        'no_gnn_log': os.path.join(base_dir, 'loss_history_no_gnn.json'),
        'gnn_base_hyper': os.path.join(base_dir, 'hyperparameters.json'),
        'wavelet_gnn_hyper': os.path.join(base_dir, 'hyperparameters_wavelet_gnn.json'),
        'no_gnn_hyper': os.path.join(base_dir, 'hyperparameters_no_gnn.json')
    }

    # Directorio de salida para los gráficos
    output_plot_dir = os.path.join(base_dir, "model_comparison_plots")
    os.makedirs(output_plot_dir, exist_ok=True)

    # Configurar logging
    logger = setup_logging(output_plot_dir)

    # --- 1. Cargar Historiales de Pérdida ---
    logger.info("--- Cargando Historiales de Entrenamiento ---")

    # Modelo 1: GNN-Base (desde .txt)
    hist_gnn, best_val_gnn = parse_gnn_base_log(log_paths['gnn_base_log'])

    # Modelo 2: Wavelet-GNN (desde .json)
    hist_wavelet = load_json_data(log_paths['wavelet_gnn_log'])

    # Modelo 3: No-GNN (desde .json)
    hist_no_gnn = load_json_data(log_paths['no_gnn_log'])

    # Diccionario de historiales para ploteo
    hist_dict = {
        'GNN (Base)': hist_gnn,
        'Wavelet-GNN': hist_wavelet,
        'No-GNN AE': hist_no_gnn
    }

    # --- 2. Cargar Hiperparámetros y Métricas Finales ---
    logger.info("--- Extrayendo Métricas Finales de Hiperparámetros ---")

    hyper_wavelet = load_json_data(log_paths['wavelet_gnn_hyper'])
    hyper_no_gnn = load_json_data(log_paths['no_gnn_hyper'])
    # hyper_gnn_base no tiene todas las métricas, las extrajimos del log

    # Recopilar estadísticas para el gráfico de barras
    stats_data = {
        'GNN (Base)': {
            'Best Val Loss': best_val_gnn,
            'Total Params': 0,  # No está en los archivos, poner 0 o 'np.nan'
            'Training Time (min)': parse_duration_to_minutes("5:48:23")  # Hardcodeado del log 07:56:19 - 02:07:56
        },
        'Wavelet-GNN': {
            'Best Val Loss': hyper_wavelet.get('best_val_loss', float('inf')) if hyper_wavelet else float('inf'),
            'Total Params': hyper_wavelet.get('total_params', 0) if hyper_wavelet else 0,
            'Training Time (min)': parse_duration_to_minutes(
                hyper_wavelet.get('training_duration', '')) if hyper_wavelet else 0
        },
        'No-GNN AE': {
            'Best Val Loss': hyper_no_gnn.get('best_val_loss', float('inf')) if hyper_no_gnn else float('inf'),
            'Total Params': hyper_no_gnn.get('total_params', 0) if hyper_no_gnn else 0,
            'Training Time (min)': parse_duration_to_minutes(
                hyper_no_gnn.get('training_duration', '')) if hyper_no_gnn else 0
        }
    }

    # Convertir a DataFrame de Pandas para ploteo fácil
    stats_df = pd.DataFrame.from_dict(stats_data, orient='index')
    logger.info("Estadísticas finales recopiladas:\n" + stats_df.to_string())

    # --- 3. Generar Gráficos ---
    logger.info("--- Generando Gráficos Comparativos ---")

    # Gráfico 1: Curvas de Pérdida Superpuestas
    plot_combined_loss_curves(hist_dict, output_plot_dir)

    # Gráfico 2: Comparación de Métricas Clave (Barras)
    plot_performance_barcharts(stats_df, output_plot_dir)

    logger.info("--- Proceso de Comparación Completado ---")


if __name__ == "__main__":
    # *** IMPORTANTE: CAMBIA ESTA RUTA ***
    # Debes poner la ruta a la carpeta que contiene TODOS tus archivos .json y .txt
    BASE_PROJECT_DIR = r"D:\Ruta\A\Tu\Carpeta\DeLogs"

    # --- Configuración de Rutas (Asumiendo que todos los archivos están en BASE_PROJECT_DIR) ---
    paths = {
        'gnn_base_log': os.path.join(BASE_PROJECT_DIR, 'training_log.txt'),
        'wavelet_gnn_log': os.path.join(BASE_PROJECT_DIR, 'loss_history_wavelet_gnn.json'),
        'no_gnn_log': os.path.join(BASE_PROJECT_DIR, 'loss_history_no_gnn.json'),
        'gnn_base_hyper': os.path.join(BASE_PROJECT_DIR, 'hyperparameters.json'),
        'wavelet_gnn_hyper': os.path.join(BASE_PROJECT_DIR, 'hyperparameters_wavelet_gnn.json'),
        'no_gnn_hyper': os.path.join(BASE_PROJECT_DIR, 'hyperparameters_no_gnn.json')
    }

    # Directorio de salida para los gráficos
    output_dir = os.path.join(BASE_PROJECT_DIR, "model_comparison_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Configurar logging
    logger = setup_logging(output_dir)

    # --- 1. Cargar Historiales de Pérdida ---
    logger.info("--- Cargando Historiales de Entrenamiento ---")

    hist_gnn, best_val_gnn = parse_gnn_base_log(paths['gnn_base_log'])
    hist_wavelet = load_json_data(paths['wavelet_gnn_log'])
    hist_no_gnn = load_json_data(paths['no_gnn_log'])

    hist_dict = {
        'GNN (Base)': hist_gnn,
        'Wavelet-GNN': hist_wavelet,
        'No-GNN AE': hist_no_gnn
    }

    # --- 2. Cargar Hiperparámetros y Métricas Finales ---
    logger.info("--- Extrayendo Métricas Finales de Hiperparámetros ---")

    hyper_wavelet = load_json_data(paths['wavelet_gnn_hyper'])
    hyper_no_gnn = load_json_data(paths['no_gnn_hyper'])
    # GNN-Base no tiene las métricas en su JSON, usamos las parseadas
    hyper_gnn_base = load_json_data(paths['gnn_base_hyper'])  # Para otros HParams si es necesario

    # Calcular duración de GNN-Base (hardcodeado de tu log)
    # 2025-09-10 07:56:19 (fin) - 2025-09-10 02:08:24 (inicio)
    duration_gnn_base_min = (timedelta(hours=7, minutes=56) - timedelta(hours=2, minutes=8)).total_seconds() / 60.0

    stats_data = {
        'GNN (Base)': {
            'Best Val Loss': best_val_gnn,
            'Total Params': np.nan,  # No disponible
            'Training Time (min)': duration_gnn_base_min
        },
        'Wavelet-GNN': {
            'Best Val Loss': hyper_wavelet.get('best_val_loss', float('inf')) if hyper_wavelet else float('inf'),
            'Total Params': hyper_wavelet.get('total_params', 0) if hyper_wavelet else 0,
            'Training Time (min)': parse_duration_to_minutes(
                hyper_wavelet.get('training_duration', '')) if hyper_wavelet else 0
        },
        'No-GNN AE': {
            'Best Val Loss': hyper_no_gnn.get('best_val_loss', float('inf')) if hyper_no_gnn else float('inf'),
            'Total Params': hyper_no_gnn.get('total_params', 0) if hyper_no_gnn else 0,
            'Training Time (min)': parse_duration_to_minutes(
                hyper_no_gnn.get('training_duration', '')) if hyper_no_gnn else 0
        }
    }

    stats_df = pd.DataFrame.from_dict(stats_data, orient='index')
    logger.info("Estadísticas finales recopiladas:\n" + stats_df.to_string())

    # --- 3. Generar Gráficos ---
    logger.info("--- Generando Gráficos Comparativos ---")

    plot_combined_loss_curves(hist_dict, output_dir)
    plot_performance_barcharts(stats_df, output_dir)

    logger.info(f"--- Proceso de Comparación Completado. Gráficos guardados en: {output_dir} ---")
