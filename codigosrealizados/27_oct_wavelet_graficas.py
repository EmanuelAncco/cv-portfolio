# -*- coding: utf-8 -*-
"""
plot_history.py

Carga un archivo `loss_history_wavelet_gnn.json` de una carpeta de resultados
y genera un nuevo gráfico de curvas de pérdida con:
1. Una escala LINEAL en el eje Y.
2. SIN la línea vertical de "Resumed...".

Uso:
1. Actualiza la variable `RUN_TO_ANALYZE_PATH` para que apunte a tu
   carpeta de resultados (la que contiene el .json).
2. Ejecuta el script.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging

# --- Configuración del Logging (solo consola) ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


def regenerate_plot_linear(run_folder_path):
    """Carga el historial y genera un nuevo gráfico lineal."""

    logger.info(f"Cargando historial desde: {run_folder_path}")

    history_path = os.path.join(run_folder_path, 'loss_history_wavelet_gnn.json')
    if not os.path.exists(history_path):
        logger.error(f"Archivo de historial no encontrado: {history_path}")
        return

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        logger.info("Historial de pérdidas cargado.")
    except Exception as e:
        logger.error(f"Error cargando JSON: {e}")
        return

    # --- Lógica de Ploteo Modificada ---
    try:
        # Extraer datos y limpiar Nones (si los hubiera)
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        train_loss = [l for l in history.get('train_loss', []) if l is not None and np.isfinite(l)]
        val_loss = [l for l in history.get('val_loss', []) if l is not None and np.isfinite(l)]
        epochs_train = [epochs[i] for i, l in enumerate(history.get('train_loss', [])) if
                        l is not None and np.isfinite(l)]
        epochs_val = [epochs[i] for i, l in enumerate(history.get('val_loss', [])) if l is not None and np.isfinite(l)]

        if not epochs_train or not epochs_val:
            logger.warning("No hay suficientes datos válidos de pérdida para plotear.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(epochs_train, train_loss, label='Training Loss', marker='.', linestyle='-', markersize=4)
        plt.plot(epochs_val, val_loss, label='Validation Loss', marker='.', linestyle='--', markersize=4)

        # 1. Título Modificado
        plt.title('Training & Validation Loss (STG-AE Wavelet - Final)')
        plt.xlabel('Epochs')

        # 2. Eje Y Lineal
        plt.ylabel('MSE Loss (Linear Scale)')

        # 3. Línea "Resumed" ELIMINADA
        # plt.axvline(x=start_epoch + 0.5, color='r', linestyle='--', label=f'Resumed at Epoch {start_epoch+1}')

        # Definir límites para la escala lineal
        all_losses = train_loss + val_loss
        min_loss = min(all_losses)
        max_loss = max(all_losses)

        # Establecer límites para que 0 sea visible y haya algo de espacio
        plt.ylim(bottom=0, top=max_loss * 1.1)
        plt.xlim(left=0, right=len(epochs) + 1)

        plt.legend()
        plt.grid(True, linestyle=':')

        # Nuevo nombre de archivo
        loss_curve_path = os.path.join(run_folder_path, 'loss_curve_wavelet_gnn_FINAL_linear.png')
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()

        logger.info(f"¡Éxito! Nuevo gráfico guardado en: {loss_curve_path}")

    except Exception as e:
        logger.error(f"Error generando gráfico de curvas de pérdida: {e}", exc_info=True)


if __name__ == '__main__':

    # --- ¡¡¡ CONFIGURAR ESTA RUTA !!! ---

    # 1. Apunta esto a la carpeta de resultados que CONTIENE el archivo
    #    `loss_history_wavelet_gnn.json` de tu entrenamiento finalizado.
    RUN_TO_ANALYZE_PATH = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento_wavelet\RESUME_run_wavelet_db45_h128_r256_lr0.0005_wd1e-05_20251027-143343_e50_lr0.0001_20251027-184547"

    # --- Fin de Configuración ---

    if not os.path.isdir(RUN_TO_ANALYZE_PATH):
        logger.error(f"Error: Directorio de resultados no encontrado: {RUN_TO_ANALYZE_PATH}")
        sys.exit(1)

    regenerate_plot_linear(RUN_TO_ANALYZE_PATH)
