import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


def load_sensor_data(file_path):
    """
    Carga los datos del sensor desde un archivo de texto.
    Devuelve la señal de aceleración y la frecuencia de muestreo estimada.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo no se encontró en la ruta: {file_path}")

    df = pd.read_csv(file_path, sep='\s+', header=None, names=['time', 'acceleration'])
    signal = df['acceleration'].values

    avg_time_delta = df['time'].diff().mean()
    fs = 1.0 / avg_time_delta if pd.notna(avg_time_delta) and avg_time_delta > 0 else 1000

    return signal, fs


def analyze_and_plot_samples(signal, fs, file_info, windows_to_analyze, window_size=512, n_fft_features=32):
    """
    Analiza y grafica múltiples ventanas de muestra para comparación.
    """
    num_samples = len(windows_to_analyze)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples), squeeze=False)
    fig.suptitle(f'Análisis Comparativo de Muestras: Del Tiempo a las Características de IA\n{file_info}', fontsize=16)

    all_features_to_export = {}

    for i, (sample_name, start_index) in enumerate(windows_to_analyze.items()):
        if start_index + window_size > len(signal):
            print(f"Advertencia: start_index para '{sample_name}' es muy alto. Se omitirá.")
            continue

        # --- Extracción y Cálculo para la Muestra Actual ---
        time_window = signal[start_index: start_index + window_size]
        time_axis = np.arange(window_size) / fs

        fft_result = np.fft.rfft(time_window)
        fft_magnitude = np.abs(fft_result)
        freq_axis = np.fft.rfftfreq(window_size, 1.0 / fs)

        feature_vector = fft_magnitude[1:n_fft_features + 1]
        feature_indices = np.arange(1, n_fft_features + 1)

        # Guardar para exportación
        all_features_to_export[sample_name] = {
            'start_index': start_index,
            'frequencies_hz': freq_axis[1:n_fft_features + 1],
            'magnitudes': feature_vector
        }

        # --- Visualización ---
        # Gráfica 1: Dominio del Tiempo
        axes[i, 0].plot(time_axis, time_window, color='dodgerblue')
        axes[i, 0].set_title(f'Paso 1: Señal Cruda ({sample_name})')
        axes[i, 0].set_ylabel('Aceleración')
        axes[i, 0].grid(True, linestyle='--', alpha=0.6)

        # Gráfica 2: Dominio de la Frecuencia con Anotaciones
        axes[i, 1].plot(freq_axis, fft_magnitude, color='crimson', marker='.', markersize=4)
        axes[i, 1].set_title(f'Paso 2: Espectro de Frecuencias')
        axes[i, 1].set_ylabel('Magnitud')
        axes[i, 1].set_xlim(0, fs / 8)
        axes[i, 1].grid(True, linestyle='--', alpha=0.6)

        # Evidencia de Cálculo: Anotaciones en el gráfico
        for k in [1, 2]:
            if k < len(freq_axis):
                freq_val = freq_axis[k]
                mag_val = fft_magnitude[k]
                axes[i, 1].annotate(f'k={k}\nf={freq_val:.2f} Hz',
                                    xy=(freq_val, mag_val),
                                    xytext=(freq_val + 2, mag_val + 0.1 * np.max(fft_magnitude)),
                                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                    fontsize=9)

        # Gráfica 3: Vector de Características para la IA
        axes[i, 2].bar(feature_indices, feature_vector, color='darkviolet')
        axes[i, 2].set_title(f'Paso 3: Vector de {n_fft_features} Características')
        axes[i, 2].set_ylabel('Magnitud')
        axes[i, 2].grid(True, linestyle='--', alpha=0.6)

    # Añadir etiquetas de eje X solo en la última fila
    for j in range(3):
        axes[num_samples - 1, j].set_xlabel(['Tiempo (s)', 'Frecuencia (Hz)', 'Índice de Característica'][j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.show()

    return all_features_to_export


def export_fft_features(features_dict, output_filename="fft_features_export.txt"):
    """
    Exporta los datos de características de FFT a un archivo de texto.
    """
    with open(output_filename, 'w') as f:
        f.write("Exportación de Características del Análisis de Fourier\n")
        f.write("=" * 50 + "\n")
        f.write(
            "Este archivo contiene los vectores de características que se usarían para entrenar el modelo de IA.\n\n")

        for sample_name, data in features_dict.items():
            f.write(f"--- {sample_name} (iniciando en la muestra #{data['start_index']}) ---\n")
            f.write(f"{'Índice':<10} | {'Frecuencia (Hz)':<20} | {'Magnitud':<20}\n")
            f.write("-" * 55 + "\n")
            for i in range(len(data['magnitudes'])):
                idx = i + 1
                freq = data['frequencies_hz'][i]
                mag = data['magnitudes'][i]
                f.write(f"{idx:<10} | {freq:<20.4f} | {mag:<20.6f}\n")
            f.write("\n\n")
    print(f"-> Los datos transformados se han guardado exitosamente en: '{output_filename}'")


def get_sort_key_from_filename(filename):
    try:
        parts = filename.replace('.txt', '').split(' ')
        date_part = parts[0].split('_')[1]
        time_part = parts[1].replace('_', '')
        day, month = date_part[:2], date_part[2:]
        return f"2025{month}{day}{time_part}"
    except IndexError:
        return "999999999999"


if __name__ == '__main__':
    data_directory = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    if not os.path.isdir(data_directory):
        print(f"Error: Directorio no encontrado: {data_directory}")
    else:
        print("\n--- Herramienta Avanzada de Análisis de Sensor con FFT ---")
        sensor_1_files = sorted(
            [f for f in os.listdir(data_directory) if f.endswith('.txt') and f.startswith('1_')],
            key=get_sort_key_from_filename
        )

        if len(sensor_1_files) < 1:
            print(f"No se encontraron archivos del Sensor 1 en: {data_directory}")
        else:
            print("Concatenando archivos en orden cronológico...")
            full_signal_parts = []
            estimated_fs = 1000
            for file_name in sensor_1_files:
                try:
                    signal, fs = load_sensor_data(os.path.join(data_directory, file_name))
                    full_signal_parts.append(signal)
                    estimated_fs = fs
                except Exception as e:
                    print(f"  Advertencia al cargar '{file_name}': {e}")

            concatenated_signal = np.concatenate(full_signal_parts)
            print(f"Concatenación completa. Señal total: {len(concatenated_signal)} muestras.")
            print(f"Frecuencia de muestreo estimada: {estimated_fs:.2f} Hz\n")

            # Definir las dos ventanas de muestra a analizar
            windows_to_analyze = {
                'Muestra A': 20000,
                'Muestra B': 150000
            }

            # Ejecutar análisis, visualización y obtener datos para exportar
            features_to_export = analyze_and_plot_samples(
                concatenated_signal,
                estimated_fs,
                f"Señal combinada de {len(sensor_1_files)} archivos del Sensor 1",
                windows_to_analyze,
                window_size=512,
                n_fft_features=32
            )

            # Exportar los datos a un archivo de texto
            if features_to_export:
                export_fft_features(features_to_export)

