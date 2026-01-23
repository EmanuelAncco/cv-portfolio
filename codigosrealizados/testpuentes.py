import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# --- REPLICACIÓN DE LA ARQUITECTURA DEL MODELO Y CLASES DE DATOS ---
from torch_geometric.nn import GCNConv


def define_bridge_graph():
    """Define la topología del puente como un grafo."""
    edge_index = torch.tensor([
        [0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1],
        [2, 3], [3, 2], [2, 4], [4, 2], [3, 4], [4, 3],
    ], dtype=torch.long).t().contiguous()
    return edge_index


class SpatioTemporalWindowDataset(Dataset):
    """Crea ventanas deslizantes a partir de datos de múltiples sensores."""

    def __init__(self, data_dict, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride
        min_len = min(len(data) for data in data_dict.values())
        self.data = np.stack([data[:min_len] for sid, data in sorted(data_dict.items())], axis=1)
        self.n_samples = (len(self.data) - window_size) // stride + 1
        if self.n_samples <= 0:
            self.n_samples = 0
            raise ValueError("Not enough data to create a single window.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.data[start:end]
        return torch.FloatTensor(window), torch.FloatTensor(window)


class GNNLayer(nn.Module):
    """Capa de Red Neuronal de Grafo (GNN)."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class SpatioTemporalAutoencoder(nn.Module):
    """Arquitectura del Autoencoder Híbrido GNN-GRU (Versión Asimétrica Corregida)."""

    def __init__(self, num_nodes, num_features, window_size, gnn_hidden=32, gnn_out=16, rnn_hidden=64,
                 gnn_out_decoder=16):
        super(SpatioTemporalAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.gnn_encoder = GNNLayer(num_features, gnn_hidden, gnn_out)
        self.rnn_encoder = nn.GRU(input_size=gnn_out * num_nodes, hidden_size=rnn_hidden, batch_first=True,
                                  num_layers=2)
        # El decodificador usa una dimensión de entrada diferente (gnn_out_decoder)
        self.rnn_decoder = nn.GRU(input_size=rnn_hidden, hidden_size=gnn_out_decoder * num_nodes, batch_first=True,
                                  num_layers=2)
        self.gnn_decoder = GNNLayer(gnn_out_decoder, gnn_hidden, num_features)

        # Guardamos este valor para usarlo en el forward pass
        self.gnn_out_decoder = gnn_out_decoder

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        gnn_encoded_steps = []
        for t in range(self.window_size):
            snapshot = x[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(batch_size,
                                                                               device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            gnn_out = self.gnn_encoder(snapshot, batch_edge_index)
            gnn_encoded_steps.append(gnn_out.reshape(batch_size, self.num_nodes, -1))

        gnn_encoded = torch.stack(gnn_encoded_steps, dim=1)
        gnn_encoded_flat = gnn_encoded.reshape(batch_size, self.window_size, -1)
        _, hidden_state = self.rnn_encoder(gnn_encoded_flat)

        # --- LÓGICA CORREGIDA ---
        context_vector = hidden_state[-1]
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.window_size, 1)

        rnn_decoded, _ = self.rnn_decoder(decoder_input)
        rnn_decoded_unflat = rnn_decoded.reshape(batch_size, self.window_size, self.num_nodes, self.gnn_out_decoder)

        reconstructed_steps = []
        for t in range(self.window_size):
            snapshot = rnn_decoded_unflat[:, t, :, :].reshape(batch_size * self.num_nodes, -1)
            batch_edge_index = edge_index.repeat(1, batch_size) + torch.arange(batch_size,
                                                                               device=x.device).repeat_interleave(
                edge_index.size(1)) * self.num_nodes
            reconstructed_snapshot = self.gnn_decoder(snapshot, batch_edge_index)
            reconstructed_steps.append(reconstructed_snapshot.reshape(batch_size, self.num_nodes, -1))
        return torch.stack(reconstructed_steps, dim=1)


# --- FUNCIÓN PRINCIPAL DE INFERENCIA Y VISUALIZACIÓN ---

def run_inference_and_visualize(training_run_dir, damage_data_dir, healthy_data_dir, output_plot_dir):
    """
    Carga un modelo entrenado, ejecuta la inferencia sobre nuevos datos de daño
    y genera gráficos para validar y visualizar los resultados.
    """
    print("--- Iniciando Proceso de Inferencia y Visualización ---")
    os.makedirs(output_plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Cargar artefactos del entrenamiento
    print(f"Cargando artefactos desde: {training_run_dir}")
    scaler_path = os.path.join(training_run_dir, 'scaler.gz')

    try:
        scaler = joblib.load(scaler_path)
        print("Scaler cargado exitosamente desde archivo.")
    except FileNotFoundError:
        print(f"Advertencia: No se encontró '{scaler_path}'.")
        print("Creando y guardando un nuevo scaler a partir de los datos de entrenamiento sanos...")

        try:
            all_files = [os.path.join(healthy_data_dir, f) for f in os.listdir(healthy_data_dir) if f.endswith('.txt')]
            sensor_data_healthy = []
            for f in all_files:
                try:
                    sensor_data_healthy.append(pd.read_csv(f, sep='\s+', header=None, usecols=[1]).values)
                except Exception:
                    continue

            if not sensor_data_healthy:
                print("Error: No se pudieron cargar los datos sanos para crear el scaler.")
                return

            concatenated_healthy_data = np.concatenate(sensor_data_healthy)
            scaler = StandardScaler()
            scaler.fit(concatenated_healthy_data)
            joblib.dump(scaler, scaler_path)
            print(f"Nuevo scaler creado y guardado en: {scaler_path}")

        except Exception as e:
            print(f"Error fatal al intentar crear el scaler: {e}")
            return

    try:
        with open(os.path.join(training_run_dir, 'hyperparameters.json'), 'r') as f:
            hp = json.load(f)
        model_path = os.path.join(training_run_dir, 'best_model.pth')
        num_nodes = 5

        model = SpatioTemporalAutoencoder(
            num_nodes=num_nodes, num_features=1, window_size=hp['window_size'],
            gnn_hidden=hp['gnn_hidden'], rnn_hidden=hp['rnn_hidden'],
            gnn_out=16, gnn_out_decoder=32
        ).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Modelo y hiperparámetros cargados exitosamente.")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # 2. Cargar y preprocesar los datos de daño
    print(f"Cargando datos de daño desde: {damage_data_dir}")
    damage_data = {}
    damage_files = [f for f in os.listdir(damage_data_dir) if f.endswith('.txt')]

    # Lógica para agrupar archivos por ejemplo (ej: _ejemplo_1, _ejemplo_2)
    num_examples = len(set(f.split('_')[-1].split('.')[0] for f in damage_files))

    # Vamos a procesar solo el primer conjunto de archivos de ejemplo que encontremos
    target_example_id = "ejemplo_1"  # o el que se quiera
    files_to_load = [f for f in damage_files if target_example_id in f]
    if not files_to_load:
        files_to_load = damage_files  # Fallback a cargar lo que haya si no hay ejemplos

    print(f"Procesando el conjunto de archivos: {files_to_load}")

    for i in range(1, num_nodes + 1):
        # Buscar un archivo que corresponda al sensor 'i' y al ejemplo_id
        file_found = next((f for f in files_to_load if f.startswith(f"{i}_")), None)

        if file_found:
            file_path = os.path.join(damage_data_dir, file_found)
            damage_data[i] = pd.read_csv(file_path, sep='\s+', header=None, usecols=[1]).values
        else:
            # Fallback para el formato original
            file_path = os.path.join(damage_data_dir, f'{i}_sismo.txt')
            if os.path.exists(file_path):
                damage_data[i] = pd.read_csv(file_path, sep='\s+', header=None, usecols=[1]).values
            else:
                print(f"Advertencia: No se encontró archivo para el sensor {i}.")
                return

    scaled_damage_data = {sid: scaler.transform(data) for sid, data in damage_data.items()}
    damage_dataset = SpatioTemporalWindowDataset(scaled_damage_data, hp['window_size'], hp['stride'])
    damage_loader = DataLoader(damage_dataset, batch_size=hp['batch_size'], shuffle=False)
    edge_index = define_bridge_graph().to(device)

    # 3. Ejecutar inferencia y calcular errores
    print("Ejecutando inferencia sobre los datos de daño...")
    all_errors, all_originals, all_reconstructed = [], [], []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for inputs, _ in tqdm(damage_loader, desc="Procesando datos de daño"):
            inputs = inputs.to(device)
            outputs = model(inputs, edge_index)
            all_originals.append(inputs.cpu().numpy())
            all_reconstructed.append(outputs.cpu().numpy())
            all_errors.append(criterion(outputs, inputs).cpu().numpy())

    all_errors = np.concatenate(all_errors, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    mean_error_per_node = np.mean(all_errors, axis=(0, 1, 3))

    print("\n--- Resultados de Inferencia ---")
    for i, err in enumerate(mean_error_per_node):
        print(f"  - Error de Reconstrucción Medio para Sensor {i + 1}: {err:.6f}")

    # 4. Generar Gráficos
    print("\nGenerando gráficos...")

    # --- Gráfico 1: Comparación de Señal Original vs. Reconstruida (Ventana con Mayor Error) ---
    idx_max_error = np.argmax(np.mean(all_errors, axis=(1, 2, 3)))
    original_sample = all_originals[idx_max_error]
    reconstructed_sample = all_reconstructed[idx_max_error]

    fig, axes = plt.subplots(num_nodes, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Gráfico 1: Comparación de Señal (Ventana con Mayor Error)', fontsize=16)
    for i in range(num_nodes):
        axes[i].plot(original_sample[:, i, 0], label='Original (Dañada)', color='b')
        axes[i].plot(reconstructed_sample[:, i, 0], label='Reconstruida', color='r', linestyle='--')
        axes[i].set_title(f'Sensor {i + 1}')
        axes[i].set_ylabel('Aceleración Norm.')
        axes[i].legend()
        axes[i].grid(True)
    axes[-1].set_xlabel('Paso de Tiempo en la Ventana')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot1_path = os.path.join(output_plot_dir, "1_comparacion_senales_max_error.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Gráfico 1 guardado en: {plot1_path}")

    # --- Gráfico 2: Error de Reconstrucción por Sensor (Gráfico de Barras) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f'Sensor {i + 1}' for i in range(num_nodes)], y=mean_error_per_node)
    plt.title('Gráfico 2: Error de Reconstrucción Medio por Sensor', fontsize=16)
    plt.ylabel('Error Cuadrático Medio (MSE)')
    plt.xlabel('Sensor')
    plt.grid(axis='y')
    plot2_path = os.path.join(output_plot_dir, "2_error_por_sensor.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Gráfico 2 guardado en: {plot2_path}")

    # --- Gráfico 3: Mapa de Calor de Localización de Daño ---
    sensor_coords = {1: (1, 3), 2: (1, 1), 3: (2, 2), 4: (3, 3), 5: (3, 1)}
    x_coords = [c[0] for c in sensor_coords.values()]
    y_coords = [c[1] for c in sensor_coords.values()]
    plt.figure(figsize=(12, 7))
    plt.scatter(x_coords, y_coords, c=mean_error_per_node, cmap='coolwarm', s=1000, edgecolors='k')
    plt.plot([0, 4], [4, 4], 'k-', lw=2);
    plt.plot([0, 4], [0, 0], 'k-', lw=2)
    plt.fill_between([0, 4], 0, 4, color='grey', alpha=0.1)
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f'S{i + 1}', ha='center', va='center', color='white', fontsize=12, weight='bold')
    plt.title('Gráfico 3: Mapa de Calor para Localización de Anomalías', fontsize=16)
    plt.xlabel('Eje Longitudinal (Esquemático)');
    plt.ylabel('Eje Transversal (Esquemático)')
    cbar = plt.colorbar();
    cbar.set_label('Error de Reconstrucción (MSE)')
    plt.grid(True, linestyle='--', alpha=0.6);
    plt.xlim(-0.5, 4.5);
    plt.ylim(-0.5, 4.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plot3_path = os.path.join(output_plot_dir, "3_heatmap_localizacion.png")
    plt.savefig(plot3_path)
    plt.close()
    print(f"Gráfico 3 guardado en: {plot3_path}")

    # --- ¡NUEVO! Gráfico 4: Comparaciones Aleatorias de Señales ---
    print("\nGenerando gráficos de comparación aleatorios...")
    NUM_RANDOM_PLOTS = 3
    num_windows = all_originals.shape[0]

    if num_windows < NUM_RANDOM_PLOTS:
        print(f"Advertencia: Solo hay {num_windows} ventanas. Se generarán {num_windows} gráficos.")
        NUM_RANDOM_PLOTS = num_windows

    random_indices = np.random.choice(num_windows, size=NUM_RANDOM_PLOTS, replace=False)

    for i, idx in enumerate(random_indices):
        original_sample_rand = all_originals[idx]
        reconstructed_sample_rand = all_reconstructed[idx]

        fig, axes = plt.subplots(num_nodes, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Gráfico 4.{i + 1}: Comparación Aleatoria (Ventana de muestra #{idx})', fontsize=16)

        for sensor_idx in range(num_nodes):
            axes[sensor_idx].plot(original_sample_rand[:, sensor_idx, 0], label='Original (Dañada)', color='b')
            axes[sensor_idx].plot(reconstructed_sample_rand[:, sensor_idx, 0], label='Reconstruida', color='r',
                                  linestyle='--')
            axes[sensor_idx].set_title(f'Sensor {sensor_idx + 1}')
            axes[sensor_idx].set_ylabel('Aceleración Norm.')
            axes[sensor_idx].legend()
            axes[sensor_idx].grid(True)

        axes[-1].set_xlabel('Paso de Tiempo en la Ventana')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_path = os.path.join(output_plot_dir, f"4_comparacion_aleatoria_{i + 1}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico de comparación aleatoria {i + 1} guardado en: {plot_path}")

    print("\n--- Proceso Finalizado ---")


# --- CONFIGURACIÓN Y EJECUCIÓN ---
if __name__ == '__main__':
    # 1. Ruta al directorio que contiene los resultados de un entrenamiento específico.
    PATH_AL_MODELO_ENTRENADO = r"D:\Python_proyectos_2025\GAIATECH\resultados_entrenamiento\run_gnn_20250910-020756"

    # 2. Ruta al directorio que contiene los datos de entrenamiento SANOS.
    #    Este directorio se usará para RE-CREAR el scaler si no se encuentra.
    PATH_A_DATOS_SANOS = r"D:\descargas 2025\limpiar-20250619T152105Z-1-001\limpiar"

    # 3. Ruta al directorio que contiene los archivos de aceleraciones con DAÑO.
    PATH_A_DATOS_CON_DAÑO = r"D:\descargas 2025-2\articulo tesis delgadillo\Aceleraciones con daño\Aceleraciones"

    # 4. Ruta al directorio donde se guardarán los gráficos generados.

    PATH_PARA_GUARDAR_GRAFICOS = "resultados_inferencia_aleatorios"

    if not os.path.isdir(PATH_AL_MODELO_ENTRENADO):
        print(f"Error: La ruta al modelo entrenado no existe: '{PATH_AL_MODELO_ENTRENADO}'")
    elif not os.path.isdir(PATH_A_DATOS_CON_DAÑO):
        print(f"Error: La ruta a los datos con daño no existe: '{PATH_A_DATOS_CON_DAÑO}'")
    else:
        run_inference_and_visualize(
            training_run_dir=PATH_AL_MODELO_ENTRENADO,
            damage_data_dir=PATH_A_DATOS_CON_DAÑO,
            healthy_data_dir=PATH_A_DATOS_SANOS,
            output_plot_dir=PATH_PARA_GUARDAR_GRAFICOS
        )