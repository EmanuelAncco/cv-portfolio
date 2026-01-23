# -*- coding: utf-8 -*-
"""
1D Signal Processor and Autoencoder Trainer for Local Execution

Instructions for local setup:
1.  Save this script as a Python file (e.g., `train_model.py`).
2.  Create a folder named 'data' in the same directory as this script.
3.  Place all your '.txt' data files inside the 'data' folder.
4.  Install the required libraries by running the following command in your terminal:
    pip install numpy pandas matplotlib seaborn torch scikit-learn scikit-image
5.  Run the script from your terminal:
    python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import StandardScaler
# NEW: Import for SSIM metric and defaultdict
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

# --- Configuration ---
# Directorio con datos NORMALES para entrenamiento
DATA_DIR = r'D:\Python_proyectos_2025\JEAN SISMOS\DATA'
# Directorio con datos ANÓMALOS (sismo).
ANOMALY_DATA_DIR = r'D:\Python_proyectos_2025\JEAN SISMOS\DATA6.1'

MODEL_FILE = 'best_model.pth'  # Define model filename
HISTORY_FILE = 'training_history.json'  # Define history filename

# Check if the data directory exists
if not os.path.isdir(DATA_DIR):
    print(f"Error: Data directory '{DATA_DIR}' not found.")
    print("Please check if the path is correct.")
    exit()

# --- Data Loading ---
# Load all '.txt' files from the specified directory.
data_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])  # Sorted to keep order
if not data_files:
    print(f"Error: No '.txt' files found in the '{DATA_DIR}' directory.")
    exit()

print(f"Found {len(data_files)} data file(s): {data_files}")

# Load all data files into a list
all_data = []
for file_i in data_files:
    file_path = os.path.join(DATA_DIR, file_i)
    try:
        all_data.append(np.loadtxt(file_path))
    except Exception as e:
        print(f"Could not read file {file_i}: {e}")

# --- Preprocessing for Training ---
# For training, we will combine all data into one big dataset to train a robust model.
# We still only use the second column (accelerations).
combined_data_for_training = np.concatenate([d[:, 1:2] for d in all_data], axis=0)

scaler = StandardScaler()
data_scaled_for_training = scaler.fit_transform(combined_data_for_training)


# --- Dataset and DataLoader ---
class SlidingWindowDataset1D(Dataset):
    """
    A PyTorch Dataset for creating sliding windows from a 1D signal.
    """

    def __init__(self, data, window_size, stride=1, transform=None):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if data.dim() == 1:
            data = data.unsqueeze(1)
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.num_windows = (len(self.data) - self.window_size) // self.stride + 1
        if self.num_windows <= 0:
            raise ValueError(f"Not enough data for a single window.")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        window = self.data[start_idx:end_idx]
        if self.transform:
            window = self.transform(window)
        return window


# Information
window_size = 128
stride = 64

# Create the dataset from the combined data
dataset = SlidingWindowDataset1D(data_scaled_for_training, window_size, stride)

# Split into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
batch_size = 100
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n--- DataLoader Info ---")
print(f"Total windows for training/validation: {len(dataset)}")
print(f"Training windows: {len(train_dataset)}")
print(f"Test windows: {len(test_dataset)}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of test batches: {len(test_dataloader)}")
print("-----------------------\n")


# --- Model Definition ---
class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super(SelfAttention1D, self).__init__()
        self.channels = channels
        self.query_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        x = x.permute(0, 2, 1)
        query = self.query_conv(x);
        key = self.key_conv(x);
        value = self.value_conv(x)
        query = query.permute(0, 2, 1);
        key = key.permute(0, 2, 1);
        value = value.permute(0, 2, 1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output


class ModelWithSkipConnections(nn.Module):
    def __init__(self, input_channels, base_channels, input_sequence_length,
                 internal_channels_list=None, output_channels_linear=1):
        super(ModelWithSkipConnections, self).__init__()
        if internal_channels_list is None:
            internal_channels_list = [max(1, base_channels // (2 ** i)) for i in range(1, 4)]
        if not internal_channels_list or len(internal_channels_list) < 1:
            raise ValueError("internal_channels_list must contain at least one channel size.")
        self.input_channels = input_channels;
        self.base_channels = base_channels
        self.input_sequence_length = input_sequence_length
        self.internal_channels_list = internal_channels_list
        self.num_downsample_blocks = len(internal_channels_list)
        if input_channels != base_channels:
            self.initial_projection = nn.Conv1d(input_channels, base_channels, kernel_size=1)
            self_attention_input_channels = base_channels
        else:
            self.initial_projection = nn.Identity()
            self_attention_input_channels = input_channels
        self.self_attention = SelfAttention1D(self_attention_input_channels)
        encoder_blocks = nn.ModuleList()
        current_in_channels = self_attention_input_channels
        for i in range(self.num_downsample_blocks):
            out_c = internal_channels_list[i]
            encoder_blocks.append(nn.Sequential(
                nn.Conv1d(current_in_channels, out_c, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm1d(out_c), nn.LeakyReLU(0.1)))
            current_in_channels = out_c
        self.encoder_blocks = encoder_blocks
        self.encoder_deepest_channels = current_in_channels
        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.encoder_deepest_channels, self.encoder_deepest_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.encoder_deepest_channels * 2), nn.LeakyReLU(0.1),
            nn.Conv1d(self.encoder_deepest_channels * 2, self.encoder_deepest_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.encoder_deepest_channels), nn.LeakyReLU(0.1))
        decoder_blocks = nn.ModuleList()
        current_decoder_in_channels = self.encoder_deepest_channels
        for i in range(self.num_downsample_blocks):
            if i == 0:
                skip_channels = internal_channels_list[self.num_downsample_blocks - 1]
            else:
                skip_channels = internal_channels_list[self.num_downsample_blocks - 1 - i]
            if i == self.num_downsample_blocks - 1:
                out_channels_decoder_block = base_channels
            else:
                out_channels_decoder_block = internal_channels_list[self.num_downsample_blocks - 2 - i]
            decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose1d(current_decoder_in_channels, out_channels_decoder_block,
                                   kernel_size=3, padding=1, output_padding=1, stride=2),
                nn.BatchNorm1d(out_channels_decoder_block), nn.LeakyReLU(0.1),
                nn.Conv1d(out_channels_decoder_block + skip_channels, out_channels_decoder_block, kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels_decoder_block), nn.LeakyReLU(0.1)))
            current_decoder_in_channels = out_channels_decoder_block
        self.decoder_blocks = decoder_blocks
        self.fc_final_output = nn.Linear(base_channels, output_channels_linear)
        self.final_activation = nn.Identity()
        self.expected_decoder_output_length = (self.input_sequence_length // (2 ** self.num_downsample_blocks)) * (
                    2 ** self.num_downsample_blocks)
        self.fc_length_adjust = nn.Linear(
            self.base_channels * self.expected_decoder_output_length,
            self.base_channels * self.input_sequence_length)

    def forward(self, x):
        original_length = x.shape[1]
        if original_length != self.input_sequence_length:
            raise ValueError(f"Input length {original_length} != expected {self.input_sequence_length}.")
        batch_size = x.shape[0]
        x_proj = x.permute(0, 2, 1)
        x_proj = self.initial_projection(x_proj)
        x_proj = x_proj.permute(0, 2, 1)
        x_sa = self.self_attention(x_proj)
        x_enc = x_sa.permute(0, 2, 1)
        skip_connections = [x_enc]
        current_x_enc = x_enc
        for block in self.encoder_blocks:
            current_x_enc = block(current_x_enc)
            skip_connections.append(current_x_enc)
        x = self.bottleneck(current_x_enc)
        for i, block in enumerate(self.decoder_blocks):
            x_skip = skip_connections[self.num_downsample_blocks - i]
            x_up = block[:3](x)
            if x_up.shape[2] != x_skip.shape[2]:
                x_up = F.interpolate(x_up, size=x_skip.shape[2], mode='linear', align_corners=True)
            x_combined = torch.cat([x_up, x_skip], dim=1)
            x = block[3:](x_combined)
        if x.shape[2] != self.expected_decoder_output_length:
            x = F.interpolate(x, size=self.expected_decoder_output_length, mode='linear', align_corners=True)
        x = x.reshape(batch_size, -1)
        x = self.fc_length_adjust(x)
        x = x.reshape(batch_size, self.input_sequence_length, self.base_channels)
        x = self.fc_final_output(x)
        x = self.final_activation(x)
        return x


# --- Model Initialization ---
model = ModelWithSkipConnections(
    input_channels=1, base_channels=64, input_sequence_length=window_size,
    internal_channels_list=[64, 128, 256], output_channels_linear=1)

# --- Training Loop ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model.to(device)

# *** Lógica para saltar el entrenamiento si el modelo ya existe ***
if os.path.exists(MODEL_FILE):
    print(f"\nPre-trained model '{MODEL_FILE}' found. Skipping training.")
else:
    print(f"\nModel file '{MODEL_FILE}' not found. Starting training...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 100
    best_loss = float('inf')
    patience = 10;
    counter = 0
    final_num_epoch = num_epochs
    train_losses, test_losses, test_ssims = [], [], []

    print("--- Starting Training ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in train_dataloader:
            inputs = inputs.to(device)
            targets = inputs.clone()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)

        model.eval()
        running_test_loss = 0.0
        running_test_ssim = 0.0
        with torch.no_grad():
            for inputs in test_dataloader:
                inputs = inputs.to(device)
                targets = inputs.clone()
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                running_test_loss += test_loss.item() * inputs.size(0)

                # Calcular SSIM para el batch
                for i in range(inputs.size(0)):
                    original = inputs[i].squeeze().cpu().numpy()
                    reconstructed = outputs[i].squeeze().cpu().detach().numpy()
                    data_range = original.max() - original.min()
                    if data_range == 0: data_range = 1
                    running_test_ssim += ssim(original.reshape(8, 16), reconstructed.reshape(8, 16),
                                              data_range=data_range)

        epoch_test_loss = running_test_loss / len(test_dataset)
        epoch_test_ssim = running_test_ssim / len(test_dataset)
        test_losses.append(epoch_test_loss)
        test_ssims.append(epoch_test_ssim)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4e}, Test Loss: {epoch_test_loss:.4e}, Test SSIM: {epoch_test_ssim:.4f}')

        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss;
            counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            counter += 1
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}.')
            final_num_epoch = epoch + 1
            break
    print("Training finished.")

    # Guardar el historial de entrenamiento
    history = {'train_loss': train_losses, 'test_loss': test_losses, 'test_ssim': test_ssims}
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {HISTORY_FILE}")

# --- Cargar y mostrar la curva de entrenamiento ---
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    train_losses = history['train_loss']
    test_losses = history['test_loss']
    test_ssims = history.get('test_ssim', [])  # Usar .get para compatibilidad con historiales viejos

    # Gráfico de entrenamiento con dos subplots para Pérdida y Precisión (SSIM)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Training & Validation Performance per Epoch', fontsize=16)

    # Subplot 1: Pérdida
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='royalblue')
    ax1.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='darkorange')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Model Loss')

    # Subplot 2: Precisión (SSIM)
    if test_ssims:
        ax2.plot(range(1, len(test_ssims) + 1), test_ssims, label='Test SSIM', color='forestgreen')
        ax2.set_ylabel('SSIM')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Model Precision (SSIM)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

else:
    print("No training history file found. Run training to generate it.")

# Cargar el mejor modelo guardado para la evaluación final
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
print(f"\nLoaded best model ('{MODEL_FILE}') for final evaluation.")

# --- Scenario-Based Evaluation and Plotting ---
print("\n--- Starting Scenario-Based Evaluation (Normal Data) ---")
model.eval()
sensor_results = defaultdict(lambda: {'errors': [], 'ssims': []})
mse_loss = nn.MSELoss()

with torch.no_grad():
    for filename, data_file_content in zip(data_files, all_data):
        try:
            sensor_id = int(filename[0])
        except (ValueError, IndexError):
            print(f"Could not determine sensor ID for file {filename}. Skipping.")
            continue
        print(f"Evaluating file {filename} for Sensor {sensor_id}...")
        scenario_data = scaler.transform(data_file_content[:, 1:2])
        scenario_dataset = SlidingWindowDataset1D(scenario_data, window_size, stride)
        scenario_dataloader = DataLoader(scenario_dataset, batch_size=batch_size, shuffle=False)
        if len(scenario_dataset) == 0: continue
        for original_windows in scenario_dataloader:
            original_windows = original_windows.to(device)
            reconstructed_windows = model(original_windows)
            for i in range(original_windows.size(0)):
                original = original_windows[i].squeeze().cpu().numpy()
                reconstructed = reconstructed_windows[i].squeeze().cpu().detach().numpy()
                error = mse_loss(torch.tensor(reconstructed), torch.tensor(original)).item()
                sensor_results[sensor_id]['errors'].append(error)
                data_range = np.max(original) - np.min(original)
                if data_range == 0: data_range = 1
                similarity = ssim(original.reshape(8, 16), reconstructed.reshape(8, 16), data_range=data_range)
                sensor_results[sensor_id]['ssims'].append(similarity)

# --- Creación de los gráficos de análisis ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Evaluation Results of Normal Data by Sensor', fontsize=16)

results_list = []
for sensor_id, data in sensor_results.items():
    for i in range(len(data['errors'])):
        results_list.append({'sensor': sensor_id, 'error': data['errors'][i], 'ssim': data['ssims'][i]})
results_df = pd.DataFrame(results_list)

unique_sensors = sorted(results_df['sensor'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sensors)))
color_map = {sensor_id: color for sensor_id, color in zip(unique_sensors, colors)}

error_upper_bound = results_df['error'].quantile(0.98)
ssim_lower_bound = results_df['ssim'].quantile(0.02)
error_display_max = error_upper_bound * 1.1
ssim_display_min = ssim_lower_bound * 0.98

current_x_offset = 0
for sensor_id in unique_sensors:
    sensor_df = results_df[results_df['sensor'] == sensor_id]
    n_samples = len(sensor_df)
    if n_samples == 0: continue
    x_values = np.arange(current_x_offset, current_x_offset + n_samples)
    color = color_map[sensor_id]
    axes[0, 0].plot(x_values, sensor_df['error'], marker='o', linestyle='None', color=color, alpha=0.5, markersize=3)
    axes[0, 1].plot(x_values, sensor_df['ssim'], marker='o', linestyle='None', color=color, alpha=0.5, markersize=3)
    current_x_offset += n_samples

axes[0, 0].set_title('(a) Error vs Sample Index')
axes[0, 0].set_xlabel('The number of samples');
axes[0, 0].set_ylabel('Error (MSE)')
axes[0, 0].grid(True);
axes[0, 0].set_ylim(0, error_display_max)
axes[0, 1].set_title('(b) SSIM vs Sample Index')
axes[0, 1].set_xlabel('The number of samples');
axes[0, 1].set_ylabel('SSIM')
axes[0, 1].grid(True);
axes[0, 1].set_ylim(ssim_display_min, 1.01)

sns.boxplot(ax=axes[1, 0], x='sensor', y='error', data=results_df, palette=color_map, hue='sensor', legend=False,
            showfliers=False)
axes[1, 0].set_title('(c) Error by Sensor');
axes[1, 0].set_xlabel('Sensors');
axes[1, 0].set_ylabel('Error (MSE)')
axes[1, 0].grid(True);
axes[1, 0].set_ylim(0, error_display_max)
sns.boxplot(ax=axes[1, 1], x='sensor', y='ssim', data=results_df, palette=color_map, hue='sensor', legend=False,
            showfliers=False)
axes[1, 1].set_title('(d) SSIM by Sensor');
axes[1, 1].set_xlabel('Sensors');
axes[1, 1].set_ylabel('SSIM')
axes[1, 1].grid(True);
axes[1, 1].set_ylim(ssim_display_min, 1.01)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
plt.show()

# --- NUEVO: Análisis de Datos Anómalos (Sismo) ---
if os.path.isdir(ANOMALY_DATA_DIR):
    print(f"\n--- Starting Anomaly Data Evaluation (Sismo 6.1) ---")
    anomaly_files = sorted([f for f in os.listdir(ANOMALY_DATA_DIR) if f.endswith('.txt')])
    anomaly_results_list = []
    anomaly_data_samples = []

    if not anomaly_files:
        print(f"No .txt files found in anomaly data directory: {ANOMALY_DATA_DIR}")
    else:
        with torch.no_grad():
            for filename in anomaly_files:
                print(f"Evaluating anomaly file {filename}...")
                file_path = os.path.join(ANOMALY_DATA_DIR, filename)
                anomaly_data_raw = np.loadtxt(file_path)

                # Extraer ID del sensor del nombre del archivo (ej. 'sensor1.txt')
                try:
                    sensor_id_str = ''.join(filter(str.isdigit, filename))
                    sensor_id = int(sensor_id_str)
                except:
                    print(f"Could not parse sensor ID from {filename}, skipping sensor ID for this file.")
                    sensor_id = 0  # ID por defecto

                anomaly_data_scaled = scaler.transform(anomaly_data_raw[:, 1:2])

                anomaly_dataset = SlidingWindowDataset1D(anomaly_data_scaled, window_size, stride)
                anomaly_dataloader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=False)

                if len(anomaly_dataset) > 0 and len(anomaly_data_samples) < 3:
                    anomaly_data_samples.append(anomaly_dataset[0])

                for original_windows in anomaly_dataloader:
                    original_windows = original_windows.to(device)
                    reconstructed_windows = model(original_windows)
                    for i in range(original_windows.size(0)):
                        original = original_windows[i]
                        reconstructed = reconstructed_windows[i]
                        error = mse_loss(reconstructed, original).item()

                        original_np = original.squeeze().cpu().numpy()
                        reconstructed_np = reconstructed.squeeze().cpu().detach().numpy()
                        data_range = original_np.max() - original_np.min()
                        if data_range == 0: data_range = 1
                        similarity = ssim(original_np.reshape(8, 16), reconstructed_np.reshape(8, 16),
                                          data_range=data_range)

                        anomaly_results_list.append({'sensor': sensor_id, 'error': error, 'ssim': similarity})

        anomaly_df = pd.DataFrame(anomaly_results_list)

        # --- Gráfico de Caja de Bigotes Comparativo por Sensor ---
        normal_error_df = results_df[['sensor', 'error']].copy()
        normal_error_df['type'] = 'Normal'

        anomaly_error_df_copy = anomaly_df[['sensor', 'error']].copy()
        anomaly_error_df_copy['type'] = 'Sismo 6.1'

        comparison_df = pd.concat([normal_error_df, anomaly_error_df_copy], ignore_index=True)

        plt.figure(figsize=(14, 8))
        sns.boxplot(x='sensor', y='error', hue='type', data=comparison_df, showfliers=False)
        plt.title('Error Comparison by Sensor: Normal vs. Anomaly (Sismo)')
        plt.xlabel('Sensor ID')
        plt.ylabel('Reconstruction Error (MSE)')
        plt.yscale('log')  # Escala logarítmica para ver mejor la gran diferencia
        plt.grid(True, which="both")
        plt.legend()
        plt.show()

        # --- NUEVO: Gráfico 2x2 para datos del SISMO ---
        fig_sismo, axes_sismo = plt.subplots(2, 2, figsize=(15, 12))
        fig_sismo.suptitle('Evaluation Results of Sismo 6.1 Samples by Sensor', fontsize=16)

        sismo_error_upper = anomaly_df['error'].quantile(0.98)
        sismo_ssim_lower = anomaly_df['ssim'].quantile(0.02)
        sismo_error_display_max = sismo_error_upper * 1.1
        sismo_ssim_display_min = ssim_lower_bound * 0.98

        current_x_offset_sismo = 0
        for sensor_id in unique_sensors:
            sensor_df_sismo = anomaly_df[anomaly_df['sensor'] == sensor_id]
            n_samples = len(sensor_df_sismo)
            if n_samples == 0: continue
            x_values = np.arange(current_x_offset_sismo, current_x_offset_sismo + n_samples)
            color = color_map.get(sensor_id)  # Usar el mismo color que en el gráfico normal
            axes_sismo[0, 0].plot(x_values, sensor_df_sismo['error'], marker='o', linestyle='None', color=color,
                                  alpha=0.5, markersize=3)
            axes_sismo[0, 1].plot(x_values, sensor_df_sismo['ssim'], marker='o', linestyle='None', color=color,
                                  alpha=0.5, markersize=3)
            current_x_offset_sismo += n_samples

        axes_sismo[0, 0].set_title('(a) Error vs Sample Index (Sismo)')
        axes_sismo[0, 0].set_xlabel('The number of samples');
        axes_sismo[0, 0].set_ylabel('Error (MSE)')
        axes_sismo[0, 0].grid(True);
        axes_sismo[0, 0].set_ylim(0, sismo_error_display_max)

        axes_sismo[0, 1].set_title('(b) SSIM vs Sample Index (Sismo)')
        axes_sismo[0, 1].set_xlabel('The number of samples');
        axes_sismo[0, 1].set_ylabel('SSIM')
        axes_sismo[0, 1].grid(True);
        axes_sismo[0, 1].set_ylim(sismo_ssim_display_min, 1.01)

        sns.boxplot(ax=axes_sismo[1, 0], x='sensor', y='error', data=anomaly_df, palette=color_map, hue='sensor',
                    legend=False, showfliers=False)
        axes_sismo[1, 0].set_title('(c) Error by Sensor (Sismo)');
        axes_sismo[1, 0].set_xlabel('Sensors');
        axes_sismo[1, 0].set_ylabel('Error (MSE)')
        axes_sismo[1, 0].grid(True);
        axes_sismo[1, 0].set_ylim(0, sismo_error_display_max)

        sns.boxplot(ax=axes_sismo[1, 1], x='sensor', y='ssim', data=anomaly_df, palette=color_map, hue='sensor',
                    legend=False, showfliers=False)
        axes_sismo[1, 1].set_title('(d) SSIM by Sensor (Sismo)');
        axes_sismo[1, 1].set_xlabel('Sensors');
        axes_sismo[1, 1].set_ylabel('SSIM')
        axes_sismo[1, 1].grid(True);
        axes_sismo[1, 1].set_ylim(sismo_ssim_display_min, 1.01)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        plt.show()

        # Visualización de Muestras Anómalas
        if anomaly_data_samples:
            fig, axes = plt.subplots(len(anomaly_data_samples), 2, figsize=(14, 4 * len(anomaly_data_samples)),
                                     gridspec_kw={'width_ratios': [3, 1]})
            if len(anomaly_data_samples) == 1: axes = np.array([axes])
            fig.suptitle('Anomaly (Sismo) Samples: Original vs. Reconstructed & Error Signal', fontsize=16)

            for i, sample in enumerate(anomaly_data_samples):
                original_sample_scaled = sample.unsqueeze(0).to(device)
                reconstructed_sample_scaled = model(original_sample_scaled)
                original_sample_cpu = original_sample_scaled.squeeze().cpu().numpy()
                reconstructed_sample_cpu = reconstructed_sample_scaled.squeeze().cpu().detach().numpy()
                original_sample_inv = scaler.inverse_transform(original_sample_cpu.reshape(-1, 1))
                reconstructed_sample_inv = scaler.inverse_transform(reconstructed_sample_cpu.reshape(-1, 1))
                error_signal = original_sample_inv - reconstructed_sample_inv
                ax1 = axes[i, 0]
                ax1.plot(original_sample_inv, label='Original (Sismo)')
                ax1.plot(reconstructed_sample_inv, label='Reconstructed', linestyle='--')
                ax1.set_title(f'Anomaly Sample {i + 1}')
                ax1.set_xlabel('Time Step');
                ax1.set_ylabel('Original Value')
                ax1.legend();
                ax1.grid(True)
                ax2 = axes[i, 1]
                ax2.plot(error_signal, color='crimson')
                ax2.set_title('Reconstruction Error')
                ax2.set_xlabel('Time Step');
                ax2.set_ylabel('Error')
                ax2.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
else:
    print(f"\nAnomaly data directory not found, skipping anomaly evaluation: {ANOMALY_DATA_DIR}")

# --- Visualización de Muestras Individuales (Normal) ---
print("\n--- Visualizing Individual Sample Reconstructions (Normal Data) ---")
if len(test_dataset) > 0:
    num_samples_to_plot = 3
    random_indices = np.random.choice(len(test_dataset), num_samples_to_plot, replace=False)
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(14, 4 * num_samples_to_plot),
                                 gridspec_kw={'width_ratios': [3, 1]})
        if num_samples_to_plot == 1: axes = np.array([axes])

        fig.suptitle('Original vs Reconstructed Samples from Normal Test Set & Error Signal', fontsize=16)

        for i, idx in enumerate(random_indices):
            original_sample_scaled = test_dataset[idx].unsqueeze(0).to(device)
            reconstructed_sample_scaled = model(original_sample_scaled)
            original_sample_cpu = original_sample_scaled.squeeze().cpu().numpy()
            reconstructed_sample_cpu = reconstructed_sample_scaled.squeeze().cpu().detach().numpy()
            original_sample_inv = scaler.inverse_transform(original_sample_cpu.reshape(-1, 1))
            reconstructed_sample_inv = scaler.inverse_transform(reconstructed_sample_cpu.reshape(-1, 1))

            error_signal = original_sample_inv - reconstructed_sample_inv

            ax1 = axes[i, 0]
            ax1.plot(original_sample_inv, label='Original')
            ax1.plot(reconstructed_sample_inv, label='Reconstructed', linestyle='--')
            ax1.set_title(f'Normal Sample (Index: {idx})')
            ax1.set_xlabel('Time Step');
            ax1.set_ylabel('Original Value')
            ax1.legend();
            ax1.grid(True)
            ax2 = axes[i, 1]
            ax2.plot(error_signal, color='crimson')
            ax2.set_title('Reconstruction Error')
            ax2.set_xlabel('Time Step');
            ax2.set_ylabel('Error')
            ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
else:
    print("Test dataset is empty, skipping individual sample visualization.")

