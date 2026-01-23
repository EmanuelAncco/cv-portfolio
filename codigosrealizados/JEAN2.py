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
# Se ha actualizado la ruta para apuntar a tu directorio de datos específico.
DATA_DIR = r'D:\Python_proyectos_2025\JEAN SISMOS\DATA'
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
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 100
best_loss = float('inf')
patience = 10;
counter = 0
final_num_epoch = num_epochs
train_losses, test_losses = [], []

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
    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            targets = inputs.clone()
            outputs = model(inputs)
            test_loss = criterion(outputs, targets)
            running_test_loss += test_loss.item() * inputs.size(0)
    epoch_test_loss = running_test_loss / len(test_dataset)
    test_losses.append(epoch_test_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4e}, Test Loss: {epoch_test_loss:.4e}')
    if epoch_test_loss < best_loss:
        best_loss = epoch_test_loss;
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Guardar el mejor modelo
    else:
        counter += 1
    if counter >= patience:
        print(f'Early stopping triggered at epoch {epoch + 1}.')
        final_num_epoch = epoch + 1
        break
print("Training finished.")

# Cargar el mejor modelo guardado para la evaluación final
model.load_state_dict(torch.load('best_model.pth'))
print("\nLoaded best model for final evaluation.")

# --- Results Visualization (Loss Curve) ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, final_num_epoch + 1), train_losses, label='Train Loss')
plt.plot(range(1, final_num_epoch + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.title('Training and Test Loss per Epoch')
plt.legend();
plt.grid(True);
plt.show()

# --- Scenario-Based Evaluation and Plotting ---
print("\n--- Starting Scenario-Based Evaluation ---")
model.eval()
# Almacenar resultados para cada escenario (ahora un 'escenario' es un sensor)
sensor_results = defaultdict(lambda: {'errors': [], 'ssims': []})

mse_loss = nn.MSELoss()

with torch.no_grad():
    # Iterar a través de cada archivo de datos, identificando el sensor por el nombre del archivo
    for filename, data_file_content in zip(data_files, all_data):
        try:
            # Obtener el ID del sensor del primer caracter del nombre del archivo
            sensor_id = int(filename[0])
        except (ValueError, IndexError):
            print(f"Could not determine sensor ID for file {filename}. Skipping.")
            continue

        print(f"Evaluating file {filename} for Sensor {sensor_id}...")

        # Preparar los datos para este archivo específico
        scenario_data = scaler.transform(data_file_content[:, 1:2])
        scenario_dataset = SlidingWindowDataset1D(scenario_data, window_size, stride)
        scenario_dataloader = DataLoader(scenario_dataset, batch_size=batch_size, shuffle=False)

        if len(scenario_dataset) == 0:
            continue

        for original_windows in scenario_dataloader:
            original_windows = original_windows.to(device)
            reconstructed_windows = model(original_windows)

            for i in range(original_windows.size(0)):
                original = original_windows[i].squeeze().cpu().numpy()
                reconstructed = reconstructed_windows[i].squeeze().cpu().numpy()

                # Calcular Error (MSE)
                error = mse_loss(torch.tensor(reconstructed), torch.tensor(original)).item()
                sensor_results[sensor_id]['errors'].append(error)

                # Calcular SSIM
                data_range = np.max(original) - np.min(original)
                if data_range == 0: data_range = 1
                similarity = ssim(original.reshape(8, 16), reconstructed.reshape(8, 16), data_range=data_range)
                sensor_results[sensor_id]['ssims'].append(similarity)

# --- Creación de los gráficos de análisis ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Evaluation Results of All Testing Samples by Sensor', fontsize=16)

# Determinar el número de sensores únicos para los colores
unique_sensors = sorted(sensor_results.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sensors)))
color_map = {sensor_id: color for sensor_id, color in zip(unique_sensors, colors)}

# Gráficos de dispersión (a) y (b)
all_errors_flat, all_ssims_flat, all_colors_flat = [], [], []

for sensor_id in unique_sensors:
    results = sensor_results[sensor_id]
    num_samples_in_sensor = len(results['errors'])
    all_errors_flat.extend(results['errors'])
    all_ssims_flat.extend(results['ssims'])
    all_colors_flat.extend([color_map[sensor_id]] * num_samples_in_sensor)

x_axis_samples = range(len(all_errors_flat))

# (a) Error vs The number of samples
axes[0, 0].scatter(x_axis_samples, all_errors_flat, c=all_colors_flat, alpha=0.7, s=20)
axes[0, 0].set_title('(a) Error vs Sample Index')
axes[0, 0].set_xlabel('The number of samples')
axes[0, 0].set_ylabel('Error (MSE)')
axes[0, 0].grid(True)

# (b) SSIM vs The number of samples
axes[0, 1].scatter(x_axis_samples, all_ssims_flat, c=all_colors_flat, alpha=0.7, s=20)
axes[0, 1].set_title('(b) SSIM vs Sample Index')
axes[0, 1].set_xlabel('The number of samples')
axes[0, 1].set_ylabel('SSIM')
axes[0, 1].grid(True)

# Listas para los diagramas de caja
error_data_for_boxplot = [sensor_results[sid]['errors'] for sid in unique_sensors]
ssim_data_for_boxplot = [sensor_results[sid]['ssims'] for sid in unique_sensors]
sensor_labels = [str(sid) for sid in unique_sensors]

# (c) Box plot de Error por Escenario/Sensor
axes[1, 0].boxplot(error_data_for_boxplot, labels=sensor_labels, patch_artist=True)
axes[1, 0].set_title('(c) Error by Sensor')
axes[1, 0].set_xlabel('Sensors')
axes[1, 0].set_ylabel('Error (MSE)')
axes[1, 0].grid(True)

# (d) Box plot de SSIM por Escenario/Sensor
axes[1, 1].boxplot(ssim_data_for_boxplot, labels=sensor_labels, patch_artist=True)
axes[1, 1].set_title('(d) SSIM by Sensor')
axes[1, 1].set_xlabel('Sensors')
axes[1, 1].set_ylabel('SSIM')
axes[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
