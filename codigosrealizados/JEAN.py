# -*- coding: utf-8 -*-
"""
1D Signal Processor and Autoencoder Trainer for Local Execution

Instructions for local setup:
1.  Save this script as a Python file (e.g., `train_model.py`).
2.  Create a folder named 'data' in the same directory as this script.
3.  Place all your '.txt' data files inside the 'data' folder.
4.  Install the required libraries by running the following command in your terminal:
    pip install numpy pandas matplotlib seaborn torch scikit-learn
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
data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
if not data_files:
    print(f"Error: No '.txt' files found in the '{DATA_DIR}' directory.")
    exit()

print(f"Found {len(data_files)} data file(s): {data_files}")

data = []
for file_i in data_files:
    file_path = os.path.join(DATA_DIR, file_i)
    try:
        data.append(np.loadtxt(file_path))
    except Exception as e:
        print(f"Could not read file {file_i}: {e}")

# Note from original code: "Como cada txt file tiene diferentes intervalos de tiempo,
# no puedo juntarlos a menos que haga interpolacion, por lo que el siguiente
# codigo solo utilizara el primer txt."
# We will proceed using only the first loaded dataset.
if not data:
    print("Error: No data was successfully loaded.")
    exit()

data_i = data[0]

# --- Preprocessing ---
# Scaler
scaler = StandardScaler()
# Reshape for scaler which expects 2D array, and select only the second column (accelerations)
data_i_scaled = scaler.fit_transform(data_i[:, 1].reshape(-1, 1))


# --- Dataset and DataLoader ---
class SlidingWindowDataset1D(Dataset):
    """
    A PyTorch Dataset for creating sliding windows from a 1D signal.
    """

    def __init__(self, data, window_size, stride=1, transform=None):
        """
        Args:
            data (torch.Tensor or np.ndarray): The input 1D signal.
                                                Expected shape: [total_length, channels] or [total_length]
                                                If [total_length], it will be unsqueezed to [total_length, 1].
            window_size (int): The length of each window.
            stride (int): How many steps to move the window for the next sample.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
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
            raise ValueError(f"Not enough data to form a single window with "
                             f"length {len(self.data)}, window_size {window_size}, "
                             f"and stride {stride}.")

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
total_length = len(data_i_scaled)
window_size = 128
stride = 64

# Create the dataset
dataset = SlidingWindowDataset1D(data_i_scaled, window_size, stride)

# Split into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
batch_size = 100
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n--- DataLoader Info ---")
print(f"Total windows: {len(dataset)}")
print(f"Training windows: {len(train_dataset)}")
print(f"Test windows: {len(test_dataset)}")
print(f"Number of training batches: {len(train_dataloader)}")
print(f"Number of test batches: {len(test_dataloader)}")
print("-----------------------\n")


# --- Model Definition ---

class SelfAttention1D(nn.Module):
    """
    Self-attention module for 1D signals.
    Input shape: [batch_size, length, channels]
    Output shape: [batch_size, length, channels]
    """

    def __init__(self, channels):
        super(SelfAttention1D, self).__init__()
        self.channels = channels
        self.query_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.key_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x):
        x = x.permute(0, 2, 1)
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        query = query.permute(0, 2, 1)
        key = key.permute(0, 2, 1)
        value = value.permute(0, 2, 1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output


class ModelWithSkipConnections(nn.Module):
    def __init__(self, input_channels, base_channels, input_sequence_length,
                 internal_channels_list=None, output_channels_linear=1):
        super(ModelWithSkipConnections, self).__init__()

        # *** CORRECTED THIS LINE: 'is NoEarly Stoppingne' to 'is None' ***
        if internal_channels_list is None:
            internal_channels_list = [max(1, base_channels // (2 ** i)) for i in range(1, 4)]

        if not internal_channels_list or len(internal_channels_list) < 1:
            raise ValueError("internal_channels_list must contain at least one channel size for downsampling.")

        self.input_channels = input_channels
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
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.1)
            ))
            current_in_channels = out_c

        self.encoder_blocks = encoder_blocks
        self.encoder_deepest_channels = current_in_channels

        self.bottleneck = nn.Sequential(
            nn.Conv1d(self.encoder_deepest_channels, self.encoder_deepest_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.encoder_deepest_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.encoder_deepest_channels * 2, self.encoder_deepest_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.encoder_deepest_channels),
            nn.LeakyReLU(0.1)
        )

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
                nn.ConvTranspose1d(
                    current_decoder_in_channels,
                    out_channels_decoder_block,
                    kernel_size=3, padding=1, output_padding=1, stride=2
                ),
                nn.BatchNorm1d(out_channels_decoder_block),
                nn.LeakyReLU(0.1),
                nn.Conv1d(out_channels_decoder_block + skip_channels, out_channels_decoder_block, kernel_size=3,
                          padding=1),
                nn.BatchNorm1d(out_channels_decoder_block),
                nn.LeakyReLU(0.1)
            ))
            current_decoder_in_channels = out_channels_decoder_block

        self.decoder_blocks = decoder_blocks
        self.fc_final_output = nn.Linear(base_channels, output_channels_linear)
        self.final_activation = nn.Identity()
        self.expected_decoder_output_length = (self.input_sequence_length // (2 ** self.num_downsample_blocks)) * (
                    2 ** self.num_downsample_blocks)
        self.fc_length_adjust = nn.Linear(
            self.base_channels * self.expected_decoder_output_length,
            self.base_channels * self.input_sequence_length
        )

    def forward(self, x):
        original_length = x.shape[1]
        if original_length != self.input_sequence_length:
            raise ValueError(f"Input sequence length {original_length} does not match "
                             f"the expected length {self.input_sequence_length}.")
        batch_size = x.shape[0]
        x_proj = x.permute(0, 2, 1)
        x_proj = self.initial_projection(x_proj)
        x_proj = x_proj.permute(0, 2, 1)
        x_sa = self.self_attention(x_proj)
        x_enc = x_sa.permute(0, 2, 1)

        skip_connections = []
        skip_connections.append(x_enc)
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
    input_channels=1,
    base_channels=64,
    input_sequence_length=window_size,
    internal_channels_list=[64, 128, 256],
    output_channels_linear=1
)

# Test model with a dummy input
try:
    input_tensor = torch.randn(batch_size, window_size, 1)
    output = model(input_tensor)
    print(f"Model initialized successfully. Output shape: {output.shape}\n")
except Exception as e:
    print(f"Error during model initialization: {e}")
    exit()

# --- Training Loop ---
# Device configuration (checks for GPU, falls back to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training parameters
num_epochs = 100
best_loss = float('inf')
patience = 10
counter = 0
final_num_epoch = num_epochs

# History tracking
train_losses = []
test_losses = []

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
        best_loss = epoch_test_loss
        counter = 0
        # Optional: Save the best model
        # torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1

    if counter >= patience:
        print(f'Early stopping triggered at epoch {epoch + 1}.')
        final_num_epoch = epoch + 1
        break

print("Training finished.")

# --- Results Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, final_num_epoch + 1), train_losses, label='Train Loss')
plt.plot(range(1, final_num_epoch + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss per Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Select 3 random indices from the test set for plotting
if len(test_dataset) > 0:
    num_samples_to_plot = min(3, len(test_dataset))
    random_indices = np.random.choice(len(test_dataset), num_samples_to_plot, replace=False)

    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(12, 4 * num_samples_to_plot), squeeze=False)
        fig.suptitle('Original vs Reconstructed Samples from Test Set', fontsize=16)

        for i, idx in enumerate(random_indices):
            original_sample = test_dataset[idx].unsqueeze(0).to(device)
            reconstructed_sample = model(original_sample)

            original_sample_cpu = original_sample.squeeze(0).squeeze(-1).cpu().numpy()
            reconstructed_sample_cpu = reconstructed_sample.squeeze(0).squeeze(-1).cpu().numpy()

            # Inverse transform to see the original scale
            original_sample_inv = scaler.inverse_transform(original_sample_cpu.reshape(-1, 1))
            reconstructed_sample_inv = scaler.inverse_transform(reconstructed_sample_cpu.reshape(-1, 1))

            ax = axes[i, 0]
            ax.plot(original_sample_inv, label='Original')
            ax.plot(reconstructed_sample_inv, label='Reconstructed', linestyle='--')
            ax.set_title(f'Test Sample Index: {idx}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Original Scale Value')
            ax.legend()
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
else:
    print("Test dataset is empty, skipping sample visualization.")
