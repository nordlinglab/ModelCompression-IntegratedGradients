import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Define constants
BASE_DIR = "./../GPUS"
GPU_SETUPS = ["A5000", "3090", "3060 Ti"]
MODEL_TYPE = "KD_IG"  # Focus only on KD_IG model
COMPRESSION_FACTORS = [
    "2.19",
    "4.12",
    "7.29",
    "12.04",
    "28.97",
    "54.59",
    "139.43",
    "1121.71",
]


# Function to convert string compression factors to float for proper sorting
def cf_to_float(cf_str):
    return float(cf_str)


# Initialize dictionaries to store results
training_times = {gpu: [] for gpu in GPU_SETUPS}
inference_times = {gpu: [] for gpu in GPU_SETUPS}

# Process all files
for gpu in GPU_SETUPS:
    for cf in COMPRESSION_FACTORS:
        file_path = os.path.join(
            BASE_DIR, gpu, f"compression_time_acc/{MODEL_TYPE}_{cf}_1.csv"
        )

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Calculate mean epoch time and convert to ms per image
            # Using batch size of 64 as specified
            mean_epoch_time = (
                df["Epoch Time (s)"].mean() * 1000
            ) / 64  # Convert to ms per image

            # Calculate mean inference time and convert to ms per image
            mean_inference_time = (
                df["Inference Time (s)"].mean() * 1000
            ) / 64  # Convert to ms per image

            # Store results
            training_times[gpu].append((cf_to_float(cf), mean_epoch_time))
            inference_times[gpu].append((cf_to_float(cf), mean_inference_time))

        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Sort results by compression factor
for gpu in GPU_SETUPS:
    training_times[gpu].sort(key=lambda x: x[0])
    inference_times[gpu].sort(key=lambda x: x[0])

# Set the style
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

# sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Create a figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()  # Create second y-axis that shares the same x-axis

# Define colors for GPUs using ColorBrewer qualitative palette
gpu_colors = sns.color_palette(
    "Set1", len(GPU_SETUPS)
)  # Set1 is a ColorBrewer qualitative palette
markers = ["o", "s", "^"]  # One marker per GPU

# Legend handles
handles1 = []
handles2 = []
labels = []

# Plot training times on the left y-axis
for i, gpu in enumerate(GPU_SETUPS):
    data = training_times[gpu]
    if data:
        x_values = [x[0] for x in data]
        y_values = [y[1] for y in data]
        (line,) = ax1.plot(
            x_values,
            y_values,
            marker=markers[i],
            label=f"{gpu} - Training",
            color=gpu_colors[i],
            linewidth=2,
            markersize=6,
            linestyle="-",  # Solid line for training
        )
        handles1.append(line)
        labels.append(f"{gpu} - Training")

# Plot inference times on the right y-axis
for i, gpu in enumerate(GPU_SETUPS):
    data = inference_times[gpu]
    if data:
        x_values = [x[0] for x in data]
        y_values = [y[1] for y in data]
        (line,) = ax2.plot(
            x_values,
            y_values,
            marker=markers[i],
            label=f"{gpu} - Inference",
            color=gpu_colors[i],
            linewidth=2,
            markersize=6,
            linestyle="--",  # Dashed line for inference
        )
        handles2.append(line)
        labels.append(f"{gpu} - Inference")

# Combine the legends from both axes
all_handles = handles1 + handles2
ax1.legend(handles=all_handles, labels=labels, loc="best")

# Set labels and title
ax1.set_xlabel("Compression Factor (log scale)")
ax1.set_ylabel("Training Time (ms/image)")
ax2.set_ylabel("Inference Time (ms/image)")
ax1.set_xscale("log")  # Log scale for compression factor
ax1.grid(True)

# Save the figure
plt.savefig("gpu_performance_comparison.pdf", dpi=300, bbox_inches="tight")
