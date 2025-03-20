import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

# Define constants
GPU_FOLDERS = ["A5000", "3090", "3060 Ti"]
MODEL_TYPES = ["Student", "KD", "IG", "KD_IG"]  # Reordered to make KD_IG last
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
BASE_DIR = "./../GPUS"  # Updated base directory

# Dictionary to store results
results = {
    model: {cf: {"train": [], "test": []} for cf in COMPRESSION_FACTORS}
    for model in MODEL_TYPES
}


# Function to extract final accuracy from CSV
def extract_final_accuracy(filepath):
    try:
        df = pd.read_csv(filepath)
        if "Training Accuracy" in df.columns and "Testing Accuracy" in df.columns:
            # Get the last row's training and testing accuracy
            last_row = df.iloc[-1]
            return last_row["Training Accuracy"], last_row["Testing Accuracy"]
        else:
            print(f"Warning: Columns not found in {filepath}")
            return None, None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


# Process files for each model type and compression factor
for model in MODEL_TYPES:
    for cf in COMPRESSION_FACTORS:
        # Determine how many files to look for based on compression factor
        max_file_num = 8 if float(cf) <= 12.04 else 1

        for gpu in GPU_FOLDERS:
            for file_num in range(1, max_file_num + 1):
                # Updated filepath to match the requested structure
                filepath = os.path.join(
                    BASE_DIR, gpu, f"compression_time_acc/{model}_{cf}_{file_num}.csv"
                )

                if os.path.exists(filepath):
                    train_acc, test_acc = extract_final_accuracy(filepath)
                    if train_acc is not None and test_acc is not None:
                        results[model][cf]["train"].append(train_acc)
                        results[model][cf]["test"].append(test_acc)

# Calculate means and standard deviations
stats = {
    model: {
        "cf": COMPRESSION_FACTORS,
        "train_mean": [],
        "train_std": [],
        "test_mean": [],
        "test_std": [],
    }
    for model in MODEL_TYPES
}

for model in MODEL_TYPES:
    for cf in COMPRESSION_FACTORS:
        train_values = results[model][cf]["train"]
        test_values = results[model][cf]["test"]

        if train_values:
            stats[model]["train_mean"].append(np.mean(train_values))
            stats[model]["train_std"].append(np.std(train_values))
        else:
            stats[model]["train_mean"].append(np.nan)
            stats[model]["train_std"].append(np.nan)

        if test_values:
            stats[model]["test_mean"].append(np.mean(test_values))
            stats[model]["test_std"].append(np.std(test_values))
        else:
            stats[model]["test_mean"].append(np.nan)
            stats[model]["test_std"].append(np.nan)

# Use the provided exact values
compression_factors = [1.0, 2.19, 4.12, 7.29, 12.04, 28.97, 54.59, 139.43, 1121.71]
teacher_accuracy = 93.91
speedup_values = [1.0, 10.6, 11.1, 15.0, 17.1, 20.6, 25.27, 35.71, 103.5]

# Normalize test accuracy relative to the teacher model (teacher = 100%)
normalized_acc = {}
for model in MODEL_TYPES:
    # Add the teacher accuracy as the first point (100%)
    normalized_acc[model] = [100.0]  # Teacher is 100%

    # Add the normalized values for all other compression factors
    for i, mean in enumerate(stats[model]["test_mean"]):
        if not np.isnan(mean):
            norm_acc = (mean / teacher_accuracy) * 100
            normalized_acc[model].append(norm_acc)
        else:
            normalized_acc[model].append(np.nan)

# Define color map
color_map = {"Student": "blue", "KD": "orange", "KD_IG": "green", "IG": "red"}

# Main figure setup
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["lines.linewidth"] = 2

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(24, 15))
ax2 = ax1.twinx()

# Add teacher point with star marker
ax1.scatter(
    compression_factors[0], 100, marker="*", color="black", s=400, label="Teacher"
)

# Plotting on left y-axis (Accuracy)
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    if model == "KD_IG":
        # Plot KD_IG with line and shaded region for std
        ax1.plot(
            compression_factors[1:],
            normalized_acc[model][1:],
            marker="o",
            color=color,
            label=display_name,
        )

        # Calculate upper and lower bounds for the shaded region
        upper_bound = []
        lower_bound = []

        for i, cf in enumerate(COMPRESSION_FACTORS):
            mean = stats[model]["test_mean"][i]
            std = stats[model]["test_std"][i]
            if not np.isnan(mean) and not np.isnan(std):
                upper = ((mean + std) / teacher_accuracy) * 100
                lower = ((mean - std) / teacher_accuracy) * 100
                upper_bound.append(upper)
                lower_bound.append(lower)
            else:
                upper_bound.append(np.nan)
                lower_bound.append(np.nan)

        ax1.fill_between(
            compression_factors[1:], lower_bound, upper_bound, alpha=0.3, color=color
        )
    else:
        # Plot other models with just lines
        ax1.plot(
            compression_factors[1:],
            normalized_acc[model][1:],
            marker="o",
            color=color,
            label=display_name,
        )

# Plotting on right y-axis (Speedup)
ax2.plot(
    compression_factors[1:],
    speedup_values[1:],
    marker="s",
    color="black",
    linestyle="--",
    label="Speed up",
)
ax2.scatter(compression_factors[0], speedup_values[0], marker="*", color="black", s=400)

# Configure left y-axis (Accuracy)
ax1.set_xscale("log")
ax1.set_xlabel("Compression Factor (a.u.)")
ax1.set_ylabel("Testing Accuracy vs. Teacher Model (%)")
ax1.set_ylim(0, 105)  # Adjusted to accommodate 100% for teacher
ax1.grid(True, which="both", axis="both", alpha=0.3)

# Configure right y-axis (Speedup)
ax2.set_ylabel("Speed up vs. Teacher Model (a.u.)")
ax2.set_ylim(0, 110)  # Adjusted based on speedup values

# Add minor ticks to both axes
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="lower right",
    # bbox_to_anchor=(0.5, 0.5),
    fancybox=True,
    ncol=3,
)

# Add inset for detailed view of compression factors 2.19-12.04
inset_ax1 = plt.axes([0.1999, 0.3, 0.25, 0.25])
inset_ax2 = inset_ax1.twinx()

# Extract data for inset (first 5 compression factors)
inset_indices = range(1, 5)  # Indices 1-4 (CF 2.19 to 12.04)
cf_mini = [compression_factors[i] for i in inset_indices]
speedup_mini = [speedup_values[i] for i in inset_indices]

# Plot data in inset
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    model_acc_mini = [normalized_acc[model][i] for i in inset_indices]

    if model == "KD_IG":
        # Plot KD_IG with line and markers
        inset_ax1.plot(cf_mini, model_acc_mini, marker="o", color=color)

        # Calculate upper and lower bounds for shaded region
        upper_bound = []
        lower_bound = []

        for i in range(1, 5):  # CF 2.19 to 12.04
            mean = stats[model]["test_mean"][i - 1]  # Adjust index
            std = stats[model]["test_std"][i - 1]
            if not np.isnan(mean) and not np.isnan(std):
                upper = ((mean + std) / teacher_accuracy) * 100
                lower = ((mean - std) / teacher_accuracy) * 100
                upper_bound.append(upper)
                lower_bound.append(lower)
            else:
                upper_bound.append(np.nan)
                lower_bound.append(np.nan)

        inset_ax1.fill_between(
            cf_mini, lower_bound, upper_bound, alpha=0.3, color=color
        )
    else:
        # Plot other models
        inset_ax1.plot(cf_mini, model_acc_mini, marker="o", color=color)

# Add speedup line to inset
inset_ax2.plot(cf_mini, speedup_mini, marker="s", color="black", linestyle="--")

# Configure inset axes
# inset_ax1.set_xscale("log")
inset_ax1.grid(True)
inset_ax1.xaxis.set_major_formatter(ScalarFormatter())

plt.savefig("Compression_speedup.pdf")
plt.show()

# # Print summary statistics
# print("\nSummary Statistics:")
# for model in MODEL_TYPES:
#     display_name = "KD & IG" if model == "KD_IG" else model
#     print(f"\n{display_name} Model:")
#     for i, cf in enumerate(COMPRESSION_FACTORS):
#         print(f"  CF={cf}:")
#         print(
#             f"    Train: {stats[model]['train_mean'][i]:.4f} ± {stats[model]['train_std'][i]:.4f}"
#         )
#         print(
#             f"    Test:  {stats[model]['test_mean'][i]:.4f} ± {stats[model]['test_std'][i]:.4f}"
#         )
