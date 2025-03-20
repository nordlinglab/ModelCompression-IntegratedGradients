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
# Assume teacher training accuracy is 100% (you may want to adjust this)
teacher_train_accuracy = 100.0
speedup_values = [1.0, 10.6, 11.1, 15.0, 17.1, 20.6, 25.27, 35.71, 103.5]

# Normalize test accuracy relative to the teacher model (teacher = 100%)
normalized_test_acc = {}
for model in MODEL_TYPES:
    # Add the teacher accuracy as the first point (100%)
    normalized_test_acc[model] = [100.0]  # Teacher is 100%

    # Add the normalized values for all other compression factors
    for i, mean in enumerate(stats[model]["test_mean"]):
        if not np.isnan(mean):
            norm_acc = (mean / teacher_accuracy) * 100
            normalized_test_acc[model].append(norm_acc)
        else:
            normalized_test_acc[model].append(np.nan)

# Normalize training accuracy relative to the teacher model (teacher = 100%)
normalized_train_acc = {}
for model in MODEL_TYPES:
    # Add the teacher accuracy as the first point (100%)
    normalized_train_acc[model] = [100.0]  # Teacher is 100%

    # Add the normalized values for all other compression factors
    for i, mean in enumerate(stats[model]["train_mean"]):
        if not np.isnan(mean):
            norm_acc = (mean / teacher_train_accuracy) * 100
            normalized_train_acc[model].append(norm_acc)
        else:
            normalized_train_acc[model].append(np.nan)

# Define color map
color_map = {"Student": "blue", "KD": "orange", "KD_IG": "green", "IG": "red"}

# Main figure setup
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["lines.linewidth"] = 2

# Create figure with subplots - left for training, right for testing
fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(24, 12), sharey=True)

# Create twin axes for speedup in both subplots
ax_train_speedup = ax_train.twinx()
ax_test_speedup = ax_test.twinx()

# Add teacher point with star marker to both plots
ax_train.scatter(
    compression_factors[0], 100, marker="*", color="black", s=400, label="Teacher"
)
ax_test.scatter(
    compression_factors[0], 100, marker="*", color="black", s=400, label="Teacher"
)

# Plotting on left subplot (Training Accuracy)
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    # Plot training accuracy
    ax_train.plot(
        compression_factors[1:],
        normalized_train_acc[model][1:],
        marker="o",
        color=color,
        label=display_name,
    )

    if model == "KD_IG":
        # Calculate upper and lower bounds for the shaded region (train)
        upper_bound_train = []
        lower_bound_train = []

        for i, cf in enumerate(COMPRESSION_FACTORS):
            mean = stats[model]["train_mean"][i]
            std = stats[model]["train_std"][i]
            if not np.isnan(mean) and not np.isnan(std):
                upper = ((mean + std) / teacher_train_accuracy) * 100
                lower = ((mean - std) / teacher_train_accuracy) * 100
                upper_bound_train.append(upper)
                lower_bound_train.append(lower)
            else:
                upper_bound_train.append(np.nan)
                lower_bound_train.append(np.nan)

        ax_train.fill_between(
            compression_factors[1:],
            lower_bound_train,
            upper_bound_train,
            alpha=0.3,
            color=color,
        )

# Plotting on right subplot (Testing Accuracy)
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    # Plot test accuracy
    ax_test.plot(
        compression_factors[1:],
        normalized_test_acc[model][1:],
        marker="o",
        color=color,
        label=display_name,
    )

    if model == "KD_IG":
        # Calculate upper and lower bounds for the shaded region (test)
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

        ax_test.fill_between(
            compression_factors[1:], lower_bound, upper_bound, alpha=0.3, color=color
        )

# Plotting speedup on both subplots (use dashed black line)
ax_train_speedup.plot(
    compression_factors[1:],
    speedup_values[1:],
    marker="s",
    color="black",
    linestyle="--",
    label="Speed up",
)
ax_train_speedup.scatter(
    compression_factors[0], speedup_values[0], marker="*", color="black", s=400
)

ax_test_speedup.plot(
    compression_factors[1:],
    speedup_values[1:],
    marker="s",
    color="black",
    linestyle="--",
    label="Speed up",
)
ax_test_speedup.scatter(
    compression_factors[0], speedup_values[0], marker="*", color="black", s=400
)

# Configure axes
for ax in [ax_train, ax_test]:
    ax.set_xscale("log")
    ax.set_xlabel("Compression Factor (a.u.)")
    ax.set_ylim(0, 105)  # Adjusted to accommodate 100% for teacher
    ax.grid(True, which="both", axis="both", alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# Set specific y-axis labels
ax_train.set_ylabel("Training Accuracy vs. Teacher Model (%)")
# No y-label for ax_test since it shares y-axis with ax_train

# Configure speedup axes
for ax in [ax_train_speedup, ax_test_speedup]:
    ax.set_ylabel("Speed up vs. Teacher Model (a.u.)")
    ax.set_ylim(0, 110)  # Adjusted based on speedup values
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# Remove inner y-axis of right subplot
ax_test_speedup.set_ylabel("")
ax_test_speedup.get_yaxis().set_visible(False)

# Set titles
ax_train.set_title("Training Accuracy")
ax_test.set_title("Testing Accuracy")

# Add common legend
lines_train, labels_train = ax_train.get_legend_handles_labels()
lines_speedup, labels_speedup = ax_train_speedup.get_legend_handles_labels()

# Create a combined legend
fig.legend(
    lines_train + lines_speedup,
    labels_train + labels_speedup,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.03),
    fancybox=True,
    ncol=5,
)

# Add insets for detailed view of compression factors 2.19-12.04
inset_ax_train = plt.axes([0.22, 0.3, 0.15, 0.25])
inset_ax_test = plt.axes([0.72, 0.3, 0.15, 0.25])

# Twin axes for speedup in insets
inset_ax_train_speedup = inset_ax_train.twinx()
inset_ax_test_speedup = inset_ax_test.twinx()

# Extract data for inset (first 5 compression factors)
inset_indices = range(1, 5)  # Indices 1-4 (CF 2.19 to 12.04)
cf_mini = [compression_factors[i] for i in inset_indices]
speedup_mini = [speedup_values[i] for i in inset_indices]

# Plot data in insets
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    # Plot training accuracy in left inset
    train_acc_mini = [normalized_train_acc[model][i] for i in inset_indices]
    inset_ax_train.plot(cf_mini, train_acc_mini, marker="o", color=color)

    # Plot testing accuracy in right inset
    test_acc_mini = [normalized_test_acc[model][i] for i in inset_indices]
    inset_ax_test.plot(cf_mini, test_acc_mini, marker="o", color=color)

    if model == "KD_IG":
        # Training inset shaded region
        upper_bound_train = []
        lower_bound_train = []

        for i in range(1, 5):  # CF 2.19 to 12.04
            mean = stats[model]["train_mean"][i - 1]  # Adjust index
            std = stats[model]["train_std"][i - 1]
            if not np.isnan(mean) and not np.isnan(std):
                upper = ((mean + std) / teacher_train_accuracy) * 100
                lower = ((mean - std) / teacher_train_accuracy) * 100
                upper_bound_train.append(upper)
                lower_bound_train.append(lower)
            else:
                upper_bound_train.append(np.nan)
                lower_bound_train.append(np.nan)

        inset_ax_train.fill_between(
            cf_mini, lower_bound_train, upper_bound_train, alpha=0.3, color=color
        )

        # Testing inset shaded region
        upper_bound_test = []
        lower_bound_test = []

        for i in range(1, 5):  # CF 2.19 to 12.04
            mean = stats[model]["test_mean"][i - 1]  # Adjust index
            std = stats[model]["test_std"][i - 1]
            if not np.isnan(mean) and not np.isnan(std):
                upper = ((mean + std) / teacher_accuracy) * 100
                lower = ((mean - std) / teacher_accuracy) * 100
                upper_bound_test.append(upper)
                lower_bound_test.append(lower)
            else:
                upper_bound_test.append(np.nan)
                lower_bound_test.append(np.nan)

        inset_ax_test.fill_between(
            cf_mini, lower_bound_test, upper_bound_test, alpha=0.3, color=color
        )

# Add speedup line to insets
inset_ax_train_speedup.plot(
    cf_mini, speedup_mini, marker="s", color="black", linestyle="--"
)
inset_ax_test_speedup.plot(
    cf_mini, speedup_mini, marker="s", color="black", linestyle="--"
)

# Configure inset axes
for ax in [inset_ax_train, inset_ax_test]:
    ax.grid(True)
    ax.xaxis.set_major_formatter(ScalarFormatter())

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend

# Save and show the plot
# plt.savefig("Compression_speedup_subplots.pdf")
plt.show()
