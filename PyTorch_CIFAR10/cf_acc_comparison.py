import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

# Define constants
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
BASE_DIR = "./../Experiment_histories/Compression_acc"  # New base directory

# Dictionary to store results (only testing accuracy)
results = {
    model: {cf: {"test": []} for cf in COMPRESSION_FACTORS} for model in MODEL_TYPES
}


# Function to extract all test accuracies from CSV
def extract_all_accuracies(filepath):
    try:
        df = pd.read_csv(filepath)
        if "Test Accuracy" in df.columns:
            # Get all testing accuracy values and convert to float
            try:
                # Try to convert to numeric, coerce errors to NaN
                test_accuracies = (
                    pd.to_numeric(df["Test Accuracy"], errors="coerce")
                    .dropna()
                    .tolist()
                )

                # Print for debugging
                print(
                    f"File: {filepath}, Values: {test_accuracies[:5]}... (showing up to 5)"
                )

                return test_accuracies
            except Exception as inner_e:
                print(f"Error converting values in {filepath} to numeric: {inner_e}")
                # Return sample of raw values for debugging
                raw_values = df["Testing Accuracy"].head().tolist()
                print(f"Sample raw values: {raw_values}")
                return None
        else:
            print(f"Warning: Testing Accuracy column not found in {filepath}")
            # Print available columns for debugging
            print(f"Available columns: {df.columns.tolist()}")
            return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


# Process files for each model type and compression factor
print("\nReading CSV files:")
for cf in COMPRESSION_FACTORS:
    for model in MODEL_TYPES:
        # Updated filepath to match the new folder structure
        filepath = os.path.join(BASE_DIR, cf, f"{model}_{cf}.csv")
        print(f"Looking for file: {filepath}")

        if os.path.exists(filepath):
            print(f"Found file: {filepath}")
            test_accuracies = extract_all_accuracies(filepath)
            if test_accuracies is not None and len(test_accuracies) > 0:
                results[model][cf]["test"].extend(test_accuracies)
                print(f"Added {len(test_accuracies)} values for {model}, CF={cf}")
            else:
                print(f"No valid accuracy values found in {filepath}")
        else:
            print(f"File not found: {filepath}")

# Print a summary of collected data before processing
print("\nCollected data summary:")
for model in MODEL_TYPES:
    for cf in COMPRESSION_FACTORS:
        values = results[model][cf]["test"]
        print(f"{model}, CF={cf}: {len(values)} values collected")

# Calculate means, mins, maxes, and standard deviations (for test accuracy)
stats = {
    model: {
        "cf": COMPRESSION_FACTORS,
        "test_mean": [],
        "test_min": [],
        "test_max": [],
        "test_std": [],  # Added standard deviation
    }
    for model in MODEL_TYPES
}

for model in MODEL_TYPES:
    for cf in COMPRESSION_FACTORS:
        test_values = results[model][cf]["test"]

        if test_values:
            stats[model]["test_mean"].append(np.mean(test_values))
            stats[model]["test_min"].append(np.min(test_values))
            stats[model]["test_max"].append(np.max(test_values))
            stats[model]["test_std"].append(np.std(test_values))  # Calculate std dev
        else:
            stats[model]["test_mean"].append(np.nan)
            stats[model]["test_min"].append(np.nan)
            stats[model]["test_max"].append(np.nan)
            stats[model]["test_std"].append(np.nan)  # NaN for missing data

# Print maximum test accuracies in table format
print("\n\nMaximum Test Accuracies (%)")
print("-" * 80)
print(f"{'Compression Factor':<20} {'Student':<12} {'KD':<12} {'IG':<12} {'KD_IG':<12}")
print("-" * 80)

for i, cf in enumerate(COMPRESSION_FACTORS):
    student_max = (
        stats["Student"]["test_max"][i]
        if not np.isnan(stats["Student"]["test_max"][i])
        else 0
    )
    kd_max = (
        stats["KD"]["test_max"][i] if not np.isnan(stats["KD"]["test_max"][i]) else 0
    )
    ig_max = (
        stats["IG"]["test_max"][i] if not np.isnan(stats["IG"]["test_max"][i]) else 0
    )
    kd_ig_max = (
        stats["KD_IG"]["test_max"][i]
        if not np.isnan(stats["KD_IG"]["test_max"][i])
        else 0
    )

    # Find the maximum value to highlight
    max_val = max(student_max, kd_max, ig_max, kd_ig_max)

    # Format each value, with bold marker for the maximum
    student_str = (
        f"{student_max:.2f}*" if student_max == max_val else f"{student_max:.2f}"
    )
    kd_str = f"{kd_max:.2f}*" if kd_max == max_val else f"{kd_max:.2f}"
    ig_str = f"{ig_max:.2f}*" if ig_max == max_val else f"{ig_max:.2f}"
    kd_ig_str = f"{kd_ig_max:.2f}*" if kd_ig_max == max_val else f"{kd_ig_max:.2f}"

    print(f"{cf+'x':<20} {student_str:<12} {kd_str:<12} {ig_str:<12} {kd_ig_str:<12}")
print("-" * 80)
print("* indicates the highest accuracy for each compression factor")

# Print standard deviations in table format
print("\n\nStandard Deviations of Test Accuracies (%)")
print("-" * 80)
print(f"{'Compression Factor':<20} {'Student':<12} {'KD':<12} {'IG':<12} {'KD_IG':<12}")
print("-" * 80)

for i, cf in enumerate(COMPRESSION_FACTORS):
    student_std = (
        stats["Student"]["test_std"][i]
        if not np.isnan(stats["Student"]["test_std"][i])
        else 0
    )
    kd_std = (
        stats["KD"]["test_std"][i] if not np.isnan(stats["KD"]["test_std"][i]) else 0
    )
    ig_std = (
        stats["IG"]["test_std"][i] if not np.isnan(stats["IG"]["test_std"][i]) else 0
    )
    kd_ig_std = (
        stats["KD_IG"]["test_std"][i]
        if not np.isnan(stats["KD_IG"]["test_std"][i])
        else 0
    )

    print(
        f"{cf+'x':<20} {student_std:.2f}      {kd_std:.2f}      {ig_std:.2f}      {kd_ig_std:.2f}"
    )
print("-" * 80)

# LaTeX table format for results with mean ± std
print("\n\nLaTeX Table Format with Mean ± Std:")
print(r"\begin{table*}")
print(r"\centering")
print(
    r"\caption{Testing accuracies (\%) across different compression factors and training configurations. Results are presented as mean $\pm$ standard deviation, with best results in bold.}"
)
print(r"\label{tab:compression_performance}")
print(r"\begin{tabular}{ccccc}")
print(r"\hline")
print(r"Compression & \multicolumn{4}{c}{Testing Accuracy (\%)} \\")
print(r"Factor & Student & KD & IG & KD \& IG \\")
print(r"\hline")

for i, cf in enumerate(COMPRESSION_FACTORS):
    student_mean = (
        stats["Student"]["test_mean"][i]
        if not np.isnan(stats["Student"]["test_mean"][i])
        else 0
    )
    kd_mean = (
        stats["KD"]["test_mean"][i] if not np.isnan(stats["KD"]["test_mean"][i]) else 0
    )
    ig_mean = (
        stats["IG"]["test_mean"][i] if not np.isnan(stats["IG"]["test_mean"][i]) else 0
    )
    kd_ig_mean = (
        stats["KD_IG"]["test_mean"][i]
        if not np.isnan(stats["KD_IG"]["test_mean"][i])
        else 0
    )

    student_std = (
        stats["Student"]["test_std"][i]
        if not np.isnan(stats["Student"]["test_std"][i])
        else 0
    )
    kd_std = (
        stats["KD"]["test_std"][i] if not np.isnan(stats["KD"]["test_std"][i]) else 0
    )
    ig_std = (
        stats["IG"]["test_std"][i] if not np.isnan(stats["IG"]["test_std"][i]) else 0
    )
    kd_ig_std = (
        stats["KD_IG"]["test_std"][i]
        if not np.isnan(stats["KD_IG"]["test_std"][i])
        else 0
    )

    # Find the maximum value for highlighting
    max_val = max(student_mean, kd_mean, ig_mean, kd_ig_mean)

    # Format each value with mean ± std, and \textbf for the maximum
    student_str = (
        f"\\textbf{{{student_mean:.2f} $\\pm$ {student_std:.2f}}}"
        if student_mean == max_val
        else f"{student_mean:.2f} $\\pm$ {student_std:.2f}"
    )
    kd_str = (
        f"\\textbf{{{kd_mean:.2f} $\\pm$ {kd_std:.2f}}}"
        if kd_mean == max_val
        else f"{kd_mean:.2f} $\\pm$ {kd_std:.2f}"
    )
    ig_str = (
        f"\\textbf{{{ig_mean:.2f} $\\pm$ {ig_std:.2f}}}"
        if ig_mean == max_val
        else f"{ig_mean:.2f} $\\pm$ {ig_std:.2f}"
    )
    kd_ig_str = (
        f"\\textbf{{{kd_ig_mean:.2f} $\\pm$ {kd_ig_std:.2f}}}"
        if kd_ig_mean == max_val
        else f"{kd_ig_mean:.2f} $\\pm$ {kd_ig_std:.2f}"
    )

    print(f"{cf}x & {student_str} & {kd_str} & {ig_str} & {kd_ig_str} \\\\")

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table*}")

# Continue with the rest of the script
# Use the provided exact values
compression_factors = [1.0, 2.19, 4.12, 7.29, 12.04, 28.97, 54.59, 139.43, 1121.71]
teacher_accuracy = 93.91
speedup_values = [1.0, 10.6, 11.1, 15.0, 17.1, 20.6, 25.27, 35.71, 103.5]

# Normalize test accuracy relative to the teacher model (teacher = 100%)
normalized_acc = {}
normalized_std = {}  # Add normalized std dev
for model in MODEL_TYPES:
    # Add the teacher accuracy as the first point (100%)
    normalized_acc[model] = [100.0]  # Teacher is 100%
    normalized_std[model] = [0.0]  # Teacher has zero std dev (single point)

    # Add the normalized values for all other compression factors
    for i, mean in enumerate(stats[model]["test_mean"]):
        if not np.isnan(mean):
            norm_acc = (mean / teacher_accuracy) * 100
            normalized_acc[model].append(norm_acc)

            # Calculate normalized standard deviation
            std_val = stats[model]["test_std"][i]
            if not np.isnan(std_val):
                norm_std = (std_val / teacher_accuracy) * 100
                normalized_std[model].append(norm_std)
            else:
                normalized_std[model].append(0.0)
        else:
            normalized_acc[model].append(np.nan)
            normalized_std[model].append(np.nan)

# Calculate normalized min and max values
normalized_min = {}
normalized_max = {}
for model in MODEL_TYPES:
    normalized_min[model] = [100.0]  # Teacher point
    normalized_max[model] = [100.0]  # Teacher point

    for i in range(len(COMPRESSION_FACTORS)):
        test_min = stats[model]["test_min"][i]
        test_max = stats[model]["test_max"][i]

        if not np.isnan(test_min) and not np.isnan(test_max):
            norm_min = (test_min / teacher_accuracy) * 100
            norm_max = (test_max / teacher_accuracy) * 100
            normalized_min[model].append(norm_min)
            normalized_max[model].append(norm_max)
        else:
            normalized_min[model].append(np.nan)
            normalized_max[model].append(np.nan)

# Define color map
color_map = {"Student": "blue", "KD": "orange", "KD_IG": "green", "IG": "red"}

# Main figure setup
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 36
plt.rcParams["lines.linewidth"] = 2

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(24, 15))
ax2 = ax1.twinx()

# Add teacher point with star marker
ax1.scatter(
    compression_factors[0], 100, marker="*", color="black", s=400, label="Teacher"
)

# Plotting on left y-axis (Accuracy with error bars)
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    # Plot all models with line, markers, and error bars
    ax1.errorbar(
        compression_factors[1:],
        normalized_acc[model][1:],
        yerr=normalized_std[model][1:],
        marker="o",
        color=color,
        label=display_name,
        capsize=5,
    )

    # Additionally use min/max shaded region for KD_IG
    if model == "KD_IG":
        ax1.fill_between(
            compression_factors[1:],
            normalized_min[model][1:],
            normalized_max[model][1:],
            alpha=0.3,
            color=color,
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

# Plot data in inset with error bars
for model in MODEL_TYPES:
    display_name = "KD & IG" if model == "KD_IG" else model
    color = color_map[model]

    model_acc_mini = [normalized_acc[model][i] for i in inset_indices]
    model_std_mini = [normalized_std[model][i] for i in inset_indices]

    # Add error bars to all models in the inset
    inset_ax1.errorbar(
        cf_mini, model_acc_mini, yerr=model_std_mini, marker="o", color=color, capsize=3
    )

    # Add shaded region for KD_IG in inset
    if model == "KD_IG":
        min_values = [normalized_min[model][i] for i in inset_indices]
        max_values = [normalized_max[model][i] for i in inset_indices]
        inset_ax1.fill_between(cf_mini, min_values, max_values, alpha=0.3, color=color)

# Add speedup line to inset
inset_ax2.plot(cf_mini, speedup_mini, marker="s", color="black", linestyle="--")

# Configure inset axes
# inset_ax1.set_xscale("log")
inset_ax1.grid(True)
inset_ax1.xaxis.set_major_formatter(ScalarFormatter())

# Save figure
# plt.savefig("Hernandez2025_compression_vs_acc_speedup_with_std.pdf")
