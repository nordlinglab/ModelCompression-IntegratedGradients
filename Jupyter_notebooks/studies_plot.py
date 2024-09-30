import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text

data = {
    "MNIST": {
        "wang2019private": [(-0.20, 15.7), (-0.54, 32.23)],
        "chen2019data": [(-0.71, 3.88)],
        "ashok2017n2n": [(0.01, 127.02)],
    },
    "SVHN": {
        "wang2019private": [(-0.97, 20.14), (-1.87, 45.25)],
        "su2022stkd": [(-0.21, 3.35)],
        "choi2020data": [(-1.56, 11.00)],
        "ashok2017n2n": [(0.18, 19.80)],
        "zhao2020highlight": [(-0.43, 6.40), (-0.06, 3.29)],
    },
    "CIFAR10": {
        "wang2019private": [(-1.78, 6.00), (-4.21, 20.80)],
        "su2022stkd": [(-0.37, 3.26)],
        "chen2019data": [(-3.36, 1.91)],
        "choi2020data": [(-0.50, 1.90), (-8.63, 11.00)],
        "ashok2017n2n": [(0.30, 10.28), (-0.33, 20.53)],
        "bhardwaj2019memory": [(0.96, 19.35), (-1.17, 20.7)],
        "blakeney2020parallel": [(-0.73, 0)],
        "gou2023hierarchical": [(0.14, 1.91), (-1.43, -1)],  # Last one doesnt have a CF
        "zhao2020highlight": [(-0.30, 3.19), (-0.93, 3.52)],
    },
    "CIFAR100": {
        "su2022stkd": [(-1.20, 3.26)],
        "chen2019knowledge": [(-1.01, 1.90), (-0.74, 2.06)],
        "hossain2024purf": [(1.01, 3.95), (-3.03, -1)],  # No CF
        "chen2019data": [(-3.37, 1.91)],
        "choi2020data": [(-1.33, 1.90)],
        "ashok2017n2n": [(-2.75, 5.02), (-4.21, 4.64)],
        "chen2022knowledge": [(-0.13, -1), (0.12, -1)],  # No CF
        "gou2022multilevel": [(1.54, 1.8), (-7.11, 52.2)],
        "bhardwaj2019memory": [(-3.54, 42.94)],
        "blakeney2020parallel": [(-1.61, 0)],
        "gou2023hierarchical": [(0.86, 1.67), (-2.79, -1)],  # No CF
        "zhao2020highlight": [(-1.90, 6.30), (-0.72, 3.19)],
    },
    "ImageNet": {
        "su2022stkd": [(-0.37, 2.35)],
        "chen2019knowledge": [(-6.49, 36.67)],
        "chen2022knowledge": [(-4.60, -1)],  # No CF
        "gou2022multilevel": [(-2.19, 1.90)],
        "blakeney2020parallel": [(-0.60, 0.00)],
        "gou2023hierarchical": [(-2.40, 1.90)],
        "zhao2020highlight": [(-0.36, 2.35)],
    },
    "others": {
        "chen2019data": [(-1.56, 3.20)],
        "chen2018darkrank": [(-1.00, 1.36), (-3.10, 1.36)],
        "ashok2017n2n": [(-2.94, 3.12)],
        "xie2021model": [(-0.10, 8.38), (-2.40, 15.53), (0, 8.38), (-3.3, 15.53)],
    },
    "tiny_imagenet": {
        "choi2020data": [(-2.61, 1.89)],
        "zhao2020highlight": [(-2.92, 3.29)],
    },
}

# Using a ColorBrewer qualitative palette with seaborn
colors = sns.color_palette("Set1", n_colors=len(data))

plt.figure(figsize=(21, 8))
dataset_handles = []
data_group_id = 0
texts = []

for dataset, data_dict in data.items():
    color = colors[data_group_id]  # Select color from the palette
    for key, points in data_dict.items():
        delta_accuracy, cf = zip(*points)
        scatter = plt.scatter(
            cf, delta_accuracy, marker="o", color=color, s=50
        )  # Use 'o' for dots

        # Annotating keys next to their corresponding points
        for x, y in points:
            text = plt.text(
                x, y, f"{key}", fontsize=9, ha="right", va="bottom", color="black"
            )
            texts.append(text)

    # Create a legend handle for the dataset
    dataset_handles.append(mpatches.Patch(color=color, label=dataset))
    data_group_id += 1

plt.axhline(y=0, color="black", linestyle="--")
plt.ylabel("$\Delta$ Accuracy (%)")
plt.xlabel("Compression Factor (a.u.)")
plt.title("Comparison of Different Methods Across Datasets")
plt.grid(True)
plt.legend(handles=dataset_handles, title="Datasets")

# Use adjust_text to dynamically adjust the annotations
adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray"))

plt.tight_layout()
plt.show()
