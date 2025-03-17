import os

import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
import torch.nn as nn  # type:ignore
import torchvision.transforms as transforms  # type:ignore
from cifar10_models.mobilenetv2 import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10  # type:ignore
from UTILS_TORCH import (
    CIFAR10_KD,
    CIFAR10WithIG,
    count_parameters,
    train_eval_AT,
    train_eval_kd,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.001
ALPHA = 0
TEMP = 1
GAMMA = 0
NUM_WORKERS = 16
OVERLAY_PROB = 0
LAYERS = 5
# LAYERS = [3, 5, 7, 9, 11, 13, 15, 17]
START = 0
SIMULATIONS = 10
TYPE = "Student"


Teacher = mobilenet_v2(pretrained=True)
Teacher.to(device)

teacher_params = count_parameters(Teacher)


class Smaller(nn.Module):
    def __init__(self, original_model):
        super(Smaller, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-LAYERS]
        )

        for block in reversed(self.features):
            if hasattr(block, "conv"):
                if hasattr(block.conv, "__iter__"):
                    # Find the last Conv2d module in the block
                    for layer in reversed(block.conv):
                        if isinstance(layer, nn.Conv2d):
                            num_output_channels = layer.out_channels
                            break
                    break
            elif isinstance(block, nn.Conv2d):
                num_output_channels = block.out_channels
                break

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_output_channels, 10),  # type:ignore
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


test = Smaller(mobilenet_v2(pretrained=False))
test.to(device)

S = count_parameters(test)
CF = teacher_params / S

# print(
#     f"Epoch: {NUM_EPOCHS}, Alpha = {ALPHA}, T = {TEMP}, \
#     CF: {CF}"
# )

FOLDER = "compression_vs_acc/"
os.makedirs(os.path.dirname(FOLDER), exist_ok=True)
SAVE = f"compression_vs_acc/{TYPE}_{CF:.2f}.csv"
print("Saving results as: ", SAVE)
print("Number of runs for the simulation: ", SIMULATIONS)


precomputed_logits = np.load("data/cifar10_logits.npy")

# teacher_attention_maps = np.load("./data/cifar10_attention_maps.npy")

# IGS = "./Captum_IGs.npy"
# print(f"Using IGs: {IGS}")
# Load the precomputed IGs
# igs = np.load(IGS)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

student_aug = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# Define the full training dataset without augmentation for splitting
train_dataset = CIFAR10_KD(
    root="./data",
    train=True,
    transform=student_aug,
    precomputed_logits=precomputed_logits,
    # igs=igs,
)

# Load the data into batches
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

# load student data
test_data = CIFAR10(
    root="./data",
    train=False,
    transform=transform,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)


# montecarlo simulation
if os.path.isfile(SAVE):
    csv = pd.read_csv(SAVE)
    results = csv["Test Accuracy"]
    results = list(results)
    print(results)
    best_acc = np.array(results)
    best_acc = best_acc.max()
    print(best_acc)
else:
    results = []
    best_acc = 0
for i in range(START, SIMULATIONS):
    Student = Smaller(mobilenet_v2(pretrained=False))
    Student.to(device)

    model, acc = train_eval_kd(
        student=Student,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=NUM_EPOCHS,
        lr=LEARN_RATE,
        TEMP=TEMP,
        ALPHA=ALPHA,
        # GAMMA=GAMMA,
        device=device,
        csv_path=f"{FOLDER}{i+1}.csv",
    )
    results.append(acc)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{FOLDER}/{TYPE}_{CF}.pt")
    print(f"Simulation [{i+1}/{SIMULATIONS}]: Test Acc = {acc:.2f}%")
    del Student
    torch.cuda.empty_cache()
    print("Saving simulation")
    print(f"Best Accuracy: {best_acc}")

    data = pd.DataFrame(results, columns=["Test Accuracy"])
    data.to_csv(SAVE)
