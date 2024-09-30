import os

import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
from torch.utils.data import DataLoader  # type:ignore
from torchvision import transforms  # type:ignore
from torchvision.datasets import CIFAR10  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import CIFAR10_KD, CIFAR10WithIG, SmallerMobileNet, train_eval_kd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

os.makedirs(os.path.dirname("Histories/Results/"), exist_ok=True)

SAVE = "Histories/Results/KD_0.1_3_0.5_2.csv"
print("Saving results as: ", SAVE)

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.001
ALPHA = 0.1
# ALPHA = 0
TEMP = 3.0
NUM_WORKERS = 6
OVERLAY_PROB = 0.5
IGS = "Captum_IGs.npy"
print(f"Using IGs: {IGS}")


precomputed_logits = np.load("./data/cifar10_logits.npy")
print("Shape of teacher_logits:", precomputed_logits.shape)

print("Hyperparams:")
print(
    f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, \
    learning_rate = {LEARN_RATE}, alpha = {ALPHA}, \
    temperature = {TEMP}, num_workers = {NUM_WORKERS}"
)

# Load the precomputed IGs
igs = np.load(IGS)
print(f"IGs shape: {igs.shape}")


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
)

# train_dataset = CIFAR10WithIG(
#     igs=igs,
#     root="./data",
#     train=True,
#     transform=student_aug,
#     precomputed_logits=precomputed_logits,
#     overlay_prob=OVERLAY_PROB,
#     return_ig=False,
# )

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
    download=True,
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


Student = SmallerMobileNet(mobilenet_v2(pretrained=False))
Student.to(device)

_, acc = train_eval_kd(
    student=Student,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=NUM_EPOCHS,
    lr=LEARN_RATE,
    TEMP=TEMP,
    ALPHA=ALPHA,
    device=device,
    csv_path=SAVE,
)

print(f"Test Acc = {acc:.2f}%")
