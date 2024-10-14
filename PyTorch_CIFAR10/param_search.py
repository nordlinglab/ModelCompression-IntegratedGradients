import os

import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
from cifar10_models.mobilenetv2 import mobilenet_v2
from torch.utils.data import DataLoader  # type:ignore
from torchvision import transforms  # type:ignore
from torchvision.datasets import CIFAR10  # type:ignore
from UTILS_TORCH import CIFAR10_KD, SmallerMobileNet, train_eval_kd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

os.makedirs(os.path.dirname("Histories/KD_param/Final"), exist_ok=True)

SAVE = "Histories/KD_param/Final/KD_param_search_final.csv"
print("Saving results as: ", SAVE)

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.001
ALPHAS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
TEMPS = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]
NUM_WORKERS = 16

precomputed_logits = np.load("./data/cifar10_logits.npy")
print("Shape of teacher_logits:", precomputed_logits.shape)

print("Hyperparams:")
print(
    f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, \
    learning_rate = {LEARN_RATE}, alpha = {ALPHAS}, \
    temperature = {TEMPS}, num_workers = {NUM_WORKERS}"
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        # ),
    ]
)

student_aug = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        # ),
    ]
)

# Define the full training dataset without augmentation for splitting
train_dataset = CIFAR10_KD(
    root="./data",
    train=True,
    transform=student_aug,
    precomputed_logits=precomputed_logits,
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

# param search
results = []
for alpha in ALPHAS:
    for temp in TEMPS:

        print(f"Using alpha: {alpha}, and temp: {temp}")

        Student = SmallerMobileNet(mobilenet_v2(pretrained=False))
        Student.to(device)

        _, acc = train_eval_kd(
            student=Student,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=NUM_EPOCHS,
            lr=LEARN_RATE,
            TEMP=temp,
            ALPHA=alpha,
            device=device,
            csv_path=f"Histories/KD_param/Final/KD_{alpha}_{temp}.csv",
        )

        results.append({"Alpha": alpha, "Temp": temp, "Test Accuracy": acc})
        print(f"Test Acc = {acc:.2f}%")
        del Student
        torch.cuda.empty_cache()
        print("Simulation completed")

data = pd.DataFrame(results)
data.to_csv(SAVE, index=False)
