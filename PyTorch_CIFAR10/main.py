import os
from collections import Counter

import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
from sklearn.model_selection import train_test_split  # type:ignore
from torch.utils.data import DataLoader, Subset
from torchvision import transforms  # type:ignore
from torchvision.datasets import CIFAR10  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import CIFAR10WithIG  # test_model,
from UTILS_TORCH import (
    CIFAR10_KD,
    ModifiedStudent,
    SmallerMobileNet,
    train_eval_AT,
    train_eval_kd,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.001
# ALPHA = 0.01  # distillation_loss
# TEMP = 2.5
# GAMMA = 0.8
ALPHA = 0.0  # distillation_loss
TEMP = 1
NUM_WORKERS = 16
OVERLAY_PROB = 0.1
SPLIT_SIZE = 0.8
SIMULATIONS = 60

FOLDER = "Histories/montecarlo_2/IG/"
# FOLDER = f"Histories/AT_param/KD_IG_AT_{GAMMA}/"
os.makedirs(os.path.dirname(FOLDER), exist_ok=True)
SAVE = f"Histories/montecarlo_2/IG.csv"
print("Saving results as: ", SAVE)
print("Number of runs for the simulation: ", SIMULATIONS)


teacher_logits = np.load("./data/cifar10_logits.npy")
# teacher_attention_maps = np.load("./data/cifar10_attention_maps.npy")

IGS = "Captum_IGs.npy"

print(f"Using IGs: {IGS}")

print("Hyperparams:")
print(
    f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, \
    learning_rate = {LEARN_RATE}, alpha = {ALPHA}, \
    temperature = {TEMP}, num_workers = {NUM_WORKERS},  \
    overlay_prob = {OVERLAY_PROB}"
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

# # Define the full training dataset without augmentation for splitting
# full_training_ds = CIFAR10_KD(
#     root="./data",
#     train=True,
#     transform=student_aug,
#     precomputed_logits=teacher_logits,
# )

full_training_ds = CIFAR10WithIG(
    igs=igs,
    root="./data",
    train=True,
    transform=student_aug,
    overlay_prob=OVERLAY_PROB,
    return_ig=False,
    precomputed_logits=teacher_logits,
    # precomputed_attn=teacher_attention_maps,
)

# train_loader = DataLoader(
#     full_training_ds,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     pin_memory=False,
#     persistent_workers=True,
# )

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
results = []
best_acc = 0
for i in range(SIMULATIONS):
    # Step 1: Get the class indices
    class_indices = {
        i: np.where(np.array(full_training_ds.targets) == i)[0] for i in range(10)
    }

    # Step 2: Stratified sampling of indices
    train_indices = []
    for class_idx, indices in class_indices.items():
        train_idx, _ = train_test_split(
            indices, train_size=SPLIT_SIZE, random_state=None
        )  # None for true randomness each run
        train_indices.extend(train_idx)

    # Step 3: Shuffle the training indices
    np.random.shuffle(train_indices)

    # Step 4: Create a subset and DataLoader for the training data
    train_subset = Subset(full_training_ds, train_indices)

    # Load the data into batches
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=True,
    )

    subset_labels = [full_training_ds.targets[i] for i in train_indices]
    print("Class distribution in subset:", Counter(subset_labels))

    Student = SmallerMobileNet(mobilenet_v2(pretrained=False))
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
        torch.save(model.state_dict(), f"{FOLDER}/IG.pt")
    print(f"Simulation [{i+1}/{SIMULATIONS}]: Test Acc = {acc:.2f}%")
    del Student
    torch.cuda.empty_cache()
    print("Saving simulation")
    print(f"Best Accuracy: {best_acc}")

    data = pd.DataFrame(results, columns=["Test Accuracy"])
    data.to_csv(SAVE)
