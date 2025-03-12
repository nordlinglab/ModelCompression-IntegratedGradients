import os

import matplotlib.pyplot as plt
import numpy as np  # type:ignore
import torch  # type:ignore
from torch.utils.data import DataLoader  # type:ignore
from torchvision import datasets, transforms  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import (
    CIFAR10WithIG,
    ModifiedStudent,
    ModifiedTeacher,
    test_model_att,
    train_eval_AT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 16
OVERLAY_PROB = 0.1
GAMMA = 0.8
FOLDER = "Histories/Attention_maps/"
os.makedirs(os.path.dirname(FOLDER), exist_ok=True)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


# Load the precomputed teacher logits and IGs (as in your code)
teacher_logits = np.load("./data/cifar10_logits.npy")
teacher_attention_maps = np.load("./data/cifar10_attention_maps.npy")
igs = np.load("Captum_IGs.npy")
print(f"Teacher logits shape: {teacher_logits.shape}")
print(f"Teacher attention maps shape: {teacher_attention_maps.shape}")
print(f"IGs shape: {igs.shape}")

# Define the full training dataset without augmentation for splitting
ig_training_dataset = CIFAR10WithIG(
    igs=igs,
    root="./data",
    train=True,
    transform=train_transform,
    overlay_prob=OVERLAY_PROB,
    return_ig=False,
    precomputed_logits=teacher_logits,
    precomputed_attn=teacher_attention_maps,
)

# Define the full training dataset without augmentation for splitting
training_dataset = CIFAR10WithIG(
    igs=igs,
    root="./data",
    train=True,
    transform=train_transform,
    overlay_prob=0.0,
    return_ig=False,
    precomputed_logits=teacher_logits,
    precomputed_attn=teacher_attention_maps,
)

test_data = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

# Define our four configurations:
configs = {
    "AT": {"ALPHA": 0.0, "TEMP": 1.0, "GAMMA": GAMMA, "dataset": training_dataset},
    "KD_AT": {"ALPHA": 0.01, "TEMP": 2.5, "GAMMA": GAMMA, "dataset": training_dataset},
    "KD_IG_AT": {
        "ALPHA": 0.01,
        "TEMP": 2.5,
        "GAMMA": GAMMA,
        "dataset": ig_training_dataset,
    },
    "IG_AT": {
        "ALPHA": 0.0,
        "TEMP": 1.0,
        "GAMMA": GAMMA,
        "dataset": ig_training_dataset,
    },
}

# Dictionary to hold our trained student models:
trained_models = {}

for config_name, cfg in configs.items():
    print(f"\n--- Training model: {config_name} ---")

    # Create a dataloader for the dataset specified in the config
    train_loader = DataLoader(
        cfg["dataset"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=True,
    )

    # Instantiate a student model
    student = ModifiedStudent(mobilenet_v2(pretrained=True))
    student.to(device)

    model, acc = train_eval_AT(
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        TEMP=cfg["TEMP"],
        ALPHA=cfg["ALPHA"],
        GAMMA=cfg["GAMMA"],
        device=device,
        csv_path=f"{FOLDER}{config_name}.csv",
    )

    print(f"Configuration {config_name} achieved Test Accuracy: {acc:.2f}%")
    trained_models[config_name] = model

teacher_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        ),
    ]
)

# load student data
teacher_data = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=teacher_transform,
)

teacher_loader = DataLoader(
    teacher_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

Teacher_att = ModifiedTeacher(mobilenet_v2(pretrained=True))
Teacher_att.to(device)

test_model_att(Teacher_att, teacher_loader, device)

teacher_attention_maps = {i: None for i in range(10)}
student_attention_maps = {config: {i: None for i in range(10)} for config in configs}

with torch.no_grad():
    for (t_img, _), (s_img, labels) in zip(teacher_loader, test_loader):
        t_img = t_img.to(device)
        s_img = s_img.to(device)
        labels = labels.to(device)

        # Teacher outputs (assumed to return (logits, attention_maps))
        t_out, t_at = Teacher_att(t_img)

        # Process teacher attention maps:
        for label, t_attention_map in zip(labels, t_at):
            label = label.item()  # convert tensor to scalar
            if teacher_attention_maps[label] is None:
                # Store the teacher's attention map for this class if not already stored
                teacher_attention_maps[label] = t_attention_map.cpu().detach()

        # For each configuration, compute student attention maps
        for config, model in trained_models.items():
            # Student outputs for the current batch (assumed to return (logits, attention_maps))
            s_out, s_at = model(s_img)
            for label, s_attention_map in zip(labels, s_at):
                label = label.item()
                if student_attention_maps[config][label] is None:
                    student_attention_maps[config][
                        label
                    ] = s_attention_map.cpu().detach()

        # Break if we've collected one sample per class for both teacher and all student configs
        if all(val is not None for val in teacher_attention_maps.values()) and all(
            all(val is not None for val in student_attention_maps[conf].values())
            for conf in student_attention_maps
        ):
            break

# Convert teacher attention maps from tensors to NumPy arrays (if not already)
teacher_attention_maps_np = {
    k: v.numpy() if isinstance(v, torch.Tensor) else v
    for k, v in teacher_attention_maps.items()
    if v is not None
}

# Convert student attention maps from tensors to NumPy arrays
student_attention_maps_np = {}
for config, attn_dict in student_attention_maps.items():
    student_attention_maps_np[config] = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v
        for k, v in attn_dict.items()
        if v is not None
    }

# Save both dictionaries in a single NPZ file
np.savez(
    "attention_maps.npz",
    teacher=teacher_attention_maps_np,
    student=student_attention_maps_np,
)

print("Attention maps saved to attention_maps.npz")
