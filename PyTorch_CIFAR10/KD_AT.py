import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
from torch.utils.data import DataLoader  # type:ignore
from torchvision import transforms  # type:ignore
from torchvision.datasets import CIFAR10  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import (
    CIFAR10_AT,
    CIFAR10WithIG,
    ModifiedStudent,
    ModifiedTeacher,
    count_parameters,
    train_eval_AT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARN_RATE = 0.001
ALPHA = 0  # distillation_loss
TEMP = 3.0
OVERLAY_PROB = 0.5
GAMMAS = [0.75]
# GAMMAS = [0.1, 0.25, 0.5, 0.75, 0.9]
NUM_WORKERS = 6
SAVE = "Histories/Results/IG_AT_0.5.csv"
IGS = "Captum_IGs.npy"


precomputed_logits = np.load("./data/cifar10_logits.npy")
print("Shape of teacher_logits:", precomputed_logits.shape)

precomputed_attn = np.load("./data/cifar10_attention_maps.npy")
print("Shape of teacher_logits:", precomputed_attn.shape)

igs = np.load(IGS)

print("Hyperparams:")
print(
    f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, \
    learning_rate = {LEARN_RATE}, alpha = {ALPHA}, \
    temperature = {TEMP}, num_workers = {NUM_WORKERS}, \
    LAMBDA_ATTN = {GAMMAS}"
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

# # Define the full training dataset without augmentation for splitting
# train_dataset = CIFAR10_AT(
#     root="./data",
#     train=True,
#     transform=student_aug,
#     precomputed_logits=precomputed_logits,
#     precomputed_attn=precomputed_attn,
# )

train_dataset = CIFAR10WithIG(
    igs=igs,
    root="./data",
    train=True,
    transform=student_aug,
    overlay_prob=OVERLAY_PROB,
    return_ig=False,
    precomputed_logits=precomputed_logits,
    precomputed_attn=precomputed_attn,
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

# # param search
# results = []
# for LAMBDA_ATTN in GAMMAS:
#     print(f"Using lambda: {LAMBDA_ATTN}")
#
#     Student = ModifiedStudent(mobilenet_v2(pretrained=False))
#     Student.to(device)
#
#     _, acc = train_eval_AT(
#         student=Student,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         epochs=NUM_EPOCHS,
#         lr=LEARN_RATE,
#         TEMP=TEMP,
#         ALPHA=ALPHA,
#         GAMMA=LAMBDA_ATTN,
#         device=device,
#         csv_path=f"Histories/KD_param/KD_AT_{ALPHA}_{TEMP}_{LAMBDA_ATTN}.csv",
#     )
#
#     results.append({"Lambda": LAMBDA_ATTN, "Test Accuracy": acc})
#     print(f"Test Acc = {acc:.2f}%")
#     del Student
#     torch.cuda.empty_cache()
#     print("Simulation completed")
#
# data = pd.DataFrame(results)
# data.to_csv(SAVE, index=False)


Student = ModifiedStudent(mobilenet_v2(pretrained=False))
Student.to(device)

_, acc = train_eval_AT(
    student=Student,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=NUM_EPOCHS,
    lr=LEARN_RATE,
    TEMP=TEMP,
    ALPHA=ALPHA,
    GAMMA=GAMMAS[0],
    device=device,
    csv_path=f"Histories/KD_param/IG_AT_{ALPHA}_{TEMP}_{GAMMAS[0]}_{OVERLAY_PROB}.csv",
)

print(f"Test Acc = {acc:.2f}%")
print("______________________________________________________________")

# Calculate and print model parameter counts and compression factor
Teacher = ModifiedTeacher(mobilenet_v2(pretrained=True))
Student = ModifiedStudent(mobilenet_v2(pretrained=True))


teacher_params = count_parameters(Teacher)
student_params = count_parameters(Student)
compression_factor = teacher_params / student_params

print(f"Teacher Model Parameters: {teacher_params}")
print(f"Student Model Parameters: {student_params}")
print(f"Compression Factor: {compression_factor:.2f}")
