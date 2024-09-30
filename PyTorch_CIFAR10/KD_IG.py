import time

import matplotlib.pyplot as plt  # type:ignore
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
import torch.nn as nn  # type:ignore
import torch.nn.functional as F  # type:ignore
import torch.optim as optim  # type:ignore
from torch.utils.data import DataLoader  # , random_split  # type:ignore
from torchvision import datasets, transforms  # type:ignore
from torchvision.datasets import CIFAR10  # type:ignore
from tqdm import tqdm  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import (
    CIFAR10WithIG,
    SmallerMobileNet,
    softmax_with_temperature,
    test_model,
)

# plt.style.use(r"../rw_visualization.mplstyle")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ALPHA = 0.1  # distillation_loss
TEMP = 3.0
NUM_WORKERS = 6
OVERLAY_PROB = 0.5
SAVE_HIST = f"Histories/Results/KD_IG_{ALPHA}_{TEMP}_{OVERLAY_PROB}.csv"
# SAVE_HIST = "Histories/Results/Student_baseline.csv"
SAVE_FIG = f"Figures/Results/KD_IG_{ALPHA}_{TEMP}_{OVERLAY_PROB}.pdf"
# SAVE_FIG = "Figures/Results/Student_baseline.pdf"
# IGS = "Captum_IGs.npy"

# print(f"Using IGs: {IGS}")

print("Hyperparams:")
print(
    f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, \
    learning_rate = {LEARNING_RATE}, alpha = {ALPHA}, \
    temperature = {TEMP}, num_workers = {NUM_WORKERS}, \
    overlay_prob = {OVERLAY_PROB}"
)

# # Load the precomputed IGs
# igs = np.load(IGS)
# print(f"IGs shape: {igs.shape}")


teacher_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        ),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

teacher_aug = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        ),
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
full_training_ds = CIFAR10(
    root="./data", train=True, download=True, transform=student_aug
)

# # Generate validation set
# train_size = int(0.8 * len(full_training_ds))
# valid_size = len(full_training_ds) - train_size
# train_indices, valid_indices = random_split(
#     range(len(full_training_ds)), [train_size, valid_size]
# )

# Create training and validation datasets
# train_data = CIFAR10(root="./data", train=True, transform=student_aug)

# full_training_ds = CIFAR10WithIG(
#     igs=igs,
#     root="./data",
#     train=True,
#     # transform=teacher_aug,  # Testing with normalise
#     transform=student_aug,
#     download=True,
#     overlay_prob=OVERLAY_PROB,
#     return_ig=False,
# )

teacher_train_data = CIFAR10(root="./data", train=True, transform=teacher_aug)

# # Testing with normalise
# valid_data = CIFAR10(root="./data", train=True, transform=teacher_transform)
# valid_data = CIFAR10(root="./data", train=True, transform=transform)

# # Create subsets for training and validation
# train_dataset = torch.utils.data.Subset(train_data, train_indices)
# valid_dataset = torch.utils.data.Subset(valid_data, valid_indices)

# Define the test dataset without augmentation
teacher_test_data = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=teacher_transform
)
test_data = CIFAR10(
    # Test with teacher's normalisation
    root="./data",
    train=False,
    download=True,
    # Testing with normalisation
    #    transform=teacher_transform,
    transform=transform,
)

# Load the data into batches
train_loader = DataLoader(
    # train_dataset,
    full_training_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

# valid_loader = DataLoader(
#     valid_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     pin_memory=False,
#     persistent_workers=True,
# )

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

# Load the data into batches for the teacher
teacher_train_loader = DataLoader(
    teacher_train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)
teacher_test_loader = DataLoader(
    teacher_test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)


# Teacher model (pretrained)
Teacher = mobilenet_v2(pretrained=True)
Teacher.to(device)
# summary(Teacher, (3, 32, 32))

print("Testing on the Teacher: ")
test_model(Teacher, teacher_test_loader, device, "Testing")

# Set the teacher to eval mode
Teacher.eval()

Student = SmallerMobileNet(mobilenet_v2(pretrained=False))
Student.to(device)
# summary(Student, (3, 32, 32))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Student.parameters(), lr=LEARNING_RATE)

# Training

total_start_time = time.time()

train_losses = []
val_losses = []
train_acc = []
val_acc = []

total_step = len(train_loader)

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    Student.train()

    train_tqdm = tqdm(
        train_loader,
        total=total_step,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
        leave=False,
    )

    TOTAL_TRAIN = 0
    CORRECT_TRAIN = 0
    RUNNING_TRAIN_LOSS = 0.0

    for i, ((teacher_images, _), (images, labels)) in enumerate(
        zip(teacher_train_loader, train_loader)
    ):
        teacher_images, images, labels = (
            teacher_images.to(device),
            images.to(device),
            labels.to(device),
        )

        # Forward pass with Teacher
        with torch.no_grad():
            teacher_logits = Teacher(teacher_images)
            teacher_probs = softmax_with_temperature(teacher_logits, TEMP)

        # Forward pass with Student
        student_logits = Student(images)
        student_probs = softmax_with_temperature(student_logits, TEMP)

        # Calculate the distillation loss using KL divergence
        distillation_loss = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(student_logits / TEMP, dim=1),
            softmax_with_temperature(teacher_logits, TEMP),
        ) * (TEMP**2)

        # Calculate the student loss
        student_loss = criterion(student_logits, labels)

        # Calculate the total loss
        loss = ALPHA * distillation_loss + (1 - ALPHA) * student_loss

        # Backward pass and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(student_logits.data, 1)
        TOTAL_TRAIN += labels.size(0)
        CORRECT_TRAIN += (predicted == labels).sum().item()
        RUNNING_TRAIN_LOSS += loss.item()

        train_tqdm.set_postfix(
            loss=loss.item(), accuracy=100 * CORRECT_TRAIN / TOTAL_TRAIN
        )

    train_tqdm.close()

    epoch_end_time = time.time()
    print(
        f"Time taken for epoch {epoch+1}: \
        {epoch_end_time - epoch_start_time:.2f} seconds"
    )

    train_accuracy = 100 * CORRECT_TRAIN / TOTAL_TRAIN
    train_losses.append(RUNNING_TRAIN_LOSS / total_step)
    train_acc.append(train_accuracy)

    print(
        "Accuracy of the student network on the train images: {} %".format(
            train_accuracy
        )
    )

    # # Validation phase
    # Student.eval()
    # TOTAL_VAL = 0
    # CORRECT_VAL = 0
    # RUNNING_VAL_LOSS = 0.0
    #
    # with torch.no_grad():
    #     for images, labels in valid_loader:
    #         images, labels = images.to(device), labels.to(device)
    #
    #         student_logits = Student(images)
    #         student_probs = softmax_with_temperature(student_logits, TEMP)
    #
    #         loss = criterion(student_logits, labels)
    #
    #         RUNNING_VAL_LOSS += loss.item()
    #
    #         _, predicted = torch.max(student_logits.data, 1)
    #         TOTAL_VAL += labels.size(0)
    #         CORRECT_VAL += (predicted == labels).sum().item()
    #
    # val_accuracy = 100 * CORRECT_VAL / TOTAL_VAL
    # val_losses.append(RUNNING_VAL_LOSS / len(valid_loader))
    # val_acc.append(val_accuracy)
    #
    # print(
    #     "Accuracy of the student network on \
    #     the validation images: {} %".format(
    #         val_accuracy
    #     )
    # )

total_end_time = time.time()
print(
    f"Total training time for {NUM_EPOCHS}\
        epochs: {total_end_time - total_start_time:.2f} seconds"
)

# Create a DataFrame
data = {
    "train_loss": train_losses,
    "train_acc": train_acc,
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(SAVE_HIST, index=False)

# Plotting losses
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Plotting accuracies
plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(SAVE_FIG)

# Test the model
print("Student evaluation on testing data: ")
test_model(Student, test_loader, device, "Testing")

# print("______________________________________________________________________")
#
# print("Teacher evaluation on training data: ")
# test_model(model=Teacher, loader=teacher_train_loader, device=device)
#
# print("Student evaluation on training data: ")
# test_model(model=Student, loader=train_loader, device=device)
