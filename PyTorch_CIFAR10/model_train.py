import matplotlib.pyplot as plt  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
import torch.nn as nn  # type:ignore

# import torch.nn.functional as F  # type:ignore
import torch.optim as optim  # type:ignore

# import torchvision.models as models  # type:ignore
from torch.utils.data import DataLoader, random_split  # type:ignore
from torchsummary import summary  # type:ignore
from torchvision import datasets, transforms  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 4
SAVE_HIST = "Histories/layers_5_pt_true.csv"
SAVE_FIG = "Figures/layers_5_pt_true.pdf"

print("Hyperparams:")
print(f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}")
print(f"learning_rate = {LEARNING_RATE}, num_workers = {NUM_WORKERS}")

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
full_training_ds = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Generate validation set
train_size = int(0.8 * len(full_training_ds))
valid_size = len(full_training_ds) - train_size
train_indices, valid_indices = random_split(
    range(len(full_training_ds)), [train_size, valid_size]
)

# Create training and validation datasets
train_data = datasets.CIFAR10(root="./data", train=True, transform=student_aug)
valid_data = datasets.CIFAR10(root="./data", train=True, transform=transform)

# Create subsets for training and validation
train_dataset = torch.utils.data.Subset(train_data, train_indices)
valid_dataset = torch.utils.data.Subset(valid_data, valid_indices)

# Define the test dataset without augmentation
teacher_test_data = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=teacher_transform
)
test_data = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
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
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

# Load the data into batches for the teacher
teacher_test_loader = DataLoader(
    teacher_test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)

# Teacher model (pretrained)
# model = mobilenet_v2(pretrained=False)
# model.to(device)
# summary(model, (3,32,32))


# model = models.shufflenet_v2_x1_0(pretrained=False)
model = SmallerMobileNet(mobilenet_v2(pretrained=True))
model.to(device)

summary(model, (3, 32, 32))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training

train_losses = []
val_losses = []
train_acc = []
val_acc = []

total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    model.train()

    TOTAL_TRAIN = 0
    CORRECT_TRAIN = 0
    RUNNING_TRAIN_LOSS = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass with Student
        logits = model(images)
        # Calculate the student loss
        loss = criterion(logits, labels)

        # Backward pass and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        TOTAL_TRAIN += labels.size(0)
        CORRECT_TRAIN += (predicted == labels).sum().item()
        RUNNING_TRAIN_LOSS += loss.item()

        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
            )

    train_accuracy = 100 * CORRECT_TRAIN / TOTAL_TRAIN
    train_losses.append(RUNNING_TRAIN_LOSS / total_step)
    train_acc.append(train_accuracy)

    print(f"Accuracy of the student network on the train images: {train_accuracy} %")

    # Validation phase
    model.eval()
    TOTAL_VAL = 0
    CORRECT_VAL = 0
    RUNNING_VAL_LOSS = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            RUNNING_VAL_LOSS += loss.item()

            _, predicted = torch.max(logits.data, 1)
            TOTAL_VAL += labels.size(0)
            CORRECT_VAL += (predicted == labels).sum().item()

    val_accuracy = 100 * CORRECT_VAL / TOTAL_VAL
    val_losses.append(RUNNING_VAL_LOSS / len(valid_loader))
    val_acc.append(val_accuracy)

    print(f"Accuracy of the student network on the validation images: {val_accuracy} %")

# Create a DataFrame
data = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "train_acc": train_acc,
    "val_acc": val_acc,
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
test_model(model=model, loader=test_loader, device=device, split_name="Testing")

print("___________________________________________________________________________")

print("Model evaluation on training data: ")
test_model(model=model, loader=train_loader, device=device)
