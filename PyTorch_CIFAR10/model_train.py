import os

import numpy as np  # type:ignore
import pandas as pd  # type:ignore
import torch  # type:ignore
import torch.nn as nn  # type:ignore
from torch.utils.data import DataLoader  # type:ignore
from torchvision import datasets, transforms  # type:ignore

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import count_parameters, train_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 16
START = 0
SIMULATIONS = 10
LAYERS = 13

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


print(f"Epoch: {NUM_EPOCHS}, CF: {CF}")


FOLDER = "Histories/compression_vs_acc/"
os.makedirs(os.path.dirname(FOLDER), exist_ok=True)
SAVE = f"Histories/compression_vs_acc/Student_{CF:.2f}.csv"
print("Saving results as: ", SAVE)
print("Number of runs for the simulation: ", SIMULATIONS)

print("Hyperparams:")
print(f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}")
print(f"learning_rate = {LEARNING_RATE}, num_workers = {NUM_WORKERS}")


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
    root="./data", train=True, download=False, transform=student_aug
)

test_data = datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

# Load the data into batches
train_loader = DataLoader(
    full_training_ds,
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

# model = models.shufflenet_v2_x1_0(pretrained=False)
model = Smaller(mobilenet_v2(pretrained=False))
model.to(device)

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

    model, acc = train_eval(
        model,
        train_loader,
        test_loader,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        csv_path=f"{FOLDER}{i+1}.csv",
    )

    results.append(acc)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{FOLDER}/Student_{CF}.pt")
    print(f"Simulation [{i+1}/{SIMULATIONS}]: Test Acc = {acc:.2f}%")
    del Student
    torch.cuda.empty_cache()
    print("Saving simulation")
    print(f"Best Accuracy: {best_acc}")

    data = pd.DataFrame(results, columns=["Test Accuracy"])
    data.to_csv(SAVE)
