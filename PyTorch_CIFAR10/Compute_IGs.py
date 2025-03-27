import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4
CAPTUM = "./data/Captum_IGs.npy"

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ]
)
transform = transforms.Compose(
    [
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ]
)

# Download the data
train_data = datasets.CIFAR10(
    "./data", train=True, download=True, transform=train_transform
)
test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

# Load the data into batches
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

# Load the teacher model
Teacher = mobilenet_v2(pretrained=True)
Teacher.to(device)

print("Testing on the Teacher: ")
test_model(model=Teacher, loader=test_loader, device=device)

Teacher.eval()

ig_captum = IntegratedGradients(Teacher)

captum_igs = []  # Store Captum's integrated gradients

# Wrap the DataLoader with tqdm for a progress bar
for inputs, targets in tqdm(train_loader, desc="Computing IGs", leave=True):
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Compute attributions using Captum
    attributions = ig_captum.attribute(
        inputs, target=targets, baselines=inputs * 0, return_convergence_delta=False
    )
    # Sum attributions over the channel dimension and take absolute value
    attributions_summed = torch.sum(torch.abs(attributions), dim=1).cpu().numpy()

    captum_igs.append(attributions_summed)

captum_igs = np.concatenate(captum_igs, axis=0)
np.save(CAPTUM, captum_igs)
print(f"Shape of IGs array: {captum_igs.shape}")


print("Captum IGs have been saved")
