from cifar10_models.mobilenetv2 import mobilenet_v2 # This is our teacher model
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize
import numpy as np
# import matplotlib.pyplot as plt
from utils_KDIG_torch import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
    ToTensor(),
    Normalize(mean= [0.4914, 0.4822, 0.4465], std = [0.2471, 0.2435, 0.2616])
    ])

# Download the data
train_data = CIFAR10("./data", train=True, download=True, transform=transform)
test_data = CIFAR10('./data', train=False, download=True, transform=transform)

# Load the data into batches
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True, num_workers=4)

# Load the teacher model
Teacher = mobilenet_v2(pretrained=True)
# print(Teacher)
Teacher.to(device)

m = 50 # number of steps
alphas = torch.linspace(0,1, steps=m+1).to(device)
x_train_igs = [] # Store all integrated gradients

for idx, (inputs, targets) in enumerate(train_loader):
    inputs = inputs.to(device)
    baseline = torch.zeros_like(inputs).to(device)
    # Initialize batch_igs to match expected dimension after summing over channels
    batch_igs = torch.zeros(inputs.size(0), inputs.size(2), inputs.size(3), device=device)  # [batch, height, width]

    # Generate interpolated images once per batch
    interpolated_images = interpolate_images(baseline, inputs, alphas)
    interpolated_images = interpolated_images.view(-1, *inputs.shape[1:]) # Flatten along batch and alpha

    for target_class_idx in range(10):
        target_class_indices = torch.full((inputs.size(0),), target_class_idx, dtype=torch.long, device=device)
        target_class_indices = target_class_indices.repeat_interleave(m+1) # Repeat for each alpha

        path_gradients = compute_gradients(Teacher, interpolated_images, target_class_indices)
        path_gradients = path_gradients.view(m+1, inputs.size(0), *inputs.shape[1:])  # Reshape to separate alphas
        ig = integral_approximation(path_gradients)
        
        # Sum over the channel dimension
        summed_ig = torch.sum(torch.abs(ig), dim=1)  # Reduce over channels
        if summed_ig.shape != batch_igs.shape:
            raise RuntimeError(f"Shape mismatch: summed_ig shape {summed_ig.shape} does not match batch_igs shape {batch_igs.shape}")

        batch_igs += summed_ig

    x_train_igs.append(batch_igs.cpu().numpy())

x_train_igs = np.concatenate(x_train_igs, axis=0)
np.save("Teacher_igs_batch.npy", x_train_igs)
print(x_train_igs.shape)
