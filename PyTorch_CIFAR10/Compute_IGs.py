import numpy as np
import torch
from captum.attr import IntegratedGradients
from cifar10_models.mobilenetv2 import mobilenet_v2
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm
from UTILS_TORCH import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# Hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4
MANUAL = "tensor_igs_norm.npy"
CAPTUM = "tensor_captum_igs_norm.npy"

transform = transforms.Compose(
    [
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]),
    ]
)

# Download the data
train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

# Load the data into batches
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

# Load the teacher model
Teacher = mobilenet_v2(pretrained=True)
Teacher.to(device)

print("Testing on the Teacher: ")
test_model(model=Teacher, loader=test_loader, device=device, split_name="Testing")

Teacher.eval()

m = 50  # number of steps
alphas = torch.linspace(0, 1, steps=m + 1).to(device)
x_train_igs = []  # Store all integrated gradients

for idx, (inputs, targets) in enumerate(
    tqdm(train_loader, desc="Batch Progress", leave=True)
):
    inputs = inputs.to(device)
    baseline = torch.zeros_like(inputs).to(device)
    batch_igs = torch.zeros(
        inputs.size(0), inputs.size(2), inputs.size(3), device=device
    )  # Initialize batch_igs

    # Generate interpolated images
    interpolated_images = interpolate_images(baseline, inputs, alphas)
    interpolated_images = interpolated_images.view(
        -1, *inputs.shape[1:]
    )  # Flatten along batch and alpha

    # Iterate over each class index, optionally with tqdm for visibility
    for target_class_idx in tqdm(range(10), desc=f"Class Calculations", leave=False):
        target_class_indices = torch.full(
            (inputs.size(0),), target_class_idx, dtype=torch.long, device=device
        )
        target_class_indices = target_class_indices.repeat_interleave(m + 1)

        path_gradients = compute_gradients(
            Teacher, interpolated_images, target_class_indices
        )
        path_gradients = path_gradients.view(m + 1, inputs.size(0), *inputs.shape[1:])
        ig = integral_approximation(path_gradients)

        summed_ig = torch.sum(torch.abs(ig), dim=1)  # Sum over the channel dimension
        if summed_ig.shape != batch_igs.shape:
            raise RuntimeError(
                f"Shape mismatch: summed_ig shape {summed_ig.shape} does not match batch_igs shape {batch_igs.shape}"
            )

        batch_igs += summed_ig

    x_train_igs.append(batch_igs.cpu().numpy())

x_train_igs = np.concatenate(x_train_igs, axis=0)
np.save(MANUAL, x_train_igs)
print(f"Integrated gradients computed for all batches: {x_train_igs.shape}")

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
