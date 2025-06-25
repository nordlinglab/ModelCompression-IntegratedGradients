import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from captum.attr import GradientShap, IntegratedGradients, LayerConductance, NoiseTunnel
from captum.attr import visualization as viz

from cifar10_models.mobilenetv2 import mobilenet_v2

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30

# # Set primary device (use cuda:2 as the primary GPU)
# primary_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#
# # Check if both GPUs are available
# if (
#     torch.cuda.is_available() and torch.cuda.device_count() >= 4
# ):  # Ensure GPUs 2 and 3 exist
#     print(f"Using GPUs: cuda:2, cuda:3")
# else:
#     print("Warning: Two GPUs not available, falling back to primary device")
#     primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directory for saving attributions
save_dir = "./attributions/"
os.makedirs(save_dir, exist_ok=True)

# File paths for saved attributions and original image
file_paths = {
    "original_image": os.path.join(save_dir, "original_image.npy"),
    "vanilla_grads": os.path.join(save_dir, "vanilla_grads.npy"),
    "smoothgrad": os.path.join(save_dir, "smoothgrad.npy"),
    "integrated_gradients": os.path.join(save_dir, "integrated_gradients.npy"),
    "lrp": os.path.join(save_dir, "lrp.npy"),
    "shap": os.path.join(save_dir, "shap.npy"),
}

# Check if all attribution files exist
all_files_exist = all(os.path.exists(path) for path in file_paths.values())

# Load or compute attributions
if all_files_exist:
    print("Loading saved attributions from", save_dir)
    original_image = np.load(file_paths["original_image"])
    vanilla_grads = np.load(file_paths["vanilla_grads"])
    smoothgrad_attr = np.load(file_paths["smoothgrad"])
    ig_attr = np.load(file_paths["integrated_gradients"])
    lrp_attr = np.load(file_paths["lrp"])
    shap_attr = np.load(file_paths["shap"])
else:
    # print("Computing attributions and saving to", save_dir)
    #
    # # Load pretrained MobileNetV2 and modify for CIFAR-10 (10 classes)
    # model = mobilenet_v2(pretrained=True)
    #
    # # Move model to primary device before wrapping with DataParallel
    # model = model.to(primary_device)
    #
    # # Wrap model with DataParallel to use GPUs 2 and 3
    # if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
    #     model = nn.DataParallel(model, device_ids=[2, 3])
    #
    # model.eval()
    #
    # # Define transforms for CIFAR-10
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
    #         ),
    #     ]
    # )
    #
    # # Load CIFAR-10 dataset
    # testset = torchvision.datasets.CIFAR10(
    #     root="./data/", train=False, download=False, transform=transform
    # )
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    #
    # # Get a single image and label
    # dataiter = iter(testloader)
    # images, labels = next(dataiter)
    # images, labels = images.to(primary_device), labels.to(primary_device)
    #
    # # Initialize attribution methods
    # ig = IntegratedGradients(model)
    # nt = NoiseTunnel(ig)  # For SmoothGrad-like effect
    # gs = GradientShap(model)
    # lc = LayerConductance(
    #     model,
    #     (
    #         model.module.features[-1]
    #         if isinstance(model, nn.DataParallel)
    #         else model.features[-1]
    #     ),
    # )  # Handle DataParallel module access
    #
    # # Function to process attribution for visualization (sum absolute values over channels)
    # def process_attribution(attr):
    #     # Sum absolute values over the channel dimension (C, H, W) -> (H, W)
    #     attr = torch.sum(torch.abs(attr), dim=1)  # Shape: (batch_size, H, W)
    #     attr = attr.detach().cpu().numpy()  # Move to CPU and convert to numpy
    #     attr = attr / (np.max(attr) + 1e-10)  # Normalize to [0, 1] for visualization
    #     return attr[0]  # Return single image attribution (batch_size=1)
    #
    # # Compute attributions
    # input_img = images.requires_grad_(True)
    #
    # # 1. Vanilla Gradients
    # gradients = model(input_img)
    # gradients = torch.autograd.grad(torch.max(gradients[0]), input_img)[0]
    # vanilla_grads = process_attribution(gradients)
    # np.save(file_paths["vanilla_grads"], vanilla_grads)
    #
    # # 2. SmoothGrad (using NoiseTunnel with Integrated Gradients)
    # smoothgrad_attr = nt.attribute(
    #     input_img, nt_type="smoothgrad", nt_samples=50, target=labels
    # )
    # smoothgrad_attr = process_attribution(smoothgrad_attr)
    # np.save(file_paths["smoothgrad"], smoothgrad_attr)
    #
    # # 3. Integrated Gradients (following your captum IG computation)
    # ig_attr = ig.attribute(input_img, target=labels, baselines=input_img * 0)
    # ig_attr = process_attribution(ig_attr)
    # np.save(file_paths["integrated_gradients"], ig_attr)
    #
    # # 4. Layer-wise Relevance Propagation (approximated via LayerConductance)
    # lrp_attr = lc.attribute(input_img, target=labels)
    # lrp_attr = process_attribution(lrp_attr)
    # np.save(file_paths["lrp"], lrp_attr)
    #
    # # 5. SHAP (GradientShap)
    # shap_attr = gs.attribute(
    #     input_img, baselines=torch.zeros_like(input_img), target=labels
    # )
    # shap_attr = process_attribution(shap_attr)
    # np.save(file_paths["shap"], shap_attr)
    #
    # # Prepare original image for visualization
    # original_image = images[0].detach().cpu().numpy().transpose(1, 2, 0)
    # original_image = original_image * np.array([0.2471, 0.2435, 0.2616]).reshape(
    #     1, 1, 3
    # ) + np.array([0.4914, 0.4822, 0.4465]).reshape(
    #     1, 1, 3
    # )  # Denormalize correctly
    # np.save(file_paths["original_image"], original_image)

    print("Need to compute first!")

# Define colormaps to test
# colormaps = ["viridis", "hot", "inferno", "cividis", "magma"]
colormaps = ["viridis"]

# Generate figures for each colormap
for cmap in colormaps:
    # Create figure with adjusted size and spacing for single colorbar
    fig, axes = plt.subplots(1, 6, figsize=(35, 6))

    # Adjust spacing between subplots to make room for single colorbar
    plt.subplots_adjust(wspace=0.1, right=1.0)

    # Plot original image
    axes[0].imshow(original_image)
    axes[0].set_title("(a) Original Image")
    axes[0].axis("off")

    # Plot attribution maps
    attributions = [
        (vanilla_grads, "(b) Vanilla Gradients"),
        (smoothgrad_attr, "(c) SmoothGrad"),
        (ig_attr, "(d) IG"),
        (lrp_attr, "(e) LRP (Approx)"),
        (shap_attr, "(f) SHAP"),
    ]

    # Store the last image for colorbar reference
    im = None
    for i, (attr, title) in enumerate(attributions):
        # Create the image plot
        im = axes[i + 1].imshow(attr, cmap=cmap, vmin=0, vmax=1)
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    # Add single colorbar on the right side
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, aspect=10, pad=0.01)
    cbar.set_label("Attribution Strength", rotation=270, labelpad=40, fontsize=36)
    cbar.ax.tick_params(labelsize=20)

    # plt.tight_layout()
    plt.savefig(f"Hernandez2025_attribution_comparison_{cmap}.pdf")
    plt.close()

print("Figures generated for colormaps:", colormaps)
