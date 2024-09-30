import pickle

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import transforms

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")


# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 4


# Assuming your data is in CIFAR-like format (data is flattened)
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def parse_class_map(file_path):
    """
    Parses the class map file to get a dictionary mapping ImageNet class ID to the class index.
    """
    class_map = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 2)  # Split by first two spaces
            class_id = parts[0]  # ImageNet ID (e.g., n01440764)
            class_index = int(parts[1])  # Corresponding integer index (e.g., 0)
            class_map[class_id] = class_index
    return class_map


# Load the class map from map_clsloc.txt
class_map = parse_class_map("./data/map_clsloc.txt")

# Step 2: List of CIFAR-10-like classes from ImageNet
desired_class_ids = [
    "n02690373",  # airliner
    # "n04285008",  # sports_car
    # "n01514668",  # cock
    # "n02124075",  # Egyptian_cat
    # "n02423022",  # gazelle
    # "n02110185",  # Siberian_husky
    # "n01644900",  # tailed_frog
    # "n02437312",  # Arabian_camel (closest to horse)
    # "n04612504",  # yacht
    # "n04467665",  # trailer_truck
]

# Handle missing class IDs
missing_class_ids = []

# Step 2: Map desired class IDs to indices
desired_class_indices = []
for class_id in desired_class_ids:
    if class_id in class_map:
        desired_class_indices.append(class_map[class_id])
    else:
        missing_class_ids.append(class_id)

# Check for missing class IDs and log them
if missing_class_ids:
    print(
        f"Warning: The following class IDs were not found in the class map: {missing_class_ids}"
    )
else:
    print("All desired class IDs found in the class map.")

# Load the unpickled data
imagenet = unpickle("./data/val_data")

# Extract labels and data
labels = imagenet["labels"]
data = imagenet["data"]  # Assuming the data is in a flattened format

# Filter data based on desired classes
filtered_indices = [
    i for i, label in enumerate(labels) if label in desired_class_indices
]
filtered_data = data[filtered_indices]
filtered_labels = [labels[i] for i in filtered_indices]

# Here, 32x32x3 images are stored as rows in the data matrix
num_images = len(filtered_data)
reshaped_data = filtered_data.reshape(
    num_images, 3, 32, 32
)  # Change the shape to [num_images, 3, 32, 32]


# Define the transforms for the dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        ),
    ]
)

reshaped_data = reshaped_data.transpose(0, 2, 3, 1)

# Convert to PyTorch tensors
# images_tensor = torch.tensor(reshaped_data, dtype=torch.float32)
images_tensor = torch.stack([transform(img) for img in reshaped_data])
labels_tensor = torch.tensor(filtered_labels, dtype=torch.long)

# Normalize the image data to [0, 1] (assuming pixel values are in [0, 255])
# images_tensor /= 255.0

# Optionally, create a dataset and dataloader
dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# # Iterate over the data
# for images, labels in dataloader:
#     print(f"Batch image size: {images.size()}")  # [64, 3, 32, 32] for a batch of 64
#     plt.imshow(images[3].detach().numpy().transpose(1, 2, 0))
#     plt.show()
#     break  # Just to show the first batch


model = mobilenet_v2(pretrained=True)
model.to(device)

full_training_ds = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)
# Load the data into batches
dataloader = torch.utils.data.DataLoader(
    full_training_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    # pin_memory=False,
    # persistent_workers=True,
)

# for images, labels in dataloader:
#     outputs = model(images)
#     _, predicted = torch.max(outputs.data, 1)
#
# print(predicted[:40], labels[:10])
test_model(model=model, loader=dataloader, device=device)
