import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import ModifiedTeacher

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using: ", device)

BATCH_SIZE = 64
NUM_WORKERS = 6

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

teacher_train_data = CIFAR10(root="./data", train=True, transform=teacher_aug)

# Load the data into batches for the teacher
teacher_train_loader = DataLoader(
    teacher_train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=True,
)
# Teacher model (pretrained)
Teacher = ModifiedTeacher(mobilenet_v2(pretrained=True))
Teacher.to(device)
Teacher.eval()

all_logits = []
all_attention_maps = []
with torch.no_grad():
    for images, _ in teacher_train_loader:
        images = images.to(device)
        logits, attn_maps = Teacher(images)
        all_logits.append(logits.cpu().numpy())
        all_attention_maps.append(attn_maps.cpu().numpy())  # Collecting attention maps

all_logits = np.concatenate(all_logits, axis=0)
all_attention_maps = np.concatenate(all_attention_maps, axis=0)

print(all_attention_maps.shape)
np.save("./data/cifar10_logits_attn.npy", all_logits)
np.save(
    "./data/cifar10_attention_maps.npy", all_attention_maps
)  # Saving attention maps
