import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import attention_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using: ", device)

BATCH_SIZE = 64
NUM_WORKERS = 16
divider = 20

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


class ModifiedTeacher(nn.Module):
    def __init__(self, original_model):
        super(ModifiedTeacher, self).__init__()
        # Divide the model into two parts around the middle layer
        middle_index = len(original_model.features) // divider
        self.front_layers = nn.Sequential(*original_model.features[:middle_index])
        self.middle_layer = original_model.features[middle_index]
        self.end_layers = nn.Sequential(*original_model.features[middle_index + 1 :])
        self.classifier = nn.Sequential(*original_model.classifier)

    def forward(self, x):
        x = self.front_layers(x)
        middle_feature_maps = self.middle_layer(x)
        attention_maps = attention_map(middle_feature_maps)
        x = self.end_layers(middle_feature_maps)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x, attention_maps


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
# np.save("./data/cifar10_logits_attn.npy", all_logits)
np.save(
    f"./data/cifar10_attention_maps_divider_{divider}.npy", all_attention_maps
)  # Saving attention maps
