import torch
import torch.nn as nn

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import count_parameters

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently using: '{device}'")

# This list is used for selecting specific layer indices later
LAYERS = [3, 5, 7, 9, 11, 13, 15, 17]

# Load the teacher model with pretrained weights and send it to the device
Teacher = mobilenet_v2(pretrained=True)
Teacher.to(device)
teacher_params = count_parameters(Teacher)


class SmallerMobileNet(nn.Module):
    def __init__(self, original_model, layer):
        super(SmallerMobileNet, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-layer]
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
            nn.Linear(num_output_channels, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# Function to print the data type and memory usage of each parameter in a model
def print_model_total_memory_info(model, model_name="Model"):
    total_bytes = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    total_kb = total_bytes / 1024
    total_mb = total_bytes / (1024 * 1024)
    print(
        f"{model_name} total memory requirement: ({total_kb:.2f} KB / {total_mb:.2f} MB)"
    )


# Print memory information for the entire teacher and smaller models
print_model_total_memory_info(Teacher, "Teacher Model")

for LAYER in LAYERS:
    # Instantiate the smaller model and send it to the device
    test = SmallerMobileNet(mobilenet_v2(pretrained=False), LAYER)
    test.to(device)
    test_params = count_parameters(test)
    CF = teacher_params / test_params

    print_model_total_memory_info(test, f"Model (CF={CF:.2f})")
