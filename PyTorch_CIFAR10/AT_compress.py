import torch
import torch.nn as nn

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import attention_map, count_parameters

device = "mps" if torch.backends.mps.is_available() else "cpu"


class ModifiedStudent(nn.Module):
    def __init__(self, original_model, layer, divider=2):
        super(ModifiedStudent, self).__init__()
        middle_index = len(original_model.features) // divider
        self.front_layers = nn.Sequential(*original_model.features[:middle_index])
        self.middle_layer = original_model.features[middle_index]
        self.end_layers = nn.Sequential(
            *list(original_model.features.children())[middle_index + 1 : -layer]
        )

        # Dynamically find the output channels of the last convolutional block used
        # Iterate backwards over the blocks to find the last convolutional layer
        for block in reversed(self.end_layers):
            if hasattr(block, "conv"):
                # If the block contains a 'conv' attribute, likely to be a sequential module
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
            nn.Flatten(), nn.Linear(num_output_channels, 10)
        )

    def forward(self, x):
        x = self.front_layers(x)
        middle_feature_maps = self.middle_layer(x)
        attention_maps = attention_map(middle_feature_maps)
        x = self.end_layers(middle_feature_maps)
        x = self.pool(x)
        x = self.classifier(x)
        return x, attention_maps


# List of layer configurations to test
LAYERS = [3, 5, 7, 9, 11, 13, 15, 17]
DIVIDERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

for LAYER in LAYERS:
    # Load teacher model and count its parameters
    teacher = mobilenet_v2(pretrained=True)
    teacher.to(device)
    t = count_parameters(teacher)

    if LAYER < 9:
        try:
            student = ModifiedStudent(mobilenet_v2(pretrained=True), LAYER)
            student.to(device)
            s = count_parameters(student)
            CF = t / s
            print(f"Compression Factor (CF): {CF}, using middle_layer_index: 2")

        except Exception as e:
            continue  # Skip to the next LAYER value
    else:
        created = False
        for divider in DIVIDERS:
            try:
                student = ModifiedStudent(mobilenet_v2(pretrained=True), LAYER, divider)
                student.to(device)

                s = count_parameters(student)
                CF = t / s
                print(
                    f"Compression Factor (CF): {CF}, using middle_layer_index: {divider}"
                )

                created = True

                break  # Exit the loop once a valid divider is found
            except Exception as e:
                continue

        if not created:
            print(f"Cannot create model with {LAYER} layers after trying all dividers.")
            continue  # Skip to the next LAYER value
