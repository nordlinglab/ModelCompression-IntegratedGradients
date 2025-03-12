import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import count_parameters

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def analyze_layer_structure(model):
    """Analyze and print detailed layer structure of the model."""
    print("\nLayer Structure Analysis:")
    total_layers = 0
    conv_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers += 1
            print(f"Conv2d Layer {conv_layers}:")
            print(f"  - Input channels: {module.in_channels}")
            print(f"  - Output channels: {module.out_channels}")
            print(f"  - Kernel size: {module.kernel_size}")
            print(f"  - Stride: {module.stride}")
            total_layers += 1
        elif isinstance(module, nn.Linear):
            print("Linear Layer:")
            print(f"  - Input features: {module.in_features}")
            print(f"  - Output features: {module.out_features}")
            total_layers += 1
    return total_layers, conv_layers


class Smaller(nn.Module):
    def __init__(self, original_model, removed_layers):
        super(Smaller, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-removed_layers]
        )
        for block in reversed(self.features):
            if hasattr(block, "conv"):
                if hasattr(block.conv, "__iter__"):
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


def analyze_models():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"model_analysis_{timestamp}.txt"
    sys.stdout = OutputLogger(log_file)

    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device being used: {device}")

    # Prepare a dummy input tensor (CIFAR-10: [batch, channels, height, width])
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    # Analyze teacher model
    teacher = mobilenet_v2(pretrained=True)
    teacher.to(device)
    teacher.eval()  # set model to evaluation mode
    teacher_params = count_parameters(teacher)
    print("\n=== TEACHER MODEL ANALYSIS ===")
    print(f"Total parameters: {teacher_params:,}")

    # Measure inference time for the teacher model
    with torch.no_grad():
        start_time = time.perf_counter()
        _ = teacher(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        teacher_inference_time = time.perf_counter() - start_time
    print(f"Inference Time for Teacher Model: {teacher_inference_time:.6f} seconds")

    teacher_total_layers, teacher_conv_layers = analyze_layer_structure(teacher)
    print(f"Total layers: {teacher_total_layers}")
    print(f"Convolutional layers: {teacher_conv_layers}")

    # Layers to remove for different compression levels
    layers_config = [3, 5, 7, 9, 11, 13, 15, 17]

    print("\n=== STUDENT MODELS ANALYSIS ===")
    print("\nSummary Table:")
    print("-" * 100)
    # Adding teacher model as the first row (0 layers removed, factor 1x)
    print(
        f"{'Removed Layers':<15} {'Total Params':<15} {'Compression Factor':<20} {'Total Layers':<15} {'Conv Layers':<15} {'Inference Time (s)'}"
    )
    print("-" * 100)
    print(
        f"{0:<15} {teacher_params:<15,} {1.00:<20.2f} {teacher_total_layers:<15} {teacher_conv_layers:<15} {teacher_inference_time:.6f}"
    )

    # Process each student model with layers removed
    for removed_layers in layers_config:
        student = Smaller(mobilenet_v2(pretrained=False), removed_layers)
        student.to(device)
        student_params = count_parameters(student)
        compression_factor = teacher_params / student_params

        student.eval()
        with torch.no_grad():
            start_time = time.perf_counter()
            _ = student(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            student_inference_time = time.perf_counter() - start_time

        print(f"\nStudent Model (Removed {removed_layers} layers):")
        print(f"Total parameters: {student_params:,}")
        print(f"Compression factor: {compression_factor:.2f}x")
        print(f"Inference Time: {student_inference_time:.6f} seconds")
        total_layers, conv_layers = analyze_layer_structure(student)
        print(f"Total layers: {total_layers}")
        print(f"Convolutional layers: {conv_layers}")

        # Add to summary table
        print(
            f"{removed_layers:<15} {student_params:<15,} {compression_factor:<20.2f} {total_layers:<15} {conv_layers:<15} {student_inference_time:.6f}"
        )

        del student
        # Clear CUDA cache if using CUDA (not needed for MPS)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nAnalysis completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nResults saved to: {log_file}")

    # Reset stdout
    sys.stdout = sys.stdout.terminal


if __name__ == "__main__":
    analyze_models()
