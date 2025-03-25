import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate

from AT_compress import ModifiedStudent, ModifiedTeacher
from cifar10_models.mobilenetv2 import mobilenet_v2


class ModelSummary:
    def __init__(self, model, input_size=(1, 3, 32, 32)):
        self.model = model
        self.input_size = input_size
        self.summary = []
        self.hooks = []
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def register_hooks(self):
        def hook_fn(module, input, output):
            module_name = str(module.__class__.__name__)
            layer_idx = len(self.summary)

            # Get the module's size
            m_params = sum([p.numel() for p in module.parameters()])

            # Get output shape
            if isinstance(output, tuple):
                output_shape = str([[-1] + list(o.shape[1:]) for o in output])
            else:
                output_shape = str([-1] + list(output.shape[1:]))

            self.summary.append(
                {
                    "layer_idx": layer_idx,
                    "module_name": module_name,
                    "params": m_params,
                    "output_shape": output_shape,
                }
            )

        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Sequential) and not list(module.children()):
                self.hooks.append(module.register_forward_hook(hook_fn))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def get_summary(self):
        self.register_hooks()
        x = torch.zeros(self.input_size).to(self.device)
        try:
            self.model.eval()
            if hasattr(self.model, "forward"):
                if "Modified" in self.model.__class__.__name__:
                    _ = self.model(x)
                else:
                    _ = self.model(x)
        except Exception as e:
            print(f"Error during forward pass: {e}")

        self.remove_hooks()
        return self.summary


def save_model_architecture_details(model, model_name, output_file):
    """Save detailed architecture information for a model to a file"""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"MODEL: {model_name}")
    output.append(f"{'='*80}")

    # High-level module structure
    output.append("\nHigh-level module structure:")
    for name, module in model.named_children():
        output.append(f"{name}: {module.__class__.__name__}")

    # Get detailed layer info
    summary = ModelSummary(model).get_summary()

    # Prepare table data
    table_data = []
    total_params = 0

    for i, layer in enumerate(summary):
        params = layer["params"]
        total_params += params
        table_data.append(
            [i, layer["module_name"], layer["output_shape"], f"{params:,}"]
        )

    # Add table to output
    output.append("\nDetailed layer information:")
    output.append(
        tabulate(
            table_data,
            headers=["Layer", "Type", "Output Shape", "Params"],
            tablefmt="grid",
        )
    )

    output.append(f"\nTotal parameters: {total_params:,}")

    # Write to file
    with open(output_file, "a") as f:
        f.write("\n".join(output))
        f.write("\n\n")

    print(f"Model details for {model_name} saved to {output_file}")


def analyze_attention_map_generation(
    teacher_model, student_model, divider, output_file
):
    """Analyze where attention maps are generated and save to file"""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"ATTENTION MAP ANALYSIS (Divider={divider})")
    output.append(f"{'='*80}")

    # For teacher model
    teacher_middle_index = len(teacher_model.features) // divider
    output.append(
        f"\nTeacher middle layer (attention map source) - Index: {teacher_middle_index}"
    )
    output.append(f"Layer: {teacher_model.features[teacher_middle_index]}")

    # For student model, assuming same divider
    student_middle_index = teacher_middle_index  # Same logic as in ModifiedStudent
    if (
        hasattr(student_model, "features")
        and len(student_model.features) > student_middle_index
    ):
        output.append(
            f"\nStudent middle layer (attention map source) - Index: {student_middle_index}"
        )
        output.append(f"Layer: {student_model.features[student_middle_index]}")
    else:
        output.append("\nStudent doesn't have the same middle layer structure")

    # Write to file
    with open(output_file, "a") as f:
        f.write("\n".join(output))
        f.write("\n\n")

    print(f"Attention map analysis for divider={divider} saved to {output_file}")


def main():
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    # Create output directories
    import datetime
    import os

    os.makedirs("model_summaries", exist_ok=True)

    # Output files
    teacher_file = "model_summaries/teacher_models.txt"
    student_file = "model_summaries/student_models.txt"
    modified_student_file = "model_summaries/modified_student_models.txt"
    attention_map_file = "model_summaries/attention_map_analysis.txt"

    # Clear existing files
    for file in [teacher_file, student_file, modified_student_file, attention_map_file]:
        with open(file, "w") as f:
            f.write(
                f"Model Architecture Analysis\nGenerated on: {datetime.datetime.now()}\n\n"
            )

    # Teacher model
    teacher = mobilenet_v2(pretrained=True)
    teacher.to(device)
    save_model_architecture_details(
        teacher, "Original Teacher (MobileNetV2)", teacher_file
    )

    # Specific LAYER and middle_layer_index combinations from your data
    layer_divider_pairs = [
        (3, 2),  # CF: ~2.19
        (5, 2),  # CF: ~4.12
        (7, 2),  # CF: ~7.29
        (9, 4),  # CF: ~12.04
        (11, 4),  # CF: ~28.97
        (13, 4),  # CF: ~54.59
        (15, 8),  # CF: ~139.43
        (17, 20),  # CF: ~1121.71
    ]

    # Create a summary CSV file for quick reference
    with open("model_summaries/model_comparison.csv", "w") as f:
        f.write("Model Type,Layers Removed,Divider,Compression Factor,Parameters\n")
        teacher_params = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
        f.write(f"Teacher,0,0,1.00,{teacher_params}\n")

    # Analyze regular student models first
    for layer, _ in layer_divider_pairs:
        # Regular student from compress_acc.py
        class Smaller(nn.Module):
            def __init__(self, original_model):
                super(Smaller, self).__init__()
                self.features = nn.Sequential(
                    *list(original_model.features.children())[:-layer]
                )

                # Find output channels
                num_output_channels = None
                for block in reversed(self.features):
                    if hasattr(block, "conv"):
                        if hasattr(block.conv, "__iter__"):
                            for conv_layer in reversed(block.conv):
                                if isinstance(conv_layer, nn.Conv2d):
                                    num_output_channels = conv_layer.out_channels
                                    break
                            if num_output_channels is not None:
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

        try:
            student = Smaller(mobilenet_v2(pretrained=False))
            student.to(device)
            student_params = sum(
                p.numel() for p in student.parameters() if p.requires_grad
            )
            teacher_params = sum(
                p.numel() for p in teacher.parameters() if p.requires_grad
            )
            CF = teacher_params / student_params

            save_model_architecture_details(
                student,
                f"Regular Student (Layers removed: {layer}, CF: {CF:.2f})",
                student_file,
            )

            # Add to summary CSV
            with open("model_summaries/model_comparison.csv", "a") as f:
                f.write(f"Regular Student,{layer},0,{CF:.2f},{student_params}\n")
        except Exception as e:
            print(f"Error creating Regular Student with Layers={layer}: {str(e)}")

    # Modified Student models with specified layer and divider combinations
    for layer, divider in layer_divider_pairs:
        try:
            modified_student = ModifiedStudent(
                mobilenet_v2(pretrained=False), layer, divider
            )
            modified_student.to(device)
            modified_student_params = sum(
                p.numel() for p in modified_student.parameters() if p.requires_grad
            )
            CF = teacher_params / modified_student_params

            save_model_architecture_details(
                modified_student,
                f"Modified Student with AT (Layers: {layer}, Divider: {divider}, CF: {CF:.2f})",
                modified_student_file,
            )

            # Analyze attention map generation
            analyze_attention_map_generation(
                teacher, modified_student, divider, attention_map_file
            )

            # Add to summary CSV
            with open("model_summaries/model_comparison.csv", "a") as f:
                f.write(
                    f"Modified Student,{layer},{divider},{CF:.2f},{modified_student_params}\n"
                )
        except Exception as e:
            print(
                f"Error creating Modified Student with Layers={layer}, Divider={divider}: {str(e)}"
            )

    # Modified Teacher for AT from AT_compress.py - using the dividers from your data
    for divider in [2, 4, 8, 20]:
        try:
            modified_teacher = ModifiedTeacher(mobilenet_v2(pretrained=True), divider)
            modified_teacher.to(device)

            modified_teacher_params = sum(
                p.numel() for p in modified_teacher.parameters() if p.requires_grad
            )
            CF = teacher_params / modified_teacher_params

            save_model_architecture_details(
                modified_teacher,
                f"Modified Teacher with AT (Divider: {divider}, CF: {CF:.2f})",
                teacher_file,
            )

            # Add to summary CSV
            with open("model_summaries/model_comparison.csv", "a") as f:
                f.write(
                    f"Modified Teacher,0,{divider},{CF:.2f},{modified_teacher_params}\n"
                )

            # Add attention map analysis for teacher model
            with open(attention_map_file, "a") as f:
                teacher_middle_index = len(teacher.features) // divider
                f.write(f"\n{'='*80}\n")
                f.write(
                    f"MODIFIED TEACHER ATTENTION MAP ANALYSIS (Divider={divider})\n"
                )
                f.write(f"{'='*80}\n")
                f.write(
                    f"\nTeacher middle layer (attention map source) - Index: {teacher_middle_index}\n"
                )
                f.write(f"Layer: {teacher.features[teacher_middle_index]}\n\n")
        except Exception as e:
            print(f"Error creating Modified Teacher with Divider={divider}: {str(e)}")


if __name__ == "__main__":
    main()
