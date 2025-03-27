import torch
from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import ModifiedStudent, count_parameters

if __name__ == "__main__":

    device = "mps" if torch.backends.mps.is_available() else "cpu"

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
                    student = ModifiedStudent(
                        mobilenet_v2(pretrained=True), LAYER, divider
                    )
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
                print(
                    f"Cannot create model with {LAYER} layers after trying all dividers."
                )
                continue  # Skip to the next LAYER value
