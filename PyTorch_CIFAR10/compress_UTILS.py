import time

import pandas as pd
import torch  # type: ignore
from torch import nn  # type: ignore
from tqdm import tqdm

from UTILS_TORCH import kd_loss


def test_model(model, loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Accuracy of the model on the Testing images: {test_acc:.2f}")

    return test_acc


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


def train_eval_kd(
    student,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.001,
    TEMP=2.0,
    ALPHA=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    csv_path="Compression_accuracy_time_test/test1.csv",
):

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # Store the metrics
    metrics = []
    for epoch in range(epochs):

        epoch_start_time = time.time()
        student.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Wrap the DataLoader with tqdm for progress tracking
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        )
        for images, labels, teacher_logits in train_loader:
            images, labels, teacher_logits = (
                images.to(device),
                labels.to(device),
                teacher_logits.to(device),
            )

            optimizer.zero_grad()

            student_logits = student(images)

            distillation_loss = kd_loss(
                logits_student=student_logits,
                logits_teacher=teacher_logits,
                temperature=TEMP,
            )
            student_loss = criterion(student_logits, labels)
            loss = ALPHA * distillation_loss + (1 - ALPHA) * student_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(student_logits, dim=1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # Update tqdm bar with the latest loss and accuracy
            train_loader_tqdm.set_postfix(
                loss=f"{loss.item():.4f}",
                accuracy=f"{(100 * correct_train / total_train):.2f}%",
            )
            train_loader_tqdm.update()
        epoch_loss = total_train_loss / total_train
        epoch_acc = 100 * correct_train / total_train

        epoch_time = time.time() - epoch_start_time

        inference_start_time = time.time()
        test_acc = test_model(student, test_loader, device)
        inference_time = time.time() - inference_start_time

        # Append metrics for the current epoch to the list
        metrics.append(
            {
                "Epoch": epoch + 1,
                "Training Loss": epoch_loss,
                "Training Accuracy": epoch_acc,
                "Testing Accuracy": test_acc,
                "Epoch Time (s)": epoch_time,
                "Inference Time (s)": inference_time,
            }
        )

        train_loader_tqdm.close()
        print(
            f"Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f}, "
            f"Accuracy: {epoch_acc:.2f}%, "
            f"Epoch Time: {epoch_time:.2f}s, "
            f"Inference Time: {inference_time:.2f}s"
        )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(csv_path, index=False)

    test_acc = test_model(student, test_loader, device)

    return student, test_acc, metrics_df
