import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd
import torch  # type: ignore
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from torch import nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from torchvision import datasets, transforms
from tqdm import tqdm

from cifar10_models.mobilenetv2 import mobilenet_v2


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas.view(-1, 1, 1, 1)
    alphas_x = alphas_x.to(torch.float32)

    # Add a batch dimension to baseline and image
    baseline_x = baseline.unsqueeze(0)
    input_x = image.unsqueeze(0)
    delta = input_x - baseline_x

    # Linearly interpolate images
    images = baseline_x + alphas_x * delta
    return images


def compute_gradients(model, images, target_class_idx):
    images.requires_grad_(True)
    logits = model(images)
    # Select the logits for the target class
    probs = nn.functional.softmax(logits, dim=-1)[:, target_class_idx]

    # Gradient computation
    model.zero_grad()
    probs.backward(torch.ones_like(probs))
    return images.grad


def integral_approximation(gradients):
    # Average gradients using the trapezoidal rule
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    integrated_gradients = grads.mean(dim=0)
    return integrated_gradients


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / logits_student.shape[0]
    )
    return loss_kd


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


def test_model_att(model, loader, device):
    """
    Function to test a model with attention maps
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Accuracy of the model on the Testing images: {test_acc:.2f}")

    return test_acc


def random_ig_overlay(image, ig, prob=0.5):
    if np.random.rand() < prob:
        overlay_alpha = np.random.rand()  # Random alpha for blending
        ig_tensor = torch.tensor(ig, dtype=torch.float32)

        # Expand the 2D IG tensor to 3 channels
        if len(ig_tensor.shape) == 2:
            ig_tensor = ig_tensor.unsqueeze(0).repeat(3, 1, 1)

        ig_tensor = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
        )(ig_tensor)
        # image = (1 - overlay_alpha) * image + overlay_alpha * ig_tensor
        image = image + overlay_alpha * ig_tensor
        image = torch.clamp(
            image, 0, 1
        )  # Ensure the pixel values are in the valid range [0, 1]
    return image


def attention_map(feature_maps, p=2):
    # Generate attention maps by summing absolute values across channels
    attention = F.normalize(feature_maps.pow(p).mean(1).view(feature_maps.size(0), -1))
    return attention


def attention_loss(ta, sa):
    return (ta - sa).pow(2).mean()


def compare_attn_maps(student_model, loader, device):
    """
    Compares attention maps between precomputed teacher data and the student model.
    Assumes the loader provides images, labels, teacher_logits, and teacher attention maps.
    """
    student_model.eval()

    # Dictionary to store attention maps for visualization
    attention_maps = {
        "teacher": {
            i: [] for i in range(10)
        },  # Store one attention map per class for the teacher
        "student": {
            i: [] for i in range(10)
        },  # Store one attention map per class for the student
    }

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    with torch.no_grad():
        for images, labels, teacher_logits, teacher_attn in loader:
            images, labels, teacher_logits, teacher_attn = (
                images.to(device),
                labels.to(device),
                teacher_logits.to(device),
                teacher_attn.to(device),
            )
            _, student_attn_maps = student_model(images)

            # Iterate over the batch to separate data by class
            for _, label, ta, sa in zip(
                images, labels, teacher_attn, student_attn_maps
            ):
                if (
                    len(attention_maps["teacher"][label.item()]) < 1
                ):  # Collect one image per class
                    attention_maps["teacher"][label.item()].append(ta.cpu())
                if (
                    len(attention_maps["student"][label.item()]) < 1
                ):  # Collect one image per class
                    attention_maps["student"][label.item()].append(sa.cpu())

    # Visualization of attention maps
    fig, axs = plt.subplots(
        2, 10, figsize=(40, 8)
    )  # Two rows and 10 columns for the plots
    for i in range(10):
        if attention_maps["teacher"][i] and attention_maps["student"][i]:
            teacher_attn_map = attention_maps["teacher"][i][
                0
            ].squeeze()  # First attention map for the class
            student_attn_map = attention_maps["student"][i][
                0
            ].squeeze()  # First attention map for the class

            axs[0, i].imshow(teacher_attn_map, cmap="hot", interpolation="nearest")
            axs[0, i].set_title(f"Teacher: {class_names[i]}", fontsize=28)
            axs[0, i].axis("off")

            axs[1, i].imshow(student_attn_map, cmap="hot", interpolation="nearest")
            axs[1, i].set_title(f"Student: {class_names[i]}", fontsize=28)
            axs[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("test_attn_map.pdf")


def random_logarithmic_scale(ig, scale_min=1, scale_max=10):
    """Apply a random logarithmic scaling to the IG tensor."""
    scale_factor = np.exp(np.random.uniform(np.log(scale_min), np.log(scale_max)))
    return ig**scale_factor


def norm(ig):
    """Function to normalise a sample.
    Returns:
    - normalised sample
    - min value in population
    - max value in population
    """
    norm_ig = (ig - ig.min()) / (ig.max() - ig.min())
    return norm_ig, ig.min(), ig.max()


def denorm(x, maximum, minimum):
    """Function to denormalise a sample given the max and min"""
    denorm_x = x * (maximum - minimum) + minimum
    return denorm_x


# Classes
class SmallerMobileNet(nn.Module):
    def __init__(self, original_model, layer=5):
        super(SmallerMobileNet, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-layer]
        )

        # Dynamically find the output channels of the last convolutional block used
        # Iterate backwards over the blocks to find the last convolutional layer
        for block in reversed(self.features):
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
            nn.Flatten(),
            nn.Linear(num_output_channels, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class CIFAR10WithIG(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train,
        transform,
        igs,
        overlay_prob=0.5,
        return_ig=False,
        precomputed_logits=None,
        precomputed_attn=None,
    ):
        super().__init__(root=root, train=train, transform=transform, download=True)
        self.igs = torch.tensor(igs, dtype=torch.float32).unsqueeze(
            1
        )  # Add channel dimension
        self.overlay_prob = overlay_prob
        self.return_ig = return_ig
        self.precomputed_logits = torch.tensor(precomputed_logits, dtype=torch.float32)
        self.precomputed_attn = precomputed_attn

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        ig = self.igs[index]
        logits = self.precomputed_logits[index]

        ig = random_logarithmic_scale(ig, 1, 2)
        ig, _, _ = norm(ig)
        ig = ig.repeat(1, 3, 1, 1).squeeze(0)  # Match RGB channels

        if np.random.rand() < self.overlay_prob:
            # Ensure overlay doesn't exceed [0, 1]
            # img, min, max = norm(img)
            augmented_img = 0.5 * img + 0.5 * ig
            # augmented_img = denorm(x = augmented_img, max=max, min=min)
            # img = denorm(img, max=max, min=min)
        else:
            augmented_img = img

        if self.precomputed_attn is not None:
            attn = self.precomputed_attn[index]
            return augmented_img, target, logits, attn

        if self.return_ig:
            return img, ig, augmented_img, target
        return augmented_img, target, logits


class ModifiedTeacher(nn.Module):
    def __init__(self, original_model, divider=2):
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


class ModifiedStudent(nn.Module):
    def __init__(self, original_model, layer=5, divider=2):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def train_eval_AT(
    student,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.001,
    TEMP=2.0,
    ALPHA=0.5,
    GAMMA=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    csv_path="Histories/KD_param/KD.csv",
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
        for images, labels, teacher_logits, teacher_attn in train_loader:
            images, labels, teacher_logits, teacher_attn = (
                images.to(device),
                labels.to(device),
                teacher_logits.to(device),
                teacher_attn.to(device),
            )

            optimizer.zero_grad()

            student_logits, student_attn = student(images)

            distillation_loss = kd_loss(
                logits_student=student_logits,
                logits_teacher=teacher_logits,
                temperature=TEMP,
            )

            attn_loss = attention_loss(teacher_attn, student_attn)

            student_loss = criterion(student_logits, labels)
            loss = (
                ALPHA * distillation_loss
                + (1 - ALPHA) * student_loss
                + GAMMA * attn_loss
            )

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
        test_acc = test_model_att(student, test_loader, device)
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

    test_acc = test_model_att(student, test_loader, device)

    return student, test_acc, metrics_df


class CIFAR10_KD(datasets.CIFAR10):
    def __init__(self, root, train, transform, precomputed_logits):
        super().__init__(root=root, train=train, transform=transform, download=True)
        self.precomputed_logits = torch.tensor(precomputed_logits, dtype=torch.float32)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        logits = self.precomputed_logits[idx]
        return image, label, logits


class CIFAR10_AT(datasets.CIFAR10):
    def __init__(self, root, train, transform, precomputed_logits, precomputed_attn):
        super().__init__(root=root, train=train, transform=transform, download=True)
        self.precomputed_logits = torch.tensor(precomputed_logits, dtype=torch.float32)
        self.precomputed_attn = torch.tensor(precomputed_attn, dtype=torch.float32)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        logits = self.precomputed_logits[idx]
        attn = self.precomputed_attn[idx]
        return image, label, logits, attn


def evaluate_model_performance(model, dataloader, device, num_classes=10, attn=False):
    """
    Evaluate the model performance on a dataset with unbalanced classes.

    Parameters:
    model: torch.nn.Module - The trained model to evaluate.
    dataloader: torch.utils.data.DataLoader - The dataloader containing the test/validation data.
    device: torch.device - The device to perform computation on (e.g., 'cuda' or 'cpu').
    num_classes: int - The number of classes in the dataset (default is 10 for CIFAR-10).

    Returns:
    dict: A dictionary containing class-wise accuracy, F1-score, balanced accuracy, Cohen's Kappa, and confusion matrix.
    """
    model.eval()  # Set model to evaluation mode

    # Track correct predictions and total samples per class
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    all_preds = []
    all_labels = []

    # No gradient calculation is needed during evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            if attn:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Store all predictions and labels for later metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate class-wise accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1

    # Compute class accuracy
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0  # Avoid division by zero

    # Compute other metrics using sklearn
    f1_report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        output_dict=True,
        zero_division=0,  # Avoid warnings by handling divisions by zero
    )
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Output all metrics in a structured dictionary
    metrics = {
        "class_accuracy": class_accuracy,
        "f1_report": f1_report,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "confusion_matrix": conf_matrix,
    }

    return metrics


def evaluate_and_drop_samples(model, loader, device):
    correct_indices = []
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No need to track gradients
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (
                (predicted == labels).nonzero().squeeze()
            )  # Get correct indices in batch

            # Convert batch indices to dataset indices
            dataset_indices = [i.item() for i in (correct + idx * loader.batch_size)]
            correct_indices.extend(dataset_indices)

    return correct_indices


def train_eval(
    model,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.001,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    csv_path="Histories/Training.csv",
):
    """
    Train and evaluate a model for regular supervised learning.

    Parameters:
        model: The neural network model to be trained and evaluated.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        device: Device to run the training on ('cuda' or 'cpu').
        csv_path: Path to save the training history as a CSV file.

    Returns:
        model: The trained model.
        test_acc: Test accuracy of the trained model.
    """

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Store the metrics
    metrics = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Wrap the DataLoader with tqdm for progress tracking
        train_loader_tqdm = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        )
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
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

        # Append metrics for the current epoch to the list
        metrics.append(
            {
                "Epoch": epoch + 1,
                "Training Loss": epoch_loss,
                "Training Accuracy": epoch_acc,
            }
        )

        train_loader_tqdm.close()
        print(
            f"Epoch {epoch+1}/{epochs}: Loss: {epoch_loss:.4f},\
            Accuracy: {epoch_acc:.2f}%"
        )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(csv_path, index=False)

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    return model, test_acc


# Function to extract final accuracy from CSV
def extract_final_accuracy(filepath):
    try:
        df = pd.read_csv(filepath)
        if "Training Accuracy" in df.columns and "Testing Accuracy" in df.columns:
            # Get the last row's training and testing accuracy
            last_row = df.iloc[-1]
            return last_row["Training Accuracy"], last_row["Testing Accuracy"]
        else:
            print(f"Warning: Columns not found in {filepath}")
            return None, None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


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


def analyze_models(
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
):
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
        student = SmallerMobileNet(mobilenet_v2(pretrained=False), removed_layers)
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
