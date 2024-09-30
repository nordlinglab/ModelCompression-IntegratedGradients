import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd
import torch  # type: ignore
from torch import nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from torchvision import datasets, transforms
from tqdm import tqdm


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
    loss_kd = nn.KLDivLoss(reduction="batchmean")(
        log_pred_student,
        pred_teacher,
    ) * (temperature**2)
    # loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    # loss_kd *= temperature**2
    return loss_kd


def test_model(model, loader, device, split_name="Training"):
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

        print(
            f"Accuracy of the model on the {split_name} images: {(100 * correct / total)}"
        )


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
    attention = F.normalize(feature_maps.pow(p).mean(1), dim=[1, 2])
    return attention


def test_model_with_attention(model, loader, device):
    """
    Function to test the model on a given loader and device.
    """
    model.eval()
    attention_maps = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _, attn_maps = model(
                images
            )  # Assuming your model returns outputs and attention maps
            attention_maps.append(attn_maps.cpu())  # Store attention maps
    return attention_maps


def plot_attention_maps(attention_maps, title, save_as="attention_maps.pdf"):
    """
    Function that plots the attention_maps.
    """
    _, axs = plt.subplots(nrows=1, ncols=len(attention_maps), figsize=(15, 5))
    for i, attn_map in enumerate(attention_maps):
        if len(attention_maps) > 1:
            ax = axs[i]
        else:
            ax = axs
        img = ax.imshow(attn_map.squeeze(), cmap="hot", interpolation="nearest")
        ax.axis("off")
    plt.suptitle(title)
    plt.colorbar(img, ax=axs, orientation="horizontal", fraction=0.025, pad=0.04)
    plt.savefig(save_as)


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


def test_model_att(model, loader, device, split_name="Testing"):
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

    print(
        f"Accuracy of the model on the {split_name} images: {100 * correct / total} %"
    )


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


def visualize_sample(dataset, index):
    """Function to visualise the samples of a dataset"""
    img, ig, augmented_img, _ = dataset[index]

    # Unnormalize images for display
    # img = unnormalize(img)
    print(img.max(), img.min())
    # augmented_img = unnormalize(augmented_img)
    # print(augmented_img.min(), augmented_img.max())
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Original Image", "Integrated Gradient", "Augmented Image"]
    items = [img, ig, augmented_img]

    for ax, item, title in zip(axes, items, titles):
        item = item.permute(1, 2, 0)  # Adjust dimensions for imshow
        item = item.clip(0, 1)
        ax.imshow(item.numpy())
        ax.set_title(title)
        ax.axis("off")

    plt.show()


# Classes
class SmallerMobileNet(nn.Module):
    def __init__(self, original_model):
        super(SmallerMobileNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.features.children())[:-5])

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
    def __init__(self, original_model):
        super(ModifiedTeacher, self).__init__()
        # Divide the model into two parts around the middle layer
        middle_index = len(original_model.features) // 2
        self.front_layers = nn.Sequential(*original_model.features[:middle_index])
        self.middle_layer = original_model.features[middle_index]
        self.end_layers = nn.Sequential(*original_model.features[middle_index + 1 :])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = original_model.classifier

    def forward(self, x):
        x = self.front_layers(x)
        middle_feature_maps = self.middle_layer(x)
        attention_maps = attention_map(middle_feature_maps)
        x = self.end_layers(middle_feature_maps)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, attention_maps


class ModifiedStudent(nn.Module):
    def __init__(self, original_model):
        super(ModifiedStudent, self).__init__()
        middle_index = len(original_model.features) // 2
        self.front_layers = nn.Sequential(*original_model.features[:middle_index])
        self.middle_layer = original_model.features[middle_index]
        self.end_layers = nn.Sequential(
            *list(original_model.features.children())[middle_index + 1 : -5]
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
    csv_path="Histories/KD_param/KD.csv",
):

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # Store the metrics
    metrics = []
    for epoch in range(epochs):
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
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    return student, test_acc


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
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # Store the metrics
    metrics = []
    for epoch in range(epochs):
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

            attn_loss = mse_loss(teacher_attn, student_attn)

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
    student.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = student(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

    return student, test_acc


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
