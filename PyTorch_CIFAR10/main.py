import os
from collections import Counter
from parser import parse_args

import numpy as np
import pandas as pd
import torch
from cifar10_models.mobilenetv2 import mobilenet_v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from UTILS_TORCH import (
    CIFAR10_KD,
    CIFAR10WithIG,
    ModifiedStudent,
    ModifiedTeacher,
    SmallerMobileNet,
    count_parameters,
    train_eval_AT,
    train_eval_kd,
)


def main():
    args = parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently using: '{device}'")

    AT = ["AT", "KD_AT", "IG_AT", "KD_IG_AT"]

    # Set hyperparameters from arguments
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARN_RATE = args.learn_rate
    ALPHA = args.alpha
    TEMP = args.temp
    NUM_WORKERS = args.num_workers
    OVERLAY_PROB = args.overlay_prob
    SPLIT_SIZE = args.split_size
    START = args.start
    SIMULATIONS = args.simulations
    GAMMA = args.gamma

    print(f"Using {args.layers} layers with divider {args.divider}")

    if args.type == "Student":
        ALPHA = 0
        TEMP = 1
        OVERLAY_PROB = 0
        GAMMA = 0
    elif args.type == "KD":
        GAMMA = 0
        OVERLAY_PROB = 0.0
    elif args.type == "KD_IG":
        GAMMA = 0
    elif args.type == "IG":
        GAMMA = 0
        ALPHA = 0
        TEMP = 1
    elif args.type == "AT":
        ALPHA = 0
        TEMP = 1
        OVERLAY_PROB = 0.0
    elif args.type == "KD_AT":
        OVERLAY_PROB = 0.0
    elif args.type == "IG_AT":
        ALPHA = 0
        TEMP = 1

    if args.type in AT:
        T = ModifiedTeacher(mobilenet_v2(pretrained=True), args.divider)
        test_model = ModifiedStudent(
            mobilenet_v2(pretrained=False), args.layers, args.divider
        )
    else:
        T = mobilenet_v2(pretrained=True)
        test_model = SmallerMobileNet(mobilenet_v2(pretrained=False), args.layers)

    T.to(device)

    teacher_params = count_parameters(T)

    test_model.to(device)

    S = count_parameters(test_model)
    CF = teacher_params / S

    # Set folders and paths
    FOLDER = args.folder
    os.makedirs(os.path.dirname(FOLDER), exist_ok=True)

    # Set save path
    if args.save_path:
        SAVE = args.save_path
    else:
        SAVE = f"{args.folder}{args.type}_{CF:.2f}.csv"

    print("Saving results as: ", SAVE)
    print("Number of runs for the simulation: ", SIMULATIONS)

    teacher_logits = np.load("./data/cifar10_logits.npy")
    teacher_attention_maps = None

    if args.type in AT:
        if args.divider == 2:
            teacher_attention_maps = np.load("./data/cifar10_attention_maps.npy")
        else:
            teacher_attention_maps = np.load(
                f"./data/cifar10_attention_maps_divider_{args.divider}.npy"
            )

    IGS = "./data/Captum_IGs.npy"
    print(f"Using IGs: {IGS}")

    print("Hyperparams:")
    print(
        f"num_epochs = {NUM_EPOCHS}, batch_size = {BATCH_SIZE}, \
        learning_rate = {LEARN_RATE}, alpha = {ALPHA}, \
        temperature = {TEMP}, num_workers = {NUM_WORKERS},  \
        overlay_prob = {OVERLAY_PROB}, gamma = {GAMMA}"
    )

    # Load the precomputed IGs
    igs = np.load(IGS)
    print(f"IGs shape: {igs.shape}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    student_aug = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # Define the dataset based on the type
    if args.type == "Student":
        full_training_ds = CIFAR10_KD(
            root="./data",
            train=True,
            transform=student_aug,
            precomputed_logits=teacher_logits,
        )
    else:
        full_training_ds = CIFAR10WithIG(
            igs=igs,
            root="./data",
            train=True,
            transform=student_aug,
            overlay_prob=OVERLAY_PROB,
            return_ig=False,
            precomputed_logits=teacher_logits,
            precomputed_attn=(teacher_attention_maps if args.type in AT else None),
        )

    train_loader = DataLoader(
        full_training_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=True,
    )

    # load student data
    test_data = CIFAR10(
        root="./data",
        train=False,
        transform=transform,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=True,
    )

    # main loop
    if os.path.isfile(SAVE):
        csv = pd.read_csv(SAVE)
        results = csv["Test Accuracy"]
        results = list(results)
        print(results)
        best_acc = np.array(results)
        best_acc = best_acc.max()
        print(best_acc)
    else:
        results = []
        best_acc = 0
    for i in range(START, SIMULATIONS):
        # Step 1: Get the class indices
        class_indices = {
            i: np.where(np.array(full_training_ds.targets) == i)[0] for i in range(10)
        }

        if args.split_size:
            # Step 2: Stratified sampling of indices
            train_indices = []
            for class_idx, indices in class_indices.items():
                train_idx, _ = train_test_split(
                    indices, train_size=SPLIT_SIZE, random_state=None
                )  # None for true randomness each run
                train_indices.extend(train_idx)

            # Step 3: Shuffle the training indices
            np.random.shuffle(train_indices)

            # Step 4: Create a subset and DataLoader for the training data
            train_subset = Subset(full_training_ds, train_indices)

            # Load the data into batches
            train_loader = DataLoader(
                train_subset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=False,
                persistent_workers=True,
            )

            subset_labels = [full_training_ds.targets[i] for i in train_indices]
            print("Class distribution in subset:", Counter(subset_labels))

        # Create the appropriate student model
        if args.type in AT:
            Student = ModifiedStudent(
                mobilenet_v2(pretrained=False), args.layers, args.divider
            )
        else:
            Student = SmallerMobileNet(mobilenet_v2(pretrained=False), args.layers)

        Student.to(device)

        if args.type in AT:
            model, acc, _ = train_eval_AT(
                student=Student,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=NUM_EPOCHS,
                lr=LEARN_RATE,
                TEMP=TEMP,
                ALPHA=ALPHA,
                GAMMA=GAMMA,
                device=device,
                csv_path=f"{FOLDER}{i+1}.csv",
            )
        else:
            model, acc, _ = train_eval_kd(
                student=Student,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=NUM_EPOCHS,
                lr=LEARN_RATE,
                TEMP=TEMP,
                ALPHA=ALPHA,
                device=device,
                csv_path=f"{FOLDER}{i+1}.csv",
            )

        results.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{FOLDER}/{args.type}.pt")
        print(f"Simulation [{i+1}/{SIMULATIONS}]: Test Acc = {acc:.2f}%")
        del Student
        torch.cuda.empty_cache()
        print("Saving simulation")
        print(f"Best Accuracy: {best_acc}")

        data = pd.DataFrame(results, columns=["Test Accuracy"])
        data.to_csv(SAVE)


if __name__ == "__main__":
    main()
