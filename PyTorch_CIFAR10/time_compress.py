import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cifar10_models.mobilenetv2 import mobilenet_v2
from compress_UTILS import SmallerMobileNet, train_eval_kd
from UTILS_TORCH import CIFAR10_KD, CIFAR10WithIG, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress and evaluate accuracy script with configurable parameters."
    )
    parser.add_argument(
        "--layers",
        type=int,
        required=True,
        help="Number of layers (assigned to LAYERS).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        required=True,
        help="Number of simulations (assigned to SIMULATIONS).",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["Student", "KD", "KD_IG", "IG"],
        help="Type of configuration. Options: Student, KD, KD_IG, IG",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Assign command line parameters to variables
    LAYERS = args.layers
    SIMULATIONS = args.simulations

    # Set parameters based on the provided type
    if args.type == "Student":
        ALPHA = 0
        TEMP = 1
        OVERLAY_PROB = 0
    elif args.type == "KD":
        ALPHA = 0.01
        TEMP = 2.5
        OVERLAY_PROB = 0
    elif args.type == "KD_IG":
        ALPHA = 0.01
        TEMP = 2.5
        OVERLAY_PROB = 0.1
    elif args.type == "IG":
        ALPHA = 0
        TEMP = 1
        OVERLAY_PROB = 0.1
    else:
        # This should not happen as argparse restricts choices.
        raise ValueError("Invalid type provided.")

    TYPE = args.type

    # For debugging: print the parameters to ensure they are set correctly
    print("Using configuration:")
    print("LAYERS =", LAYERS)
    print("SIMULATIONS =", SIMULATIONS)
    print("Type =", TYPE)
    print("ALPHA =", ALPHA)
    print("TEMP =", TEMP)
    print("OVERLAY_PROB =", OVERLAY_PROB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Currently using: '{device}'")

    # Hyperparameters
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    LEARN_RATE = 0.001
    NUM_WORKERS = 16

    Teacher = mobilenet_v2(pretrained=True)
    Teacher.to(device)

    teacher_params = count_parameters(Teacher)

    test_model = SmallerMobileNet(mobilenet_v2(pretrained=False), LAYERS)
    test_model.to(device)

    S = count_parameters(test_model)
    CF = teacher_params / S

    FOLDER = "compression_time_acc/"
    os.makedirs(os.path.dirname(FOLDER), exist_ok=True)
    SAVE = f"{FOLDER}{TYPE}_{CF:.2f}.csv"
    print("Saving result as: ", SAVE)
    print("Number of runs for the simulation: ", SIMULATIONS)

    precomputed_logits = np.load("data/cifar10_logits.npy")

    IGS = "./data/Captum_IGs.npy"
    print(f"Using IGs: {IGS}")
    # Load the precomputed IGs
    igs = np.load(IGS)

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

    if args.type == "KD_IG" or args.type == "IG":
        train_dataset = CIFAR10WithIG(
            root="./data",
            train=True,
            transform=student_aug,
            precomputed_logits=precomputed_logits,
            igs=igs,
            overlay_prob=OVERLAY_PROB,
        )
    else:
        train_dataset = CIFAR10_KD(
            root="./data",
            train=True,
            transform=student_aug,
            precomputed_logits=precomputed_logits,
        )

    # Load the data into batches
    train_loader = DataLoader(
        train_dataset,
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

    # montecarlo simulation
    if os.path.isfile(SAVE):
        csv = pd.read_csv(SAVE)
        results = csv["Test Accuracy"]
        results = list(results)
        times = csv["Train Time"]
        times = list(times)
        print(results)
        best_acc = np.array(results)
        best_acc = best_acc.max()
        print(best_acc)
        START = len(results) - 1
    else:
        results = []
        times = []
        best_acc = 0
        START = 0
    for i in range(START, SIMULATIONS):
        Student = SmallerMobileNet(mobilenet_v2(pretrained=False), LAYERS)
        Student.to(device)

        train_time_start = time.time()
        Student, acc, metrics_df = train_eval_kd(
            student=Student,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=NUM_EPOCHS,
            lr=LEARN_RATE,
            TEMP=TEMP,
            ALPHA=ALPHA,
            device=device,
            csv_path=f"{FOLDER}{TYPE}_{CF:.2f}_{i+1}.csv",
        )
        train_time = time.time() - train_time_start
        times.append(train_time)
        results.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(Student.state_dict(), f"{FOLDER}{TYPE}_{CF:.2f}.pt")
            print(f"Simulation [{i+1}/{SIMULATIONS}]: Test Acc = {acc:.2f}%")
        del Student
        torch.cuda.empty_cache()
        print("Saving simulation")
        print(f"Best Accuracy: {best_acc}")

        data = {"Test Accuracy": results, "Train Time": times}
        data = pd.DataFrame(data)
        data.to_csv(SAVE)


if __name__ == "__main__":
    main()
