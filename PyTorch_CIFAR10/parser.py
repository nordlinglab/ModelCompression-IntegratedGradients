import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill the models with configurable parameters."
    )
    # Keep all the existing arguments
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=100,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cuda", "mps"],
        help="Choose the device to use, e.g: cuda, mps, cpu",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=0.01,
        help="Choose the distillation loss weight.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        required=False,
        default=2.5,
        help="Choose the temperature for softening the logits.",
    )
    parser.add_argument(
        "--gamma", type=float, required=False, default=0.8, help="Choose the AT weight."
    )
    parser.add_argument(
        "--overlay_prob",
        type=float,
        required=False,
        default=0.1,
        help="Choose the overlaying probability weight.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        required=False,
        default=5,
        choices=[3, 5, 7, 9, 11, 13, 15, 17],
        help="Number of layers. Will automatically set the corresponding divider value.",
    )
    parser.add_argument(
        "--divider",
        type=int,
        required=False,
        help="Divider to use for the ModifiedTeacher architecture. If not provided, it will be set based on the layers value.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        required=False,
        default=1,
        help="Number of simulations (assigned to SIMULATIONS).",
    )
    parser.add_argument(
        "--type",
        type=str,
        required=False,
        default="Student",
        choices=["Student", "KD", "KD_IG", "IG", "KD_IG_AT", "AT", "KD_AT", "IG_AT"],
        help="Type of configuration. Options: Student, KD, KD_IG, IG, KD_IG_AT, AT, KD_AT, IG_AT",
    )
    # Remove the choices from the divider argument since we'll set it automatically
    parser.add_argument(
        "--num_workers", type=int, required=False, default=1, help="Number of workers."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learn_rate", type=float, required=False, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=False,
        default="Histories/Test/",
        help="Choose the folder to save the files in.",
    )
    parser.add_argument(
        "--start", type=int, required=False, default=0, help="Start simulation index."
    )
    parser.add_argument(
        "--split_size",
        type=float,
        required=False,
        default=None,
        help="Training split size.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        help="Path to save the results CSV file.",
    )

    args = parser.parse_args()

    # Set the divider based on the layers if not explicitly provided
    if args.divider is None:
        if args.layers in [3, 5, 7]:
            args.divider = 2
        elif args.layers in [9, 11, 13]:
            args.divider = 4
        elif args.layers == 15:
            args.divider = 8
        elif args.layers == 17:
            args.divider = 20
        else:
            # Default value if somehow the layer value is not in our mapping
            args.divider = 2

    return args
