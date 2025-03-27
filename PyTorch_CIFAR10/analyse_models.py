import sys
import time
from datetime import datetime

import torch
import torch.nn as nn

from cifar10_models.mobilenetv2 import mobilenet_v2
from UTILS_TORCH import OutputLogger, SmallerMobileNet, analyze_models, count_parameters

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    analyze_models(device)
