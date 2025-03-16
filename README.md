# ModelCompression-IntegratedGradients

Repository for the Model compression integrated gradients project

## Model Compression with Knowledge Distillation and Integrated Gradients

### Overview

This repository contains the implementation of our novel approach to model compression by integrating knowledge distillation (KD) with Integrated Gradients (IG) and attention transfer (AT), as described in our paper. This method is designed to enhance both performance and interpretability of compressed deep learning models.

### Repository Structure

```
.
├── .git                            # Git repository metadata
├── .gitignore                      # Git ignore file
├── PyTorch_CIFAR10/                # Scripts and models for CIFAR-10 dataset
│   ├── analyse_models.py           # Script to analyze model architectures and performance
│   ├── Compare_attention_maps.py   # Script for comparing attention maps between teacher and student models
│   ├── compress_acc.py             # Script for compression vs. accuracy analysis
│   ├── Compute_IGs.py              # Script to compute Integrated Gradients
│   ├── main.py                     # Main script for training various configurations
│   ├── model_train.py              # Script for training student models from scratch
│   ├── precompute_img_attn.py      # Script to precompute teacher's logits and attention maps
│   ├── UTILS_TORCH.py              # Utility functions and classes for PyTorch implementation
│   ├── cifar10_models/             # Teacher model scripts and weights
│   │   ├── mobilenetv2.py          # Script for the teacher model
│   │   └── state_dicts/            # Pre-trained model weights
│   │       └── mobilenet_v2.pt     # Weights for the teacher model
├── LICENSE                         # License file
└── README.md                       # This README file
```

### Key Components

- **analyse_models.py**: Analyzes and compares model architectures, parameters, and inference time for different compression levels.
- **Compare_attention_maps.py**: Implements attention map comparison between teacher and student models under different training configurations.
- **compress_acc.py**: Evaluates the relationship between compression factor and model accuracy.
- **Compute_IGs.py**: Computes Integrated Gradients for CIFAR-10 images using both manual implementation and Captum library.
- **main.py**: Main script for training student models with different configurations (IG, KD, AT, and combinations).
- **model_train.py**: Trains student models from scratch without knowledge distillation.
- **precompute_img_attn.py**: Pre-computes teacher model's logits and attention maps for efficient training.
- **UTILS_TORCH.py**: Contains utility functions and classes, including model architectures, custom datasets, and training loops.

### Installation

#### Prerequisites

- Docker: Ensure Docker is installed on your system. Docker will be used to create an isolated environment that contains all the necessary dependencies. Visit [Docker's website](https://www.docker.com/get-started) for installation instructions.

#### Setting Up the Docker Container

To use the scripts, you will need to create a Docker container from the provided image. This image includes the basic environment and dependencies:

1. Pull the Docker Image

```
docker pull pytorch/pytorch
```

2. Run the docker container

```
docker run -it -v absolute/path/to/ModelCompression-IntegratedGradients:/workspace/ModelCompression-IntegratedGradients -p 8888:8888 --gpus all --ipc host --name <container name> <image name or id> bash
```

This command runs the container in interactive mode and starts a Bash shell.
It maps port 8888 for Jupyter notebooks, enables GPU support (if available), and sets up IPC namespace settings for PyTorch to perform optimally.

#### Handling Additional Dependencies

In case you encounter missing Python packages while running scripts within the Docker container, you can install these packages using `pip`. For example:

```
docker exec -it <container name> pip install <package name>
```

This command installs the required package inside the running container without needing to stop or rebuild the image.

### Usage

#### Running the Analysis Scripts

1. **Analyzing Model Architectures**:

   ```
   python PyTorch_CIFAR10/analyse_models.py
   ```

   This will analyze the teacher model and generate different student models at various compression levels, comparing parameters, inference time, and structure.

2. **Computing Integrated Gradients**:

   ```
   python PyTorch_CIFAR10/Compute_IGs.py
   ```

   This will calculate the integrated gradients for the CIFAR-10 dataset, using both a manual implementation and the Captum library.

3. **Training Student Models with Knowledge Distillation**:

   ```
   python PyTorch_CIFAR10/compress_acc.py
   ```

   This script trains student models with knowledge distillation and reports accuracy at different compression factors.

4. **Training Student Models with Attention Transfer**:

   ```
   python PyTorch_CIFAR10/Compare_attention_maps.py
   ```

   This script trains and compares student models with attention transfer, allowing for visual comparison of attention maps.

5. **Main Training Script**:
   ```
   python PyTorch_CIFAR10/main.py
   ```
   The main script supports training with various configurations, including knowledge distillation, integrated gradients, and attention transfer.

### Methodology

Our approach combines three key techniques:

1. **Knowledge Distillation (KD)**: Transfers knowledge from a larger teacher model to a smaller student model by training the student to mimic the teacher's outputs.

2. **Integrated Gradients (IG)**: Incorporates feature attribution from the teacher model to guide the student model's training, enhancing its focus on important features.

3. **Attention Transfer (AT)**: Transfers the attention maps from the teacher model to the student model, helping it learn similar activation patterns.

### Results

The repository includes scripts to reproduce our experimental results showing that:

- Combining KD with IG and AT improves student model performance compared to using these techniques individually.
- Smaller student models can achieve comparable performance to larger teacher models by leveraging these techniques.
- The student models maintain interpretability and feature importance similar to the teacher model.

### Contributing

Contributions to this repository are welcome. Please fork the repository and submit a pull request with your proposed changes.

### License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this code or our results in your research, please cite as:
