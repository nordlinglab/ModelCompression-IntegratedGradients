# Model Compression-Integrated Gradients

Repository for the Model compression integrated gradients project

## Model Compression with Knowledge Distillation and Integrated Gradients

### Overview

This repository contains the implementation of our novel approach to model compression by integrating knowledge distillation (KD) with Integrated Gradients (IG) and attention transfer (AT), as described in our paper. This method is designed to enhance both performance and interpretability of compressed deep learning models.

### Repository Structure

```
.
├── .git                            # Git repository metadata
├── .gitignore                      # Git ignore file
├── Jupyter_notebooks/                           # Notebooks for analysis and visualization
│   ├── Analyse_compression_v_acc.ipynb          # Analysis of compression vs accuracy
│   ├── Analyse_model_layers.ipynb               # Analysis of model layer structures
│   ├── Attention_Map_plot.ipynb                 # Notebook for plotting attention maps
│   ├── Check_model_weights.ipynb                # Inspection of model weights
│   ├── Compression_vs_accuracy.ipynb            # Compression vs accuracy analysis
│   ├── Compression_vs_accuracy_kd.ipynb         # KD-specific compression analysis
│   ├── Compression_vs_accuracy_not_pretrained.ipynb # Analysis without pretraining
│   ├── Compression_vs_accuracy_student.ipynb    # Student model compression analysis
│   ├── IG_plots.ipynb                           # Notebook for Integrated Gradients plots
│   ├── ImageNet.ipynb                           # Notebook for ImageNet related studies
│   ├── lit_review_plot.ipynb                    # Notebook for literature review visualizations
│   ├── Paper_plots.ipynb                        # Notebook for generating plots for the paper
│   ├── statistical.ipynb                        # Notebook for statistical analysis
│   ├── studies_vis_plot.ipynb                   # Visualization of study results
│   └── Tuned_KD_param_search.ipynb              # Knowledge distillation parameter tuning
├── PyTorch_CIFAR10/                             # Scripts and models for CIFAR-10 dataset
│   ├── analyse_models.py                        # Script to analyze model architectures and performance
│   ├── AT_compress.py                           # Implementation of Attention Transfer compression
│   ├── calculate_memory.py                      # Script to calculate model memory requirements
│   ├── calculate_stats_results.py               # Statistical analysis of experimental results
│   ├── cifar10_models/                          # Teacher model scripts and weights
│   ├── collect_all_accuracies.py                # Script to aggregate accuracy metrics
│   ├── Compare_attention_maps.py                # Script for comparing attention maps
│   ├── Compute_IGs.py                           # Script to compute Integrated Gradients
│   ├── Compute_logits_and_attn.py               # Compute model logits and attention maps
│   ├── data/                                    # Data directory
│   ├── gpu_comparison.py                        # Script for GPU performance comparison
│   ├── main.py                                  # Main script for training with command-line arguments
│   ├── parser.py                                # Argument parser utilities for command-line interface
│   ├── plot_cf_acc.py                           # Script to plot compression factor vs accuracy
│   └── UTILS_TORCH.py                           # Utility functions and classes
├── saved_models/                                # Folder with model weights
│   ├── ablation/                                # Models for ablation studies
│   └── montecarlo_weights/                      # Monte Carlo simulation weights
├── LICENSE                                      # License file
└── README.md                                    # This README file
```

### Key Components

- **analyse_models.py**: Analyzes and compares model architectures, parameters, and inference time for different compression levels.
- **AT_compress.py**: Implementation of Attention Transfer for model compression with layer and divider configuration testing.
- **calculate_memory.py**: Calculates memory footprint of different model configurations.
- **calculate_stats_results.py**: Performs statistical analysis on experimental results including t-tests and p-values.
- **Compare_attention_maps.py**: Compares attention maps between teacher and student models, saving visualizations.
- **Compute_IGs.py**: Computes Integrated Gradients for CIFAR-10 images using the Captum library.
- **Compute_logits_and_attn.py**: Pre-computes teacher model logits and attention maps for efficient training.
- **main.py**: Comprehensive training script with command-line arguments for model type, layers, divider, and other parameters.
- **parser.py**: Defines command-line arguments for the main training script.
- **UTILS_TORCH.py**: Contains utility functions and model classes, including implementations for KD, IG, and AT approaches.

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

   This will analyze the teacher model and generate different student models at various compression levels, comparing parameters, inference time, and structure. The script now automatically detects and uses Apple Silicon (MPS) if available.

2. **Computing Integrated Gradients**:

   ```
   python PyTorch_CIFAR10/Compute_IGs.py
   ```

   This will calculate and save integrated gradients for the CIFAR-10 dataset using the Captum library, storing results in the data directory.

3. **Pre-computing Teacher Logits and Attention Maps**:

   ```
   python PyTorch_CIFAR10/Compute_logits_and_attn.py
   ```

   This script pre-computes and saves teacher model outputs to be used during knowledge distillation and attention transfer.

4. **Comparing Attention Maps for Different Training Methods**:

   ```
   python PyTorch_CIFAR10/Compare_attention_maps.py
   ```

   This script trains and compares attention maps between the teacher and different student models (AT, KD_AT, KD_IG_AT, IG_AT).

5. **Running the Main Training Script with Different Configurations**:

   ```
   python PyTorch_CIFAR10/main.py --type KD_IG --layers 5 --epochs 100 --batch_size 64 --alpha 0.01 --temp 2.5
   ```

   The main training script now supports comprehensive command-line arguments for flexible configuration. You can specify:

   - `--type`: Training configuration (Student, KD, KD_IG, IG, KD_IG_AT, AT, KD_AT, or IG_AT)
   - `--layers`: Number of layers in the student model (3, 5, 7, 9, 11, 13, 15, or 17)
   - `--divider`: Divider value for attention transfer (determined automatically if not specified)
   - `--alpha`: Weight for distillation loss
   - `--temp`: Temperature for softening logits
   - `--gamma`: Weight for attention transfer
   - `--overlay_prob`: Probability for IG overlay
   - `--device`: Specify the device to use (cuda, mps, or cpu)
   - Additional options for controlling batch size, learning rate, etc.

6. **Generating Statistical Analysis of Results**:

   ```
   python PyTorch_CIFAR10/calculate_stats_results.py
   ```

   This script performs statistical analysis on the results of different model configurations, including t-tests and p-values.

7. **GPU Performance Comparison**:

   ```
   python PyTorch_CIFAR10/gpu_comparison.py
   ```

   Compares training and inference times across different GPU configurations (A5000, 3090, 3060 Ti).

#### Running Jupyter Notebooks

1. Start a Jupyter Notebook Server:

   ```
   jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
   ```

2. Access the Notebook: Open your browser and go to `http://localhost:8888`. You might need the token generated by the Jupyter server to log in, which is displayed in the terminal output.

3. Navigate to the Notebook: Once logged in, navigate to the `Jupyter_notebooks` directory and open one of the available notebooks for analysis and visualization.

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

```
Hernandez, David E., Jose Ramon Chang, and Torbjörn EM Nordling. "Knowledge Distillation: Enhancing Neural Network Compression with Integrated Gradients." arXiv preprint arXiv:2503.13008 (2025).
```
