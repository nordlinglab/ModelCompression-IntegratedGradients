# ModelCompression-IntegratedGradients

Repository for the Model compression integrated gradients project

## Model Compression with Knowledge Distillation and Integrated Gradients

### Overview

This repository contains the implementation of our novel approach to model compression by integrating knowledge distillation (KD) with Integrated Gradients (IG) and attention transfer (AT), as described in our paper. This method is designed to enhance both performance and interpretability of compressed deep learning models.

### Repository Structure

.
├── Jupyter_notebooks # Notebooks for analysis and visualization
│ ├── Compression_vs_accuracy\*.ipynb # Notebooks for compression vs. accuracy studies
│ ├── IG_plots.ipynb # Notebook for Integrated Gradients plots
│ ├── ImageNet.ipynb # Notebook for ImageNet related studies
│ ├── Paper_plots.ipynb # Notebook for generating plots for the paper
│ ├── statistical.ipynb # Notebook for statistical analysis
│ └── studies_plot.py # Script for additional plotting
├── LICENSE # License file
├── PyTorch_CIFAR10 # Scripts and models for CIFAR-10 dataset
│ ├── Compute_IGs.py # Script to compute and Integrated Gradients
│ ├── KD.py # Knowledge Distillation script
│ ├── UTILS_TORCH.py # All functions and classes using PyTorch's framework
│ ├── data.py # Script for data handling
│ └── train.py # Script for training models
├── README.md # This README file
└── saved_models # Trained model weights
├── \*.pt # PyTorch model weights
