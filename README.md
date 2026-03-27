# MFI-Net

Official / unofficial PyTorch implementation of **MFI-Net** for fine-grained object recognition in point clouds.

> **Paper**: *MFI-Net: Multi-stage Feature Interaction Network for Fine-grained Object Recognition in Point Clouds*  
> **Paper Link**: https://www.sciencedirect.com/science/article/pii/S0925231226006296

---

## Overview

MFI-Net is a deep learning framework for **fine-grained point cloud object recognition**.  
It is designed to enhance feature representation through **multi-stage feature interaction**, enabling the model to capture both local geometric details and high-level semantic information more effectively.

This repository provides the core implementation of MFI-Net, including:

- data loading pipeline
- point cloud preprocessing utilities
- model training
- model evaluation / testing

---

## Network Architecture

The overall architecture of MFI-Net is shown below:

<p align="center">
  <img src="assets/mfi_net_architecture.png" alt="MFI-Net Architecture" width="900"/>
</p>

> If you do not have a local image path yet, you can replace the `src` above with your figure path, or directly use the image URL.

---

## Features

- Multi-stage feature interaction for fine-grained recognition
- Point cloud classification pipeline based on PyTorch
- Easy-to-extend training and evaluation scripts
- Modular code structure for data processing and model development

---

## Repository Structure

```bash
MFI-Net/
├── README.md
├── dataset.py           # Dataset loading and preprocessing
├── provider.py          # Data augmentation / provider functions
├── pointnet_util.py     # PointNet-related utility modules
├── train_cls.py         # Training script for classification
├── test_cls.py          # Testing / evaluation script
└── modle/               # Model definition files
```

Requirements
```bash
Python 3.8+
PyTorch
NumPy
tqdm
scikit-learn
```
You can install the main dependencies with:
```bash
pip install torch torchvision numpy tqdm scikit-learn
```
If you are using a specific CUDA version, please install the matching PyTorch build from the official PyTorch website.
