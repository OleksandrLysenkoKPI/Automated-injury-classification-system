# Automated Trauma Classification System Based on Machine Learning

[![Ukrainian](https://img.shields.io/badge/lang-uk-blue.svg)](README.uk.md)

---

This repository contains the code for the diploma project: **"Automated Trauma Classification System Based on Machine Learning"**.

## Description
Automated knee MRI trauma classification system with ML utilization.

## Requirements
### Hardware
- **GPU**: support for NVIDIA Automatic Mixed Precision (AMP)
- **VRAM**: Minimum 4GB
- **RAM**: tested on 16GB
### Software
- **Python 3.12+** (tested on 3.12.1)
- **CUDA driver** compatible with your hardware
- **Key Libraries**:
    - `torch` & `torchvision` (with CUDA support)
    - `opencv-python` (for image preprocessing)
    - `numpy`
    - `scikit-learn` (for evaluation metrics)
    - `pillow`

## Key Features
- **Two-stage Pipeline**: Screening (Binary) -> Diagnostics (Multi-class).
- **Sequential Transfer Learning**: Fine-tuning from binary weights to specialized pathology detection.
- **AMP Acceleration**: Faster training and reduced VRAM footprint using Mixed Precision.

## Configuration
Create a `.env` file in the root directory and specify the path to your dataset:

`KNEE_CONDITIONS_DATASET=path/to/conditions/Knee_MRI/dataset`\
`KNEE_DATASET=path/to/healthy/Knee_MRI/dataset`
