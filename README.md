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
    - `PyQt6` (for GUI)

## Key Features
- **Two-stage Pipeline**: Screening (Binary) -> Diagnostics (Multi-class).
- **Sequential Transfer Learning**: Fine-tuning from binary weights to specialized pathology detection.
- **AMP Acceleration**: Faster training and reduced VRAM footprint using Mixed Precision.

## Configuration

### Environmental Variables 
Create a `.env` file in the root directory and specify your dataset paths:

```env
KNEE_CONDITIONS_DATASET=path/to/conditions/Knee_MRI/dataset
KNEE_DATASET=path/to/healthy/Knee_MRI/dataset

CONVERTED_NPY = data/converted_data/converted_NumPy
CONVERTED_PNG = data/converted_data/converted_PNG
PREPARED_KNEE_DATASET = data/prepared_data
```

### Folder structure
The following directories must be created in the root folder (at the same level as src/):
```
.
├── data/           # Raw and processed data storage
├── logs/           # Application and training logs
├── models/         # Trained model weights (.pth)
└── src/            # Source code
```

### Pre-trained Models
For the GUI classification to function correctly, the following model weights must be placed inside the models/ folder:
- `knee_3d_binary_model.pth` (Stage 1)
- `knee_3d_stage2_6classes.pth` (Stage 2)

## Installation

### 1. Python Environment Setup
The project relies on the Python version specified in the `.python-version` file. If you use a version manager (such as `pyenv`), it will automatically detect and switch to the required version. Use the following command to verify your current Python version:

```Bash
python --version
```

### 2. Creating a Virtual Environment
To isolate project dependencies, it is recommended to use a virtual environment (`venv`):

```Bash
# Create a virtual environment
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Linux/macOS)
source venv/bin/activate
```

### 3. Installing PyTorch with CUDA Support (Important!)
For the machine learning algorithms to function correctly and utilize hardware acceleration (GPU), you must install the PyTorch version that matches your CUDA driver.

Do not install PyTorch directly from `requirements.txt` if you plan to use a GPU. Instead, visit the [official  PyTorch website](https://pytorch.org/get-started/locally/) and select the installation command appropriate for your system.

Ensure you check your system for a CUDA driver and its specific version. You may need to install the CUDA toolkit separately.

### 4. Installing Other Dependencies
Once PyTorch is configured, install the remaining required packages from the `requirements.txt` file:

```Bash
pip install -r requirements.txt
```