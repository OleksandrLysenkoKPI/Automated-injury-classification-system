from ..logger_module.logger import CustomLogger 
from dotenv import load_dotenv
import torch
import shutil
import random
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


logger = CustomLogger("ML_utils_log")

def verify_dataset_processing(dataset, sample_idx=0):
    tensor, label = dataset[sample_idx]
    
    logger.info(f"Tensor shape after processing: {tensor.shape}")
    logger.info(f"Label: {label} (Class: {dataset.classes[label]})")
    
    mid_d = tensor.shape[1] // 2
    slice_to_show = tensor[0, mid_d, :, :].numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_to_show, cmap='gray')
    plt.title(f"Processed Knee: {dataset.classes[label]}\nShape: {tensor.shape} | Slice: {mid_d}")
    plt.axis('off')
    
    plt.grid(color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    print(f"Min value: {tensor.min().item()}")
    print(f"Max value: {tensor.max().item()}")
    print(f"Mean value: {tensor.mean().item()}")
    plt.show()

def check_pytorch_install():
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Number OF GPUs: {torch.cuda.device_count()}")

def get_dataset_paths():
    """Loads and checks paths from environment"""
    load_dotenv()

    dataset_env = os.getenv("PREPARED_KNEE_DATASET")

    if not dataset_env:
        msg = "Dataset environment variables are not set properly."
        logger.error(msg)
        raise EnvironmentError(msg)
        
    dataset_path = Path(dataset_env)
    return {
        "train": dataset_path / "train",
        "train_augmented_npy": dataset_path / "train_augmented_npy",
        "train_augmented_png": dataset_path / "train_augmented_png",
        "val": dataset_path / "val",
        "test": dataset_path / "test"
    }

def numpy_examiner(numpy_folder_root: str | Path, print_paths: bool = False):
    """Prints grouped shapes and relative paths of all .npy files in a folder tree"""
    def print_file_paths(paths):
        print("Path to files:")
        for p in paths:
            print(p)
    
    shape_map = {}
    
    for root, _, files in os.walk(numpy_folder_root):
        for f in files:
            if f.endswith('.npy'):
                full_path = os.path.join(root, f)
                try:
                    data = np.load(full_path, mmap_mode="r")
                    shape = data.shape
                    
                    rel_path = os.path.relpath(full_path, numpy_folder_root)
                    
                    if shape not in shape_map:
                        shape_map[shape] = []
                    shape_map[shape].append(rel_path)
                except Exception as e:
                    logger.error(f"Could not read {f}: {e}")
        
    for shape, paths in shape_map.items():
        print(f"Shape {list(shape)}")
        if print_paths: print_file_paths(paths)

def organize_dataset(source_root, destination_root, train_ratio=0.8):
    conditions_list = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    
    for condition in conditions_list:
        condition_path = os.path.join(source_root, condition)
        patients_list = [p for p in os.listdir(condition_path) if os.path.isdir(os.path.join(condition_path, p))]
        
        random.shuffle(patients_list)
        split_idx = int(len(patients_list) * train_ratio)
        train_patients = patients_list[:split_idx]
        test_patients = patients_list[split_idx:]
        
        splits = {'train': train_patients, 'test': test_patients}
        
        for split_name, split_list in splits.items():
            counter = 1
            
            for patient in split_list:
                patient_path = os.path.join(condition_path, patient)
                
                for root, dirs, files in os.walk(patient_path):
                    for filename in files:
                        if filename.endswith('.npy'):
                            new_name = f"{counter:06d}.npy"
                            
                            target_dir = os.path.join(destination_root, split_name, condition)
                            os.makedirs(target_dir, exist_ok=True)
                            
                            source_file = os.path.join(root, filename)
                            destination_file = os.path.join(target_dir, new_name)
                            
                            shutil.copy2(source_file, destination_file)
                            counter += 1
                            
        logger.info(f"Class {condition} was processed. Files in Train: {counter-1}")