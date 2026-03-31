from ..logger_module.logger import CustomLogger 
import shutil
import random
import os
from pathlib import Path
import numpy as np

logger = CustomLogger("ML_utils_log")

def numpy_examinator(numpy_folder_root: str | Path):
    """Prints grouped shapes and relative paths of all .npy files in a folder tree"""
    shape_map = {}
    
    for root, dirs, files in os.walk(numpy_folder_root):
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
        print("Path to files:")
        for p in paths:
            print(p)
        print()

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