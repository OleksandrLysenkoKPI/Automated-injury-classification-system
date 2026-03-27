import logging
import shutil
import random
import os

def log_section(title):
    """Log text separator"""
    logging.info(f"{'='*10} {title.upper()} {'='*10}")

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
                            
        logging.info(f"Class {condition} was processed. Files in Train: {counter-1}")