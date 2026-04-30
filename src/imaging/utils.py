from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
from pathlib import Path
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Imaging_utils_log")

def split_data(npy_root: str | Path, png_root: str | Path, output_base: str | Path,
               train_ratio: float = 0.6, val_ratio: float = 0.25):
    """
    Splits data into Train, Val, and Test at the patient ID level.
    Ensures that the L and R knees of the same patient fall into the same sample.
    """
    logger.info("Starting the process of separating data by patient identifiers")
    
    npy_root, png_root, output_base = Path(npy_root), Path(png_root), Path(output_base)
    datasets = ["conditions_dataset", "knee_dataset"]
    
    for ds_tag in datasets:
        ds_npy_path = npy_root / ds_tag
        if not ds_npy_path.exists():
            continue
            
        for condition_path in [d for d in ds_npy_path.iterdir() if d.is_dir()]:
            condition = condition_path.name
            logger.info(f"Processing category: {ds_tag}/{condition}")
            
            # Group patients folders by id
            patient_groups = defaultdict(list)
            for p_folder in condition_path.iterdir():
                if p_folder.is_dir():
                    # get "patient#1" with "patient#1_L" or "patient#1_R"
                    base_id = p_folder.name.split('_')[0]
                    patient_groups[base_id].append(p_folder)
            
            # id shuffle
            unique_ids = list(patient_groups.keys())
            random.seed(42)
            random.shuffle(unique_ids)
            
            n_ids = len(unique_ids)
            idx_train = int(n_ids * train_ratio)
            idx_val = int(n_ids * (train_ratio + val_ratio))
            
            split_assignments = {
                "train": unique_ids[:idx_train],
                "val": unique_ids[idx_train:idx_val],
                "test": unique_ids[idx_val:]
            }
            
            # data split
            for split_name, ids in split_assignments.items():
                (output_base / split_name / "npy" / condition).mkdir(parents=True, exist_ok=True)
                (output_base / split_name / "png" / condition).mkdir(parents=True, exist_ok=True)
                
                for p_id in ids:
                    for src_patient_npy in patient_groups[p_id]:
                        patient_folder_name = src_patient_npy.name # patient#1_L
                        
                        dest_npy = output_base / split_name / "npy" / condition / patient_folder_name
                        dest_png = output_base / split_name / "png" / condition / patient_folder_name
                        
                        try:
                            shutil.copytree(src_patient_npy, dest_npy, dirs_exist_ok=True)
                            
                            src_png = png_root / ds_tag / condition / patient_folder_name
                            if src_png.exists():
                                shutil.copytree(src_png, dest_png, dirs_exist_ok=True)
                            
                        except Exception as e:
                            logger.error(f"Copy error {patient_folder_name}: {e}")

            logger.info(f"Category {condition}: Patients={n_ids}, "
                        f"Distribution: Train={len(split_assignments['train'])}, "
                        f"Val={len(split_assignments['val'])}, Test={len(split_assignments['test'])}")

    logger.info(f"Separation complete. Data located by path: {output_base}")


# TODO: Rewrite to work with transformed images
def verify_npy_conversion(processor, dicom_path, npy_path):
    """Compares original DICOM with loaded NumPy file"""
    try:
        if not processor.load_file(dicom_path):
            raise ValueError(f"Can't load DICOM: {dicom_path}")
        
        original_hu = processor.pixels_hu
        
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"NumPy file was not found: {npy_path}")
        
        npy_pixels = np.load(npy_path)
        
        print("\n" + "="*30)
        print("VERIFICATION REPORT")
        print("="*30)
        print(f"Original form (HU): {original_hu.shape}")
        print(f"NumPy form: {npy_pixels.shape}")
        
        if original_hu.shape != npy_pixels.shape:
            print("ERROR: Arrays shapes don't match!")
            return False
        
        correlation = np.corrcoef(original_hu.flatten(), npy_pixels.flatten())[0, 1]
        print(f"Data corelation:    {correlation:.6f}")
        
        if correlation > 0.999:
            print("SUCCESS: Corelation was successful (data is identical)")
        else:
            print("WARNING: Data divergences detected")
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        if original_hu.ndim == 3:
            mid_idx = original_hu.shape[0] // 2
            slice_orig = original_hu[mid_idx]
            slice_npy = npy_pixels[mid_idx]
            title_suffix = f"Slice {mid_idx}"
        else:
            slice_orig = original_hu
            slice_npy = npy_pixels
            title_suffix = ""        
        
        im1 = axes[0].imshow(slice_orig, cmap='gray')
        axes[0].set_title(f"Original HU {title_suffix}")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(slice_npy, cmap='gray')
        axes[1].set_title(f"Normalized NumPy {title_suffix}")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False