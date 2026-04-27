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
    Splits data into Train, Validation and Test sets on patient level.
    Copies .npy and .png data to ensure consistency.
    The remaining ratio (1 - train_ratio - val_ratio) goes to Test.
    """
    logger.info("Start data splitting process")
    
    npy_root = Path(npy_root)
    png_root = Path(png_root)
    output_base = Path(output_base)
    
    if not npy_root.exists():
        raise FileNotFoundError(f"NPY root not found: {npy_root}")
    if not png_root.exists():
        logger.warning(f"PNG root not found at {png_root}. Only .npy files will be processed.")
    
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-5:
        raise ValueError("The sum of train + val cannot exceed 1.0")
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    types = ["npy", "png"]
    
    conditions = [d.name for d in npy_root.iterdir() if d.is_dir()]
    
    for condition in conditions:
        logger.info(f"Splitting for condition: {condition}")
        
        patient_folders = sorted([d for d in (npy_root / condition).iterdir() if d.is_dir()])
        if not patient_folders:
            logger.warning(f"Empty condition folder: {condition}")
            continue

        random.shuffle(patient_folders)
        
        n_patients = len(patient_folders)
        idx_train = int(n_patients * train_ratio)
        idx_val = int(n_patients * (train_ratio + val_ratio))
        
        patient_subsets = {
            "train": patient_folders[:idx_train],
            "val": patient_folders[idx_train:idx_val],
            "test": patient_folders[idx_val:]
        }
        
        for split in splits:
            subset = patient_subsets[split]
            
            for t in types:
                (output_base / split / t / condition).mkdir(parents=True, exist_ok=True)
            
            for patient_path in subset:
                patient_name = patient_path.name
                
                dest_npy = output_base / split / "npy" / condition / patient_name
                dest_png = output_base / split / "png" / condition / patient_name
                
                try:
                    shutil.copytree(patient_path, dest_npy, dirs_exist_ok=True)
                    
                    src_png = png_root / condition / patient_name
                    if src_png.exists():
                        shutil.copytree(src_png, dest_png, dirs_exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to copy data for {patient_name}: {e}")
        
        logger.info(f"Condition {condition}: Train={len(patient_subsets['train'])}, "
                    f"Val={len(patient_subsets['val'])}, Test={len(patient_subsets['test'])} patients.")

    logger.info(f"Split complete. Data located at: {output_base}")


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