import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random
from pathlib import Path
from scipy.ndimage import zoom
from skimage.restoration import denoise_wavelet, estimate_sigma
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Imaging_utils_log")

def get_knee_bbox(data, threshold=0.01):
    coords = np.argwhere(data > threshold)
    if coords.size == 0:
        return data
    
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0) + 1
    
    return data[d0:d1, h0:h1, w0:w1]

def resample_3d(data, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """Changes array size based on physical distancce between voxels.

    Args:
        data (_type_): 3D array [D, H, W]
        current_spacing (tuple): Current voxel size (SliceThickness, PixelSpacing_H, PixelSpacing_W)
        target_spacing (tuple): Target voxel size.
    """
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    
    resampled_data = zoom(data, scale_factors, order=3, mode='constant', cval=0.0)
    
    return resampled_data

def wavelet_denoising_3d(data):
    """
    Noise reduction in 3D data using wavelet transformation.
    """
    sigma_est = estimate_sigma(data, average_sigmas=True)
    
    denoised_data = denoise_wavelet(
        data,
        method='BayesShrink',
        mode='soft',
        wavelet='db1',
        rescale_sigma=True
    )
    
    return denoised_data.astype(np.float32)

def split_data(root_path: str | Path, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Splits data into Train, Validation and Test sets.
    The remaining ratio (1 - train_ratio - val_ratio) goes to Test.
    """
    logger.info("Start data splitting process")
    root_path = Path(root_path)
    data_dir = root_path.parents[1] 
    prepared_data_path = data_dir / "prepared_data"
    
    split_paths = {
        "train": prepared_data_path / "train",
        "val": prepared_data_path / "val",
        "test": prepared_data_path / "test",
    }
    
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("Sum of train_ratio and val_ratio cannot exceed 1.0")
    
    try:
        prepared_data_path.mkdir(exist_ok=True)
        
        for cls_folder in root_path.iterdir():
            if not cls_folder.is_dir():
                continue
            
            for path in split_paths.values():
                (path / cls_folder.name).mkdir(parents=True, exist_ok=True)
            
            files = list(cls_folder.glob("*.npy"))
            random.shuffle(files)
            
            n_total = len(files)
            idx_train = int(n_total * train_ratio)
            idx_val = int(n_total * (train_ratio + val_ratio))
            
            subsets = {
                "train": files[:idx_train],
                "val": files[idx_train:idx_val],
                "test": files[idx_val:]
            }
            
            for key, subset_files in subsets.items():
                dest_folder = split_paths[key] / cls_folder.name
                for f in subset_files:
                    shutil.copy(f, dest_folder / f.name)
                
                logger.info(f"Class {cls_folder.name}: {len(subset_files)} files copied to {key}")
    except Exception as e:
        logger.error(f"Error occurred during data splitting: {e}")
        raise
    
    logger.info(f"Split complete. Prepared data located at: {prepared_data_path}")
    return split_paths["train"], split_paths["val"], split_paths["test"]

def add_noise(data: np.ndarray, standard: float) -> np.ndarray:
    noise = np.random.normal(0, standard, data.shape).astype(np.float32)
    return data + noise

def augment_and_save_dataset(root_path: str | Path):
    """Augments given NumPy data and saves it in a separate folder. Includes several augmentations:
    1. Original.
    2. Horizontal flip.
    3. 90 degree rotation.
    4. Noise.<br>
    And their respective combinations.

    Args:
        root_path (str | Path): Path to folder with data for augmentation
    """
    root_path = Path(root_path)
    output_base = root_path.parent / "train_augmented"
    output_base.mkdir(parents=True, exist_ok=True)
    
    for root, dirs, files in os.walk(root_path):
        npy_files = [f for f in files if f.endswith('.npy')]
            
        if not npy_files:
            continue
        
        class_name = Path(root).name
        target_folder = output_base / class_name
        target_folder.mkdir(exist_ok=True)
        
        logger.info(f"Processing class: {class_name}. Found {len(npy_files)} files")
        
        for file_name in npy_files:
            try:
                file_path = Path(root) / file_name
                data: np.ndarray = np.load(file_path)
                base_name = Path(file_name).stem

                noise_standard = 0.01 * (data.max() - data.min())
                
                # --- Basic ---
                # 1. Original
                np.save(target_folder / f"{base_name}_orig.npy", data)
                
                # 2. Flip
                flipped = np.flip(data, axis=-1)
                np.save(target_folder / f"{base_name}_flipped.npy", flipped)
                
                # 3. 90 deg Rotation
                rotated = np.rot90(data, k=1, axes=(-2, -1))
                np.save(target_folder / f"{base_name}_rotated.npy", rotated)
                
                # 4. Noise
                np.save(target_folder / f"{base_name}_noised.npy", add_noise(data, noise_standard))
                
                # --- Combinations ---
                # 1. Flip + Rotate
                f_r = np.rot90(flipped, k=1, axes=(-2, -1))
                np.save(target_folder / f"{base_name}_flip_rotation.npy", f_r)
                
                # 2. Flip + Noise
                np.save(target_folder / f"{base_name}_noise_flip.npy", add_noise(flipped, noise_standard))
                
                # 3. Rotate + Noise
                np.save(target_folder / f"{base_name}_noise_rotation.npy", add_noise(rotated, noise_standard))
                
                # 4. All
                np.save(target_folder / f"{base_name}_all.npy", add_noise(f_r, noise_standard))
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
               
    logger.info(f"Augmentation finished. New augmented dataset location: {output_base}")
            
# TODO: Rewrite to work woth transformed images 
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