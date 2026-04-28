import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Imaging_Augmentation_log")

def is_valid_slice(img_array: np.ndarray, std_threshold: float = 10.0):
    """Checks if slice is not empty (black)

    Args:
        img_array (np.ndarray): _description_
        std_threshold (float, optional): Standard deviation threshold value. Defaults to 10.0.
    """
    return np.std(img_array) > std_threshold

def add_noise_to_npy(data: np.ndarray, standard: float) -> np.ndarray:
    noise = np.random.normal(0, standard, data.shape).astype(np.float16)
    return data + noise

def add_noise_to_png(img_array: np.ndarray, intensity: float = 0.02):
    noise = np.random.randn(*img_array.shape) * (intensity * 255)
    noised_img = img_array.astype(np.float32) + noise
    return np.clip(noised_img, 0, 255).astype(np.uint8)

def augment_and_save_png_dataset(root_path: str | Path, std_threshold: float = 10.0):
    """Augments given PNG image and saves it in a separate folder. Includes several augmentations:
    1. Original.
    2. Horizontal flip.
    3. Brightness Jitter.
    4. Noise.<br>
    And their respective combinations.

    Args:
        root_path (str | Path): Path to folder with data for augmentation.
        std_threshold(float): Standard deviation threshold value. Defaults to 10.0
    """
    root_path = Path(root_path)
    output_base = root_path.parents[1] / "train_augmented_png"
    output_base.mkdir(parents=True, exist_ok=True)
    
    for root, _, files in os.walk(root_path):
        png_files = [f for f in files if f.endswith('.png')]
        if not png_files:
            continue
        
        relative_path = Path(root).relative_to(root_path)
        target_folder = output_base / relative_path
        target_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing folder: {relative_path}. Found {len(png_files)} PNG slices.")

        for file_name in png_files:
            try:
                file_path = Path(root) / file_name
                img = Image.open(file_path).convert('L')
                data = np.array(img)
                
                if not is_valid_slice(data, std_threshold):
                    continue
                
                base_name = Path(file_name).stem
                
                # --- Basic ---
                # Original
                img.save(target_folder / f"{base_name}_orig.png")
                
                # Flip
                flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                flipped_img.save(target_folder / f"{base_name}_flipped.png")
                
                # Brightness
                enhancer = ImageEnhance.Brightness(img)
                bright = enhancer.enhance(1.4)
                dark = enhancer.enhance(0.6)
                bright.save(target_folder / f"{base_name}_bright.png")
                dark.save(target_folder / f"{base_name}_dark.png")
                
                # Noise
                noised_data = add_noise_to_png(data, intensity=0.03)
                Image.fromarray(noised_data).save(target_folder / f"{base_name}_noised.png")
                
                # --- Combinations ---
                # Flip + Brightness
                enhancer_flip = ImageEnhance.Brightness(flipped_img)
                
                flipped_dark = enhancer_flip.enhance(0.6)
                flipped_dark.save(target_folder / f"{base_name}_flipped_dark.png")
                
                flipped_bright = enhancer_flip.enhance(1.4)
                flipped_bright.save(target_folder / f"{base_name}_flipped_bright.png")
                
                # Flip + Noise
                flipped_noised = add_noise_to_png(np.array(flipped_img), intensity=0.03)
                Image.fromarray(flipped_noised).save(target_folder / f"{base_name}_flipped_noised.png")
                
                # Heavy Combinations
                # Flip + Dark + Noise
                heavy_data = add_noise_to_png(np.array(flipped_dark), intensity=0.03)
                Image.fromarray(heavy_data).save(target_folder / f"{base_name}_heavy.png")
                
                # Flip + Bright + Noise
                heavy_bright = add_noise_to_png(np.array(flipped_bright), intensity=0.03)
                Image.fromarray(heavy_bright).save(target_folder / f"{base_name}_heavy_bright.png")
            except Exception as e:
                logger.error(f"Failed to augment {file_name}: {e}")
                continue
    
    logger.info(f"PNG Augmentation finished. Location: {output_base}")        
        
# TODO: update for a new folder structure
def augment_and_save_npy_dataset(root_path: str | Path):
    """Augments given NumPy data and saves it in a separate folder. Includes several augmentations:
    1. Original.
    2. Horizontal flip.
    3. Brightness Jitter.
    4. Noise.<br>
    And their respective combinations.

    Args:
        root_path (str | Path): Path to folder with data for augmentation
    """
    root_path = Path(root_path)
    output_base = root_path.parents[1] / "train_augmented_npy"
    output_base.mkdir(parents=True, exist_ok=True)
    
    for root, _, files in os.walk(root_path):
        npy_files = [f for f in files if f.endswith('.npy')]
        if not npy_files:
            continue
        
        relative_path = Path(root).relative_to(root_path)
        target_folder = output_base / relative_path
        target_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing folder: {relative_path}. Found {len(npy_files)} NumPy files.")
        
        for file_name in npy_files:
            try:
                file_path = Path(root) / file_name
                data: np.ndarray = np.load(file_path)
                base_name = Path(file_name).stem
                
                # --- Basic ---
                # 1. Original
                np.save(target_folder / f"{base_name}_orig.npy", data)
                
                # 2. Flip
                flipped = np.flip(data, axis=-1)
                np.save(target_folder / f"{base_name}_flipped.npy", flipped)
                
                # 3. Brightness Jitter
                bright = np.clip(data * 1.15, 0, 1)
                dark = np.clip(data * 0.85, 0, 1)
                np.save(target_folder / f"{base_name}_bright.npy", bright)
                np.save(target_folder / f"{base_name}_dark.npy", dark)
                
                # 4. Noise
                noise_lvl = 0.005 * (data.max() - data.min())
                noised_orig = add_noise_to_npy(data, noise_lvl)
                np.save(target_folder / f"{base_name}_noised.npy", noised_orig)
                
                # --- Combinations ---
                # 1. Flip + Brightness
                flipped_bright = np.clip(flipped * 1.15, 0, 1)
                flipped_dark = np.clip(flipped * 0.85, 0, 1)
                np.save(target_folder / f"{base_name}_flipped_bright.npy", flipped_bright)
                np.save(target_folder / f"{base_name}_flipped_dark.npy", flipped_dark)
                
                # 2. Flip + Noise
                np.save(target_folder / f"{base_name}_flipped_noised.npy", add_noise_to_npy(flipped, noise_lvl))
                
                # 3. Brightness + Noise
                np.save(target_folder / f"{base_name}_bright_noised.npy", add_noise_to_npy(bright, noise_lvl))
                np.save(target_folder / f"{base_name}_dark_noised.npy", add_noise_to_npy(dark, noise_lvl))
                
                # 4. All
                heavy = add_noise_to_npy(np.clip(flipped * 0.85, 0, 1), noise_lvl)
                np.save(target_folder / f"{base_name}_heavy.npy", heavy)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
               
    logger.info(f"Augmentation finished. New augmented dataset location: {output_base}")