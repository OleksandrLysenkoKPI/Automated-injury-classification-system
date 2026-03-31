import pydicom
from pydicom.pixels.processing import apply_rescale
import numpy as np
from PIL import Image
import os
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Imaging_log")

class DICOMProcessor:
    def __init__(self):
        self.ds = None
        self._pixels_hu = None

    def load_file(self, dicom_path):
        """Loads a single DICOM file into the processor."""
        if not os.path.exists(dicom_path):
            logger.error(f"Path does not exist: {dicom_path}")
            return False
        
        try:
            self.ds = pydicom.dcmread(dicom_path)
            self._pixels_hu = None
            return True
        except Exception as e:
            logger.error(f"Failed to load {dicom_path}: {e}")
            return False
    
    @property
    def pixels_hu(self):
        """Lazy loading and rescaling into Hounsfield units (HU)"""
        if self.ds is None:
            raise ValueError("No DICOM file loaded. Call load_file() first.")
        
        if self._pixels_hu is None:
            try:
                raw_pixels = self.ds.pixel_array.astype(np.float32)
                self._pixels_hu = apply_rescale(raw_pixels, self.ds)
            except AttributeError as e:
                logger.warning(f"Rescale tags missing, using raw pixel data: {e}")
                self._pixels_hu = self.ds.pixel_array.astype(np.float32)
        return self._pixels_hu
    
    def get_normalized(self, target_range=(0, 1)):
        """Normalizes pixel values to a specific range."""
        try:
            hu_data = self.pixels_hu
            img_min, img_max = hu_data.min(), hu_data.max()
            
            if img_max - img_min == 0:
                return np.zeros_like(hu_data)
            
            normalized = (hu_data - img_min) / (img_max - img_min)
            
            if target_range == (0, 255):
                return (normalized * 255).astype(np.uint8)
            return normalized.astype(np.float32)
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            return None
    
    def save_as_png(self, output_path):
        data = self.get_normalized(target_range=(0, 255))
        if data is not None:
            data = np.squeeze(data)
            
            if data.ndim == 3:
                num_slices = data.shape[0]
                base_path = output_path.replace('.png', '')
                
                for i in range(num_slices):
                    slice_data = data[i, :, :]
                    dicom_name = os.path.basename(base_path)
                    dicom_dir = os.path.join(os.path.dirname(base_path), dicom_name)
                    os.makedirs(dicom_dir, exist_ok=True)
                    slice_path = os.path.join(dicom_dir, f"slice{i:03d}.png")
                    Image.fromarray(slice_data).save(slice_path)
                
                logger.info(f"Saved {num_slices} slices from one DICOM to PNG")
                return True
            elif data.ndim == 2:
                Image.fromarray(data).save(output_path)
                return True
            else:
                logger.error(f"Unsupported shape: {data.shape}")
                return False
        
    def save_as_npy(self, output_path):
        data = self.get_normalized(target_range=(0, 1))
        if data is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, data)
            return True
        return False
    
    def batch_conversion(self, input_dir, output_dir, conversion_type="png"):
        """
        Iterates through a folder and converts all DICOM files.
        conversion_type: 'png', 'npy', or 'both'
        """
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
        logger.info(f"Found {len(files)} DICOM files in {input_dir}")

        for filename in files:
            file_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            if self.load_file(file_path):
                if conversion_type in ["png", "both"]:
                    self.save_as_png(os.path.join(output_dir, f"{base_name}.png"))
                if conversion_type in ["npy", "both"]:
                    self.save_as_npy(os.path.join(output_dir, f"{base_name}.npy"))
                    
    def process_all_conditions(self, root_dir, output_base_png, output_base_npy):
        """
        Iterates through all the branch of a specified path and converts files,
        recreating folder structure in output directories
        """
        for root, dirs, files in os.walk(root_dir):
            dicom_files = [f for f in files if f.endswith('.dcm')]
            
            if not dicom_files:
                continue
            
            
            relative_path = os.path.relpath(root, root_dir)
            
            target_png_dir = os.path.join(output_base_png, relative_path)
            target_npy_dir = os.path.join(output_base_npy, relative_path)
            
            os.makedirs(target_png_dir, exist_ok=True)
            os.makedirs(target_npy_dir, exist_ok=True)
            
            logger.info(f"Processing folder: {relative_path}")
            
            self.batch_conversion(root, target_png_dir, conversion_type="png")
            self.batch_conversion(root, target_npy_dir, conversion_type="npy")