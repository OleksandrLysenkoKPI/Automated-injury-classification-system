import pydicom
from pydicom.pixels import apply_rescale
import numpy as np
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DICOMProcessor:
    def __init__(self):
        """Initialized without a path to allow reuse for batch processing."""
        self.ds = None
        self._pixels_hu = None

    def load_file(self, dicom_path):
        """Loads a single DICOM file into the processor."""
        if not os.path.exists(dicom_path):
            logging.error(f"Path does not exist: {dicom_path}")
            return False
        
        try:
            self.ds = pydicom.dcmread(dicom_path)
            self._pixels_hu = None
            return True
        except Exception as e:
            logging.error(f"Failed to load {dicom_path}: {e}")
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
                logging.warning(f"Rescale tags missing, using raw pixel data: {e}")
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
            logging.error(f"Normalization error: {e}")
            return None
    
    def save_as_png(self, output_path):
        data = self.get_normalized(target_range=(0, 255))
        if data is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(data).save(output_path)
            return True
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
            logging.error(f"Input directory not found: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)

        files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
        logging.info(f"Found {len(files)} DICOM files in {input_dir}")

        for filename in files:
            file_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            if self.load_file(file_path):
                if conversion_type in ["png", "both"]:
                    self.save_as_png(os.path.join(output_dir, f"{base_name}.png"))
                if conversion_type in ["npy", "both"]:
                    self.save_as_npy(os.path.join(output_dir, f"{base_name}.npy"))