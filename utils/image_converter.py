import pydicom
from pydicom.pixels import apply_rescale
import numpy as np
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DICOMProcessor:
    def __init__(self, dicom_path):
        self.dicom_path = dicom_path
        self._pixels_hu = None
        
        if not os.path.exists(dicom_path):
            logging.error(f"Initialization failed. Path does not exist: {dicom_path}")
            raise FileNotFoundError(f"File not found: {dicom_path}")
        
        try:
            self.ds = pydicom.dcmread(dicom_path)
        except pydicom.errors.InvalidDicomError:
            logging.error(f"Initialization failed. Invalid DICOM format: {dicom_path}")
            raise ValueError(f"File is not valid DICOM: {dicom_path}")
        except Exception as e:
            logging.error(f"Initialization failed. Unexpected error for {dicom_path}: {e}")
            raise
    
    # TODO: Add error handling
    @property
    def pixels_hu(self):
        """Lazy loading and rescaling into Hounsfield units (HU)"""
        if self._pixels_hu is None:
            raw_pixels = self.ds.pixel_array.astype(np.float32)
            self._pixels_hu = apply_rescale(raw_pixels, self.ds)
        return self._pixels_hu
    
    # TODO: Add error handling
    def get_normalized(self, target_range=(0, 1)):
        """General method for data normalization"""
        img_min = self.pixels_hu.min()
        img_max = self.pixels_hu.max()
        
        if img_max - img_min == 0:
            return np.zeros_like(self.pixels_hu)
        
        normalized = (self.pixels_hu - img_min) / (img_max - img_min)
        
        if target_range == (0, 255):
            return (normalized * 255).astype(np.uint8)
        
        return normalized.astype(np.float32)
    
    def save_as_png(self, output_path):
        try:
            data = self.get_normalized(target_range=(0, 255))
            if data is None: return
            
            target_dir = os.path.dirname(output_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
        
            Image.fromarray(data).save(output_path)
            logging.info(f"Successfully saved file as PNG: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save file as PNG {output_path}: {e}")
        
    def save_as_npy(self, output_path):
        try:
            data = self.get_normalized(target_range=(0, 1))
            if data is None: return
            
            target_dir = os.path.dirname(output_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            
            np.save(output_path, data)
            logging.info(f"Successfully saved file as NumPy: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save file as NumPy {output_path}: {e}")
        