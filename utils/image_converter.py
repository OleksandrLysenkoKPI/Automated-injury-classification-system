import pydicom
from pydicom.pixels import apply_rescale
import numpy as np
from PIL import Image

class DICOMProcessor:
    def __init__(self, dicom_path):
        self.dicom_path = dicom_path
        self.ds = pydicom.dcmread(dicom_path)
        self._pixels_hu = None
        
    @property
    def pixels_hu(self):
        """Lazy download and rescaling into Hounsfield units (HU)"""
        if self._pixels_hu is None:
            raw_pixels = self.ds.pixel_array.astype(np.float32)
            self._pixels_hu = apply_rescale(raw_pixels, self.ds)
        return self._pixels_hu
    
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
        pixels_8bit = self.get_normalized(target_range=(0, 255))
        image = Image.fromarray(pixels_8bit)
        image.save(output_path)
        print(f"PNG збережено: {output_path}")
        
    def save_as_npy(self, output_path):
        pixels_norm = self.get_normalized(target_range=(0, 1))
        np.save(output_path, pixels_norm)
        print(f"NumPy збережено: {output_path}")
