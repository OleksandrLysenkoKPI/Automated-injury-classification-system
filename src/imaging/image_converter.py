import pydicom
from pydicom.pixels.processing import apply_rescale
import numpy as np
from PIL import Image
import os
from .utils import wavelet_denoising_3d, resample_3d, get_knee_bbox
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
    
    @property
    def spacing(self):
        """Returns (SliceThickness, PixelSpacing_H, PixelSpacing_W)"""
        if self.ds is None:
            return (1.0, 1.0, 1.0)
        
        z_spacing = float(getattr(self.ds, 'SpacingBetweenSlices', 1.0))
        
        pixel_spacing = None
        
        try:
            if hasattr(self.ds, 'SharedFunctionalGroupsSequence'):
                shared_groups = self.ds.SharedFunctionalGroupsSequence[0]
                if hasattr(shared_groups, 'PixelMeasuresSequence'):
                    pixel_spacing = shared_groups.PixelMeasuresSequence[0].PixelSpacing
            
            if pixel_spacing is None and hasattr(self.ds, 'PerFrameFunctionalGroupsSequence'):
                frame_groups = self.ds.PerFrameFunctionalGroupsSequence[0]
                if hasattr(frame_groups, 'PixelMeasuresSequence'):
                    pixel_spacing = frame_groups.PixelMeasuresSequence[0].PixelSpacing
        except Exception as e:
            logger.warning(f"Error traversing functional groups: {e}")
            
        
        if pixel_spacing:
            h_spacing = float(pixel_spacing[0])
            w_spacing = float(pixel_spacing[1])
        else:
            logger.warning("PixelSpacing NOT found in functional groups. Using default 1.0mm")
            h_spacing, w_spacing = 1.0, 1.0
        
        return (z_spacing, h_spacing, w_spacing)
    
    def get_normalized(self, data, target_range=(0, 1), clamping_percentile=(1, 99)):
        """Internal helper to normalize any given array to a specific range and handle outliers via clamping."""
        lower_bound = np.percentile(data, clamping_percentile[0])
        upper_bound = np.percentile(data, clamping_percentile[1])
        
        data = np.clip(data, lower_bound, upper_bound)
        img_min, img_max = data.min(), data.max()
        
        if img_max - img_min == 0:
            return np.zeros_like(data)
        
        normalized = (data - img_min) / (img_max - img_min)
        
        if target_range == (0, 255):
            return (normalized * 255).astype(np.uint8)
        
        return normalized.astype(np.float32)
    
    def get_processed_volume(self, target_range=(0, 1), target_spacing=(1.0, 1.0, 1.0)):
        """Applies the full preprocessing pipeline to the current DICOM object."""
        try:
            data = self.pixels_hu
            
            logger.info("Applying wavelet denoising...")
            data = wavelet_denoising_3d(data)
            
            logger.info(f"Resampling from {self.spacing} to {target_spacing}...")
            data = resample_3d(data, self.spacing, target_spacing=target_spacing)
            
            logger.info("Applying Bounding Box crop...")
            data = get_knee_bbox(data, threshold=0.01)
        
            normalized_data = self.get_normalized(data, target_range=target_range)
            return normalized_data
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
        return None

# TODO: Fix save folder structure
    def save_as_png(self, data, output_path):
        """Slices the volume and saves as PNGs."""
        data = np.squeeze(data)
        if data.ndim == 3:
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            dicom_dir = os.path.join(os.path.dirname(output_path), base_name)
            os.makedirs(dicom_dir, exist_ok=True)
            for i, slice_data in enumerate(data):
                Image.fromarray(slice_data).save(os.path.join(dicom_dir, f"slice{i:03d}.png"))
            return True
        elif data.ndim == 2:
            Image.fromarray(data).save(output_path)
            return True
        return False
    
    def batch_conversion(self, input_dir, output_png_dir=None, output_npy_dir=None):
        """
        Iterates through a folder and converts all DICOM files.
        conversion_type: 'png', 'npy', or 'both'
        """
        files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            if self.load_file(file_path):
                z_spacing, _, _ = self.spacing
                if z_spacing > 10.0:
                    logger.warning(f"Skipping survey/scout scan: {filename}")
                    continue
                
                processed_hu = self.get_processed_volume(target_range=(0, 1))
                if processed_hu is None:
                    continue
                
                if output_npy_dir:
                    npy_path = os.path.join(output_npy_dir, f"{base_name}.npy")
                    np.save(npy_path, processed_hu)

                if output_png_dir:
                    png_path = os.path.join(output_png_dir, f"{base_name}.png")
                    png_data = (processed_hu * 255).astype(np.uint8)
                    self.save_as_png(png_data, png_path)
                    
    def process_all_conditions(self, root_dir, output_base_png, output_base_npy):
        """
        Iterates through all the branch of a specified path and converts files,
        recreating folder structure in output directories
        """
        for root, dirs, files in os.walk(root_dir):
            if not any(f.endswith('.dcm') for f in files): continue
            
            relative_path = os.path.relpath(root, root_dir)
            target_png = os.path.join(output_base_png, relative_path)
            target_npy = os.path.join(output_base_npy, relative_path)
            
            os.makedirs(target_png, exist_ok=True)
            os.makedirs(target_npy, exist_ok=True)
            
            self.batch_conversion(root, output_png_dir=target_png, output_npy_dir=target_npy)