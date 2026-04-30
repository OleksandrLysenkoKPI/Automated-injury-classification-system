import os
import struct
import pydicom
from pydicom.pixels.processing import apply_rescale
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .image_preprocessing import wavelet_denoising_3d, resample_3d, get_knee_bbox, resize_3d_tensor
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
            self.ds = pydicom.dcmread(dicom_path, force=True)
        
            pixel_tags = ['PixelData', 'FloatPixelData', 'DoubleFloatPixelData']
            if not any(tag in self.ds for tag in pixel_tags):
                logger.warning(f"Skipping non-image DICOM (Report/SR): {os.path.basename(dicom_path)}")
                return False

            _ = self.ds.pixel_array
            
            self._pixels_hu = None
            return True
        except AttributeError:
            logger.warning(f"File has no pixel attribute: {os.path.basename(dicom_path)}")
            return False
        except (OSError, struct.error) as e:
            # Handled "No tag to read" and "unpack requires a buffer"
            logger.warning(f"Corrupted structure in {os.path.basename(dicom_path)}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in {os.path.basename(dicom_path)}: {e}")
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
        data = data.astype(np.float32)
        
        lower_bound = np.percentile(data, clamping_percentile[0])
        upper_bound = np.percentile(data, clamping_percentile[1])
        
        data = np.clip(data, lower_bound, upper_bound)
        img_min, img_max = data.min(), data.max()
        
        if img_max - img_min == 0:
            return np.zeros_like(data)
        
        normalized = (data - img_min) / (img_max - img_min)
        
        if target_range == (0, 255):
            return (normalized * 255).astype(np.uint8)
        
        return normalized.astype(np.float16)
    
    def get_processed_volume(self, target_range=(0, 1), target_spacing=(1.0, 1.0, 1.0), target_shape=(64, 160, 160)):
        """Applies the full preprocessing pipeline to the current DICOM object."""
        try:
            data = self.pixels_hu.astype(np.float32)
            
            data = wavelet_denoising_3d(data)
            
            logger.info(f"Resampling from {self.spacing} to {target_spacing}...")
            data = resample_3d(data, self.spacing, target_spacing=target_spacing)
            data = get_knee_bbox(data, threshold=0.01)

            tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
            
            current_depth = tensor.shape[2]
            tensor = F.interpolate(tensor, size=(current_depth, 224, 224), mode='trilinear')
        
            tensor = resize_3d_tensor(tensor.squeeze(0), target_shape)
            final_data_f32 = tensor.squeeze(0).numpy()
            
            normalized_data_f16 = self.get_normalized(final_data_f32, target_range=target_range)
            return normalized_data_f16
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
        return None

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
    
    def batch_conversion(self, input_dir, output_png_dir=None, output_npy_dir=None, 
                         start_idx: int = 1, target_shape: tuple[int, int, int] = (64, 160, 160), target_spacing: tuple[float, float, float]=(1.0, 1.0, 1.0)):
        """
        Iterates through a folder and converts all DICOM files to png and npy.
        """
        files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
        current_idx = start_idx
        
        for filename in files:
            file_path = os.path.join(input_dir, filename)
            
            if self.load_file(file_path):
                z_spacing, _, _ = self.spacing
                if z_spacing > 10.0:
                    logger.warning(f"Skipping survey/scout scan: {filename}")
                    continue
                
                processed_hu = self.get_processed_volume(target_range=(0, 1), target_shape=target_shape, target_spacing=target_spacing)
                if processed_hu is None:
                    continue
                
                base_name = f"{current_idx:04d}"

                if output_npy_dir:
                    npy_path = os.path.join(output_npy_dir, f"{base_name}.npy")
                    np.save(npy_path, processed_hu)

                if output_png_dir:
                    png_path = os.path.join(output_png_dir, f"{base_name}.png")
                    png_data = (processed_hu * 255).astype(np.uint8)
                    self.save_as_png(png_data, png_path)
                
                current_idx += 1
        
        return current_idx
                    
    @staticmethod
    def extract_identity(folder_name: str):
        parts = folder_name.upper().split('I')
        raw_name = parts[0].strip()
        
        side = ""
        if folder_name.upper().endswith('L'):
            side = "_L"
        elif folder_name.upper().endswith('R'):
            side = "_R"
            
        return raw_name, side
    
    @staticmethod
    def process_single_patient(patient_path, dataset_name, condition_name, patient_label, output_base_png, output_base_npy, target_shape, target_spacing):
        processor = DICOMProcessor()
        sub_path = os.path.join(dataset_name, condition_name or "", patient_label)
        
        target_png = Path(output_base_png) / sub_path
        target_npy = Path(output_base_npy) / sub_path
        target_png.mkdir(parents=True, exist_ok=True)
        target_npy.mkdir(parents=True, exist_ok=True)
        
        file_counter = 1
        for current_root, _, files in os.walk(patient_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if dcm_files:
                file_counter = processor.batch_conversion(
                    current_root,
                    output_png_dir=target_png,
                    output_npy_dir=target_npy,
                    start_idx=file_counter,
                    target_shape=target_shape,
                    target_spacing=target_spacing
                )
        return file_counter - 1
    
    def process_all_conditions(self, conditions_root_dir, knee_root_dir, output_base_png, output_base_npy, target_shape=(64, 160, 160), target_spacing=(1.0, 1.0, 1.0)):
        """
        Iterates through all the branch of a specified path and converts files,
        recreating folder structure without a timestamp in folder directories.
        """
        patient_name_to_id = {}
        next_id = 1
        
        all_folders_to_process = []
        
        def collect_folders(root, dataset_tag, condition=None):
            nonlocal next_id
            if not Path(root).exists(): return
            
            for p_path in sorted(Path(root).iterdir()):
                if p_path.is_dir():
                    name_key, side = self.extract_identity(p_path.name)
                    
                    if name_key not in patient_name_to_id:
                        patient_name_to_id[name_key] = next_id
                        next_id += 1
                    
                    assigned_id = patient_name_to_id[name_key]
                    all_folders_to_process.append({
                        "path": p_path,
                        "dataset": dataset_tag,
                        "condition": condition,
                        "id": assigned_id,
                        "side": side
                    })

        c_root = Path(conditions_root_dir)
        if c_root.exists():
            for cond_path in [d for d in c_root.iterdir() if d.is_dir()]:
                collect_folders(cond_path, "conditions_dataset", cond_path.name)
        
        collect_folders(knee_root_dir, "knee_dataset", "healthy")

        with ProcessPoolExecutor() as executor:
            all_futures = []
            for item in all_folders_to_process:
                patient_label = f"patient#{item['id']}{item['side']}"
                
                all_futures.append(
                    executor.submit(
                        self.process_single_patient,
                        item["path"], item["dataset"],
                        item["condition"], patient_label,
                        output_base_png, output_base_npy,
                        target_shape, target_spacing
                    )
                )
            
            total_files = sum(f.result() for f in all_futures)
            logger.info(f"Обробка завершена. Анонімізовано {next_id-1} пацієнтів. Файлів: {total_files}")