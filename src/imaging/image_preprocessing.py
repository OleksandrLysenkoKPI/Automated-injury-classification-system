import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from skimage.restoration import denoise_wavelet

def resize_3d_tensor(tensor: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """
    Performs centralized spatial normalization via cropping or padding.
    
    Ensures the 3D tensor matches the target dimensions required by the <br>
    neural network. If the input is larger, it crops the center; if smaller, <br>
    it applies symmetric zero-padding to maintain anatomical centering.
    """
    c, d, h, w = tensor.shape
    td, th, tw = target_shape
    
    def get_coords(current, target):
        if current == target:
            return 0, 0, None
        elif current > target:
            start = (current - target) // 2
            return start, start + target, True # True = crop
        else:
            pad_total = target - current
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            return pad_before, pad_after, False # False = pad
    
    d_start, d_end, d_crop = get_coords(d, td)
    h_start, h_end, h_crop = get_coords(h, th)
    w_start, w_end, w_crop = get_coords(w, tw)
    
    if d_crop:
        tensor = tensor[:, d_start:d_end, :, :]
    else:
        tensor = F.pad(tensor, (0, 0, 0, 0, d_start, d_end), mode='constant', value=0)

    if h_crop:
        tensor = tensor[:, :, h_start:h_end, :]
    else:
        tensor = F.pad(tensor, (0, 0, h_start, h_end, 0, 0), mode='constant', value=0)
        
    if w_crop:
        tensor = tensor[:, :, :, w_start:w_end]
    else:
        tensor = F.pad(tensor, (w_start, w_end, 0, 0, 0, 0), mode='constant', value=0)
    
    return tensor

def get_knee_bbox(data, threshold=0.01):
    """
    Extracts the anatomical region of interest (ROI) by removing empty space.
    
    Identifies the bounding box containing the knee joint based on pixel <br>
    intensity thresholds. This reduces the input volume to the relevant tissues.
    """
    coords = np.argwhere(data > threshold)
    if coords.size == 0:
        return data
    
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0) + 1
    
    return data[d0:d1, h0:h1, w0:w1]

def resample_3d(data, current_spacing, target_spacing=(1.0, 1.0, 1.0)):
    """
    Standardizes the physical resolution (voxel spacing) of the 3D volume.
    
    Uses trilinear interpolation (zoom) to adjust the volume so that each <br>
    voxel represents a uniform physical size (e.g., 1x1x1 mm).
    """
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    
    resampled_data = zoom(data, scale_factors, order=3, mode='constant', cval=0.0)
    
    return resampled_data

def wavelet_denoising_3d(data):
    """
    Applies advanced noise reduction using 3D Wavelet Transformation.
    
    Utilizes the 'BayesShrink' method and soft-thresholding to remove
    MRI-specific noise.
    """
    lower = np.percentile(data, 1)
    upper = np.percentile(data, 99)
    data = np.clip(data, lower, upper)
    
    denoised_data = denoise_wavelet(
        data,
        method='BayesShrink',
        mode='soft',
        wavelet='db2',
        rescale_sigma=True
    )
    
    return denoised_data.astype(np.float32)