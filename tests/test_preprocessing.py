import numpy as np
import torch

from src.imaging.image_preprocessing import resize_3d_tensor, get_knee_bbox, resample_3d, wavelet_denoising_3d

def test_resize_3d_tensor_padding():
    """
    Verifies that the function correctly applies center-aligned zero-padding <br>
    when the target dimensions are larger than the input tensor, ensuring data <br>
    integrity in the core.
    """
    input_tensor = torch.ones((1, 4, 4, 4))
    target_shape = (6, 6, 6)
    
    result = resize_3d_tensor(input_tensor, target_shape)
    
    assert result.shape == (1, 6, 6, 6)
    assert result[0, 1:5, 1:5, 1:5].sum() == input_tensor.sum()
    assert result[0, 0, 0, 0] == 0

def test_resize_3d_tensor_cropping():
    """
    Ensures that the function performs accurate center-aligned cropping <br>
    for tensors larger than the target shape, maintaining the central anatomical features.
    """
    input_tensor = torch.ones((1, 10, 10, 10))
    target_shape = (4, 4, 4)
    
    result = resize_3d_tensor(input_tensor, target_shape)
    
    assert result.shape == (1, 4, 4, 4)
    assert result.sum() == 4 * 4 * 4

def test_get_knee_bbox():
    """
    Validates the Region of Interest (ROI) extraction by checking <br>
    if the function can correctly identify and crop a specific <br> 
    non-zero voxel cluster within a "void" volume.
    """
    data = np.zeros((10, 10, 10))
    data[4:6, 4:6, 4:6] = 1.0
    
    result = get_knee_bbox(data, threshold=0.5)
    
    assert result.shape == (2, 2, 2)
    assert np.all(result == 1.0)

def test_get_knee_bbox_empty():
    """
    A safety check to ensure the function handles empty or low-signal volumes <br>
    gracefully by returning the original data if no voxels exceed the intensity threshold.
    """
    data = np.zeros((5, 5, 5))
    result = get_knee_bbox(data, threshold=0.1)
    assert result.shape == (5, 5, 5)

def test_resample_3d():
    """
    Tests the isotropic resampling logic, verifying that changing the physical <br>
    voxel spacing (e.g., from 2.0mm to 1.0mm) results in the correct <br>
    proportional change in array dimensions.
    """
    data = np.ones((4, 4, 4))
    current_spacing = (2.0, 2.0, 2.0)
    target_spacing = (1.0, 1.0, 1.0)
    
    result = resample_3d(data, current_spacing, target_spacing)
    
    assert result.shape == (8, 8, 8)

def test_wavelet_denoising_3d():
    """
    Confirms the denoising pipeline's stability, ensuring <br>
    the output maintains the original shape, correct floating-point type, and <br>
    stays within a normalized intensity range (0.0–1.0).
    """
    data = np.random.normal(0.5, 0.1, (16, 16, 16)).astype(np.float32)
    
    result = wavelet_denoising_3d(data)
    
    assert result.shape == data.shape
    assert result.dtype == np.float32
    assert np.max(result) <= 1.0
    assert np.min(result) >= 0.0