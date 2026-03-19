import numpy as np
import matplotlib.pyplot as plt
import os
import logging

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
        logging.error(f"Verification error: {e}")
        return False