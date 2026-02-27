import numpy as np
import matplotlib.pyplot as plt

# TODO: Add error handling
def verify_npy_conversion(original_hu, npy_path):
    """Compares original data (after rescale) with normilized NumPy file"""
    npy_pixels = np.load(npy_path)
    
    print("/n=== Verification Report ===")
    print(f"Original HU shape: {original_hu.shape}")
    print(f"NumPy shape:       {npy_pixels.shape}")
    
    correlation = np.corrcoef(original_hu.flatten(), npy_pixels.flatten())[0, 1]
    print(f"Data Correlation:  {correlation:.6f} (1.0 is perfect)")
    
    if original_hu.shape == npy_pixels.shape and correlation > 0.99:
        print("✅ Conversion was successful!")
    else:
        print("❌ Detected differences in data structure.")
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_hu, cmap='gray')
    axes[0].set_title("Original (HU)")
    axes[1].imshow(npy_pixels, cmap='gray')
    axes[1].set_title("Saved NumPy (Normalized)")
    plt.show()