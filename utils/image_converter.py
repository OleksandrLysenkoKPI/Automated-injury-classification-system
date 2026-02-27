import pydicom
from pydicom.pixels import apply_rescale
import numpy as np
from PIL import Image

def convert_dicom_to_png(dicom_path, output_path="Converted_DICOM.png"):
    ds = pydicom.dcmread(dicom_path)

    image_pixels = ds.pixel_array.astype(float)

    # Rescale Slope and Intercept
    image_hu = apply_rescale(image_pixels, ds)

    # Rescaling to 0-255
    img_min = image_hu.min()
    img_max = image_hu.max()
    norm_image = ((image_hu - img_min) / (img_max - img_min)) * 255
        
    final_image = Image.fromarray(norm_image.astype(np.uint8))
    final_image.save(output_path)
    print(f"Saved as PNG: {output_path}")


def convert_dicom_to_npy(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    
    image_hu = apply_rescale(ds.pixel_array, ds).astype(np.float32)
    
    # Normalization (0.0 to 1.0)
    img_min = image_hu.min()
    img_max = image_hu.max()
    
    if img_max - img_min != 0:
        normalized = (image_hu - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image_hu)
        
    np.save(output_path, normalized)
    print(f"Saved as NumPy: {output_path}")