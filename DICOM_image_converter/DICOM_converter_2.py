import pydicom
from pydicom.pixels import apply_rescale
import numpy as np
from PIL import Image

img_path = "DICOM_image_converter/test_data/series-00000/image-00006.dcm"
dicom_file = pydicom.dcmread(img_path)

image_pixels = dicom_file.pixel_array.astype(float)

# Rescale Slope and Intercept
image_hu = apply_rescale(image_pixels, dicom_file)

# Improved Rescaling to 0-255
img_min = image_hu.min()
img_max = image_hu.max()
norm_image = ((image_hu - img_min) / (img_max - img_min)) * 255
    
final_image = Image.fromarray(norm_image.astype(np.uint8))
final_image.show()
final_image.save("V2_image-00006.png")