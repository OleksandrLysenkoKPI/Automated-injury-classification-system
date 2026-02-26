import pydicom
import numpy as np
from PIL import Image

img_path = "DICOM_image_converter/test_data/series-00000/image-00006.dcm"
dicom_file = pydicom.dcmread(img_path)

image_pixels = dicom_file.pixel_array.astype(float)

rescaled_image = (np.maximum(image_pixels, 0)/image_pixels.max()) * 255 # float pixels
final_image = np.uint8(rescaled_image) # integer pixels

final_image = Image.fromarray(final_image)
final_image.show()
final_image.save('V1_image-00006.png')