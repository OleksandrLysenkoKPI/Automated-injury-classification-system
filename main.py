from utils.image_converter import *

def dicom_conversion(img_path, output_path):
    convert_dicom_to_png(img_path, f"{output_path}.png")
    convert_dicom_to_npy(img_path, f"{output_path}.npy")

img_path = "data/test_data/series-00000/image-00006.dcm"
output = "image-00006"

# dicom_conversion(img_path, output)
