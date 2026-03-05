from utils.image_converter import DICOMProcessor
from utils.numpy_checker import verify_npy_conversion
import os



if __name__ == "__main__":
    input_folder = "data/test_data/series-00000"
    output_png_folder = "data/converted_PNG"
    output_npy_folder = "data/converted_NumPy"
    
    processor = DICOMProcessor()
    
    processor.batch_conversion(input_folder, output_png_folder, conversion_type="png")
    processor.batch_conversion(input_folder, output_npy_folder, conversion_type="npy")
    
    test_file = os.path.join(input_folder, "image-00006.dcm")
    if processor.load_file(test_file):
        npy_path = os.path.join(output_npy_folder, "image-00006.npy")
        processor.save_as_npy(npy_path)
        verify_npy_conversion(processor.pixels_hu, npy_path)