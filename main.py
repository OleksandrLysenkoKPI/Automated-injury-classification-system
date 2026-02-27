from utils.image_converter import DICOMProcessor
from utils.numpy_checker import verify_npy_conversion



if __name__ == "__main__":
    img_path = "data/test_data/series-00000/image-00006.dcm"
    output_name = "image-00006"
    
    processor = DICOMProcessor(img_path)
    
    processor.save_as_png(f"data/converted_PNG/{output_name}.png")
    processor.save_as_npy(f"data/converted_NumPy/{output_name}.npy")
    
    verify_npy_conversion(processor.pixels_hu, f"data/converted_NumPy/{output_name}.npy")