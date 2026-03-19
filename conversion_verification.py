from src.imaging.image_converter import DICOMProcessor
from src.imaging.utils import verify_npy_conversion
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    
    load_dotenv()
    
    dataset_folder = os.getenv('KNEE_CONDITIONS_DATASET')
    output_png_folder = "data/converted_PNG"
    output_npy_folder = "data/converted_NumPy"
    
    processor = DICOMProcessor()
    
    processor.process_all_conditions(dataset_folder, output_png_folder, output_npy_folder)
    