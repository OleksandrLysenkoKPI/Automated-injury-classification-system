from src.imaging.image_converter import DICOMProcessor
from src.imaging.utils import verify_npy_conversion, augment_and_save_dataset, split_data
from src.ml_module.ml_utils import numpy_examinator
from src.ml_module.data_loader import load_dataset
from src.ml_module.ml_model import start_model_pipeline
from dotenv import load_dotenv
import os
import sys

if __name__ == "__main__":
    
    load_dotenv()
    
    dataset_folder = os.getenv('KNEE_CONDITIONS_DATASET')
    output_png_folder = "data/converted_data/converted_PNG"
    output_npy_folder = "data/converted_data/converted_NumPy"
    
    test_dicom_image = os.getenv('TEST_DICOM_IMAGE')
    test_numpy_image = os.getenv('TEST_NUMPY_IMAGE')
    
    processor = DICOMProcessor()
    
    print("1 -- Convert DICOM files from dataset")
    print("2 -- Verify DICOM to NumPy file conversion")
    print("3 -- Examine NumPy file")
    print("4 -- Split train data")
    print("5 -- Augment NumPy dataset")
    print("6 -- Load dataset")
    print("7 -- Start model pipeline")
    choice_input = int(input())
    
    if choice_input == 1:
        processor.process_all_conditions(dataset_folder, output_png_folder, output_npy_folder)
    elif choice_input == 2:
        verify_npy_conversion(processor=processor, dicom_path=test_dicom_image, npy_path=test_numpy_image)
    elif choice_input == 3:
        numpy_examinator('data/prepared_data')
    elif choice_input == 4:
        split_data('data/prepared_data/train')
    elif choice_input == 5:
        augment_and_save_dataset('data/prepared_data/train_split')
    elif choice_input == 6:
        target_shape = (32, 256, 256)
        load_dataset(target_shape, batch_size=4, load_augmented=True)
    elif choice_input == 7:
        start_model_pipeline(epochs=30, target_shape=(24, 192, 192), save_file_name="knee_3d_pathology_model", use_augmented=True)
    else:
        print("Exiting program...")
        sys.exit(0)
    