from src.imaging.image_converter import DICOMProcessor
from src.imaging.utils import verify_npy_conversion, augment_and_save_dataset, split_data
from src.ml_module.ml_utils import numpy_examiner
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
    print("3 -- Examine NumPy files")
    print("4 -- Split train data")
    print("5 -- Augment NumPy dataset")
    print("6 -- Load dataset")
    print("7 -- Start model pipeline")
    choice_input = int(input())
    
    # High Fidelity (64, 160, 160), Standard Balanced (48, 224, 224), Deep MRI (96, 128, 128)
    target_shape = (64, 160, 160)
    target_spacing = (1.0, 1.0, 1.0)
    
    if choice_input == 1:
        processor.process_all_conditions(dataset_folder, output_png_folder, output_npy_folder, target_shape=target_shape, target_spacing=target_spacing)
    elif choice_input == 2:
        verify_npy_conversion(processor=processor, dicom_path=test_dicom_image, npy_path=test_numpy_image)
    elif choice_input == 3:
        numpy_examiner('data/converted_data/converted_NumPy', print_paths=False)
    elif choice_input == 4:
        split_data('data/converted_data/converted_NumPy')
    elif choice_input == 5:
        augment_and_save_dataset('data/prepared_data/train')
    elif choice_input == 6:
        load_dataset(target_shape=target_shape, batch_size=4, load_augmented=True, verify_processing=True, img_idx=10)
    elif choice_input == 7:
        start_model_pipeline(epochs=50, batch_size=8, target_shape=target_shape, save_file_name="knee_3d_pathology_model", use_augmented=True)
    else:
        print("Exiting program...")
        sys.exit(0)