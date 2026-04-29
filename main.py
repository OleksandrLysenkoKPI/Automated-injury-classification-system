from src.imaging.image_converter import DICOMProcessor
from src.imaging.utils import verify_npy_conversion, split_data
from src.imaging.image_augmentation import augment_and_save_npy_dataset, augment_and_save_png_dataset
from src.ml_module.ml_utils import numpy_examiner
from src.ml_module.data_loader import load_dataset
from src.ml_module.ml_npy_model import start_npy_model_pipeline
from src.ml_module.ml_png_model import start_png_model_pipeline
from dotenv import load_dotenv
import os

def print_menu():
    print("\n" + "="*30)
    print("KNEE PATHOLOGY PIPELINE")
    print("="*30)
    print("1 -- Convert DICOM files from dataset")
    print("2 -- Verify DICOM to NumPy file conversion")
    print("3 -- Examine NumPy files")
    print("4 -- Split train data")
    print("5 -- Augment NumPy dataset")
    print("55 -- Augment PNG dataset")
    print("6 -- Load dataset")
    print("7 -- Start NPY model pipeline")
    print("77 -- Start PNG model pipeline")
    print("0 -- Exit")
    print("+"*30)
    print("9 -- 3 in 1: Convert, Split, Augment")
    print("10 -- 2 in 1: Split, Augment")
    print("-"*30)
    print("Choice: ", end="")

if __name__ == "__main__":
    
    load_dotenv()
    
    dataset_folder = os.getenv('KNEE_CONDITIONS_DATASET')
    converted_npy_folder = "data/converted_data/converted_NumPy"
    converted_png_folder = "data/converted_data/converted_PNG"
    prepared_data_folder = "data/prepared_data"
    
    npy_data_to_augment = "data/prepared_data/train/npy"
    png_data_to_augment = "data/prepared_data/train/png"
    
    test_dicom_image = os.getenv('TEST_DICOM_IMAGE')
    test_numpy_image = os.getenv('TEST_NUMPY_IMAGE')
    
    processor = DICOMProcessor()
    
    # High Fidelity (64, 160, 160), Standard Balanced (48, 224, 224), Deep MRI (96, 128, 128), Speed (32, 128, 128)
    target_shape = (32, 128, 128)
    target_spacing = (1.0, 1.0, 1.0)
    cache_in_ram = True
    batch_size = 64
    
    while True:
        print_menu()
        
        try:
            choice_input = input()
            if not choice_input.isdigit():
                print("!!! Please enter a valid number.")
                continue
            
            choice_input = int(choice_input)
            
            if choice_input == 1:
                processor.process_all_conditions(dataset_folder, converted_png_folder, converted_npy_folder, target_shape=target_shape, target_spacing=target_spacing)
            elif choice_input == 2:
                verify_npy_conversion(processor=processor, dicom_path=test_dicom_image, npy_path=test_numpy_image)
            elif choice_input == 3:
                numpy_examiner(converted_npy_folder, print_paths=False)
            elif choice_input == 4:
                split_data(converted_npy_folder, converted_png_folder, prepared_data_folder)
            elif choice_input == 5:
                augment_and_save_npy_dataset(npy_data_to_augment)
            elif choice_input == 55:
                augment_and_save_png_dataset(png_data_to_augment)
            elif choice_input == 6:
                load_dataset(batch_size=64, mode="npy", load_augmented=True, cache_in_ram=True)
            elif choice_input == 7:
                start_npy_model_pipeline(epochs=50, batch_size=4, mode="npy", save_file_name="knee_3d_pathology_model", use_augmented=True, cache_in_ram=cache_in_ram)
            elif choice_input == 77:
                start_png_model_pipeline(epochs=50, batch_size=128, mode="png", save_file_name="knee_2d_pathology_model", use_augmented=False, cache_in_ram=cache_in_ram)
            elif choice_input == 9:
                processor.process_all_conditions(dataset_folder, converted_png_folder, converted_npy_folder, target_shape=target_shape, target_spacing=target_spacing)
                split_data(converted_npy_folder, converted_png_folder, prepared_data_folder)
                augment_and_save_npy_dataset(npy_data_to_augment)
            elif choice_input == 10:
                split_data(converted_npy_folder, converted_png_folder, prepared_data_folder)
                augment_and_save_npy_dataset(npy_data_to_augment)
            elif choice_input == 0:
                print("Exiting program...")
                break
            else:
                print("!!! Invalid choice. Try again.")
        except Exception as e:
            print(f"\n[ERROR] An error occurred in main loop: {e}")
            print("Returning to menu...\n")