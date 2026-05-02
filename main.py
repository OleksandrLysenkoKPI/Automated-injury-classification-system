from src.imaging.image_converter import DICOMProcessor
from src.imaging.utils import verify_npy_conversion, split_data
from src.imaging.image_augmentation import augment_and_save_npy_dataset, augment_and_save_png_dataset
from src.ml_module.ml_utils import numpy_examiner
from src.ml_module.data_loader import load_dataset
from src.ml_module.ml_npy_model import start_npy_model_pipeline
from src.ml_module.ml_png_model import start_png_model_pipeline, start_stage2_png_pipeline
from src.ml_module.ml_npy_model import start_npy_model_pipeline, start_stage2_npy_pipeline
from dotenv import load_dotenv
import os

def print_menu():
    print("\n" + "="*30)
    print("KNEE PATHOLOGY PIPELINE")
    print("="*30)
    print("1 -- Convert DICOM files from datasets")
    print("2 -- Examine NumPy files")
    print("3 -- Split train data")
    print("4 -- Load dataset")
    print("5 -- Start NPY model pipeline Stage 1")
    print("6 -- Start NPY model pipeline Stage 2")
    print("7 -- Start PNG model pipeline Stage 1")
    print("8 -- Start PNG model pipeline Stage 2")
    print("0 -- Exit")
    print("-"*30)
    print("Choice: ", end="")

if __name__ == "__main__":
    
    load_dotenv()
    
    conditions_dataset_folder = os.getenv('KNEE_CONDITIONS_DATASET')
    knee_dataset_folder = os.getenv('KNEE_DATASET')
    
    converted_npy_folder = "data/converted_data/converted_NumPy"
    converted_png_folder = "data/converted_data/converted_PNG"
    prepared_data_folder = "data/prepared_data"
    
    npy_data_to_train = "data/prepared_data/train/npy"
    png_data_to_train = "data/prepared_data/train/png"
    
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
                processor.process_all_conditions(
                    conditions_root_dir=conditions_dataset_folder, 
                    knee_root_dir=knee_dataset_folder, 
                    output_base_png=converted_png_folder, 
                    output_base_npy=converted_npy_folder, 
                    target_shape=target_shape, 
                    target_spacing=target_spacing
                )
            elif choice_input == 2:
                numpy_examiner(converted_npy_folder, print_paths=False)
            elif choice_input == 3:
                split_data(converted_npy_folder, converted_png_folder, prepared_data_folder)
            elif choice_input == 4:
                load_dataset(base_data_path=prepared_data_folder, batch_size=64, mode="npy", cache_in_ram=True)
            elif choice_input == 5:
                start_npy_model_pipeline(
                    epochs=40, 
                    batch_size=4, 
                    mode="npy", 
                    save_file_name="knee_3d_binary_model", 
                    cache_in_ram=cache_in_ram
                )
            elif choice_input == 6:
                start_stage2_npy_pipeline(
                    binary_model_path="knee_3d_binary_model.pth",
                    epochs=60,
                    batch_size=4,
                    save_file_name="knee_3d_stage2_6classes",
                    cache_in_ram=True
                )
            
            elif choice_input == 7:
                start_png_model_pipeline(
                    epochs=40, 
                    batch_size=32, 
                    mode="png", 
                    save_file_name="knee_2d_binary_model",
                    cache_in_ram=cache_in_ram
                )
            elif choice_input == 8:
                start_stage2_png_pipeline(
                    binary_model_path="knee_2d_binary_model.pth",
                    epochs=40, 
                    batch_size=64, 
                    save_file_name="knee_stage2_6classes",
                    cache_in_ram=cache_in_ram
                )
            elif choice_input == 0:
                print("Exiting program...")
                break
            else:
                print("!!! Invalid choice. Try again.")
        except Exception as e:
            print(f"\n[ERROR] An error occurred in main loop: {e}")
            print("Returning to menu...\n")