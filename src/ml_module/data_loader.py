from dotenv import load_dotenv
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from pathlib import Path
import logging
from ml_utils import log_section

def get_dataset_paths():
    """Loads and checks paths from environment"""
    load_dotenv()

    root_env = os.getenv("ROOT_KNEE_OSTEOARTHRITIS_DATASET_PATH")
    name_env = os.getenv("CURRENT_KNEE_OSTEOARTHRITIS_DATASET")

    if not root_env or not name_env:
        raise EnvironmentError("Dataset environment variables are not set properly.")
        
    root_path = Path(root_env)
    dataset_path = root_path / name_env
    return {
        "train": dataset_path / "train",
        "test": dataset_path / "test"
    }

def get_transformations():
    """Returns standard transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Повні значення для 3-х каналів ImageNet
    ])

def load_image_dataset(path: Path, transform):
    """Safely loads image dataset and logs information about it"""
    log_section("Image Data Loading")
    
    if not path.exists():
        logging.error(f"Directory not found: {path}")
        return None
    
    try:
        dataset = ImageFolder(root=str(path), transform=transform)
        logging.info(f"Successfully loaded dataset from {path}")
        logging.info(f"Images: {len(dataset)} | Classes: {dataset.classes}")
        return dataset
    except Exception as e:
        logging.error(f"Failed to initialize ImageFolder: {e}")
        return None

def get_data_loader(dataset, batch_size=32, shuffle=True):
    """Creates DataLoader and tests first batch load"""
    log_section("Getting Data Loader")
    
    if not dataset:
        logging.error("Cannot create DataLoader: Dataset is None")
        return None
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    try:
        images, labels = next(iter(loader))
        
        logging.info(f"Image batch shape: {images.shape} | Label batch shape: {labels.shape}")
        logging.info(f"Image dtype: {images.dtype} | Label dtype: {labels.dtype}")
        return loader
    except StopIteration:
        logging.error("Dataset is empty! Loader has no data to yield.")
        return None
    except Exception as e:
        logging.error(f"Error during DataLoader test: {e}")
        return None