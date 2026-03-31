from dotenv import load_dotenv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from pathlib import Path
from ..logger_module.logger import CustomLogger


logger = CustomLogger("DataLoader_log")

def get_dataset_paths():
    """Loads and checks paths from environment"""
    load_dotenv()

    dataset_env = os.getenv("PREPARED_KNEE_DATASET")

    if not dataset_env:
        raise EnvironmentError("Dataset environment variables are not set properly.")
        
    dataset_path = Path(dataset_env)
    return {
        "train": dataset_path / "train",
        "test": dataset_path / "test"
    }


# TODO: Rewrite all code below

def get_transformations():
    """Returns standard transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Повні значення для 3-х каналів ImageNet
    ])

def load_dataset(path: Path, transform):
    """Safely loads dataset and logs information about it"""
    if not path.exists():
        logger.error(f"Directory not found: {path}")
        return None
    
    try:
        logger.info(f"Successfully loaded dataset from {path}")
        logger.info(f"Images: {len(dataset)} | Classes: {dataset.classes}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to initialize ImageFolder: {e}")
        return None

def get_data_loader(dataset, batch_size=32, shuffle=True):
    """Creates DataLoader and tests first batch load"""
    if not dataset:
        logger.error("Cannot create DataLoader: Dataset is None")
        return None
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    try:
        images, labels = next(iter(loader))
        
        logger.info(f"Image batch shape: {images.shape} | Label batch shape: {labels.shape}")
        logger.info(f"Image dtype: {images.dtype} | Label dtype: {labels.dtype}")
        return loader
    except StopIteration:
        logger.error("Dataset is empty! Loader has no data to yield.")
        return None
    except Exception as e:
        logger.error(f"Error during DataLoader test: {e}")
        return None