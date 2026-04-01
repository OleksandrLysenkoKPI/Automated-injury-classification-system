import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from ..logger_module.logger import CustomLogger
from ml_utils import get_dataset_paths

logger = CustomLogger("DataLoader_log")

# TODO: Have to write custom Dataset
class Knee3DPathologyDataset(Dataset):
    def __init__(self, root_dir, target_shape=(16, 128, 128)):
        self.root_dir = Path(root_dir)
        self.target_shape = target_shape
        
        # TODO: 1. Get folders
        # TODO: 2. Get .npy files paths and their labels (their respective folders)
    
    # TODO: 4. Load NumPy arrays
    # TODO: 5. Interpolate arrays
    # TODO: 6. Resize arrays