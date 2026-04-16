from ..logger_module.logger import CustomLogger
from .ml_utils import get_dataset_paths
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import random
from pathlib import Path

logger = CustomLogger("DataLoader_log")

class Knee3DPathologyDataset(Dataset):
    def __init__(self, root_dir: str | Path, target_shape: tuple[int, int, int]=(16, 128, 128), is_train: bool = False):
        self.root_dir = Path(root_dir)
        self.target_shape = target_shape
        self.is_train = is_train
        self.samples = []
        
        try:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                class_idx = self.class_to_idx[class_name]
                for f in class_dir.glob("*.npy"):
                    self.samples.append((f, class_idx))
                    
            if not self.samples:
                raise FileNotFoundError(f"No .npy files found in {self.root_dir}")

            logger.info(f"Dataset initialized: {len(self.samples)} samples, {len(self.classes)} classes.")
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {e}")
            raise
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]
        
        try:
            data = np.load(file_path)
            tensor = torch.from_numpy(data).float()
            
            tensor = tensor.unsqueeze(0) # [D, H, W] -> [1, D, H,W]
            
            if self.is_train:
                if random.random() > 0.5:
                    tensor = torch.flip(tensor, dims=[-1])
                
                if random.random() > 0.5:
                    k = random.choice([1, 2, 3])
                    tensor = torch.rot90(tensor, k, dims=[-2, -1])
                    
                if random.random() > 0.5:
                    noise = torch.rand_like(tensor) * 0.01
                    tensor += noise
            
            tensor = tensor.unsqueeze(0) # [B, C, D, H, W]
            tensor = F.interpolate(tensor, size=self.target_shape, mode='trilinear', align_corners=False)
            tensor = tensor.squeeze(0) # [C, D, H, W]
            
            return tensor, label
        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            raise e


def load_dataset(target_shape: tuple[int, int, int], batch_size: int = 4):
    """Loads dataset and returns DataLoader objects with list of found classes"""
    try:
        paths = get_dataset_paths()

        train_dataset = Knee3DPathologyDataset(paths["train"], target_shape=target_shape, is_train=True)
        test_dataset = Knee3DPathologyDataset(paths["test"], target_shape=target_shape, is_train=False)
        
        class_names = train_dataset.classes
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Loaded {len(class_names)} classes: {class_names}")
        logger.info(f"Mapping: {train_dataset.class_to_idx}")
        
        logger.info("Dataset was loaded successfully")
        return train_loader, test_loader, class_names
    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        raise