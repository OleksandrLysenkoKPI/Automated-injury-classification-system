from ..logger_module.logger import CustomLogger
from .ml_utils import get_dataset_paths
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logger = CustomLogger("DataLoader_log")

class Knee3DPathologyDataset(Dataset):
    def __init__(self, root_dir: str | Path, target_shape: tuple[int, int, int]=(16, 128, 128)):
        self.root_dir = Path(root_dir)
        self.target_shape = target_shape
        self.samples = []
        
        try:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                class_idx = self.class_to_idx[class_name]
                for f in class_dir.glob("*.npy"):
                    self.samples.append((f, class_idx))
                    
            if not self.samples:
                msg = f"No .npy files found in {root_dir}"
                logger.error(msg)
                raise FileNotFoundError(msg)

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
            
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(tensor, size=self.target_shape, mode='trilinear', align_corners=False)
            tensor = tensor.squeeze(0)
            
            return tensor, label
        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            raise e


def load_dataset(target_shape: tuple[int, int, int]):
    """Loads dataset and returns DataLoader objects"""
    try:
        paths = get_dataset_paths()

        train_dataset = Knee3DPathologyDataset(paths["train"], target_shape=target_shape)
        test_dataset = Knee3DPathologyDataset(paths["test"], target_shape=target_shape)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        logger.info(f"Classes found: {train_dataset.classes}")
        logger.info(f"Mapping: {train_dataset.class_to_idx}")
        
        logger.info("Dataset was loaded successfully")
        return train_loader, test_loader
    except Exception as e:
        logger.error("Error during dataset loading: {e}")