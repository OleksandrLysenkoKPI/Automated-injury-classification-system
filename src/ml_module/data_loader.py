from ..logger_module.logger import CustomLogger
from .ml_utils import get_dataset_paths
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F
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
        
    def _pad_or_crop_depth(self, tensor: torch.Tensor, target_depth: int) -> torch.Tensor:
        d = tensor.shape[1] # tensor shape is [1, D, H, W]
        
        if d == target_depth:
            return tensor
        
        if d > target_depth:
            start_d = (d - target_depth) // 2 # Center crop depth
            return tensor[:, start_d:start_d + target_depth :, :]
        else:
            # Pad Depth with 0
            pad_total = target_depth - d
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(tensor, (0, 0, 0, 0, pad_left, pad_right), mode="constant", value=0)
        
    def __getitem__(self, index):
        file_path, label = self.samples[index]

        try:
            data = np.load(file_path)
            tensor = torch.from_numpy(data).float()
            tensor = tensor.unsqueeze(0) # [D, H, W] -> [1, D, H,W]           
            
            temp = tensor.unsqueeze(0) # [1, 1, D, H, W]
            temp = F.interpolate(temp, size=(tensor.shape[1], self.target_shape[1], self.target_shape[2]), mode='trilinear', align_corners=False)
            
            tensor = temp.squeeze(0) # [1, D, H, W]
            
            tensor = self._pad_or_crop_depth(tensor, self.target_shape[0])
            
            return tensor, label
        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            raise e


def load_dataset(target_shape: tuple[int, int, int], batch_size: int = 4, load_augmented: bool = False):
    """Loads dataset and returns DataLoader objects with list of found classes"""
    try:
        paths = get_dataset_paths()
        
        if load_augmented:
            train_dataset = Knee3DPathologyDataset(paths["train_augmented"], target_shape=target_shape, is_train=True)
            val_dataset = Knee3DPathologyDataset(paths["val"], target_shape=target_shape, is_train=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            train_dataset =  Knee3DPathologyDataset(paths["train"], target_shape=target_shape, is_train=True)
        test_dataset = Knee3DPathologyDataset(paths["test"], target_shape=target_shape, is_train=False)
        
        class_names = train_dataset.classes
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Loaded {len(class_names)} classes: {class_names}")
        logger.info(f"Mapping: {train_dataset.class_to_idx}")
        
        logger.info("Dataset was loaded successfully")
        return train_loader,val_loader, test_loader, class_names
    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        raise