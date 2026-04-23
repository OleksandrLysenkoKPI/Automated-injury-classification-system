from ..logger_module.logger import CustomLogger
from .ml_utils import get_dataset_paths, verify_dataset_processing
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import torch
import random
from pathlib import Path

logger = CustomLogger("DataLoader_log")

class Knee3DPathologyDataset(Dataset):
    def __init__(self, root_dir: str | Path, target_shape: tuple[int, int, int], is_train: bool = False):
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
    
    def _resize_3d(self, tensor: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        c, d, h, w = tensor.shape
        td, th, tw = target_shape
        
        def get_coords(current, target):
            if current == target:
                return 0, 0, None
            elif current > target:
                start = (current - target) // 2
                return start, start + target, True # True = crop
            else:
                pad_total = target - current
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                return pad_before, pad_after, False # False = pad
        
        d_start, d_end, d_crop = get_coords(d, td)
        h_start, h_end, h_crop = get_coords(h, th)
        w_start, w_end, w_crop = get_coords(w, tw)
        
        if d_crop:
            tensor = tensor[:, d_start:d_end, :, :]
        else:
            tensor = F.pad(tensor, (0, 0, 0, 0, d_start, d_end), mode='constant', value=0)

        if h_crop:
            tensor = tensor[:, :, h_start:h_end, :]
        else:
            tensor = F.pad(tensor, (0, 0, h_start, h_end, 0, 0), mode='constant', value=0)
            
        if w_crop:
            tensor = tensor[:, :, :, w_start:w_end]
        else:
            tensor = F.pad(tensor, (w_start, w_end, 0, 0, 0, 0), mode='constant', value=0)
        
        return tensor
    
    def _apply_augmentations(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            tensor = torch.flip(tensor, dims=[-1]) # -1 = W
     
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            tensor = TF.rotate(tensor, angle)

        return tensor
    
    def __getitem__(self, index):
        file_path, label = self.samples[index]

        try:
            data = np.load(file_path)
            tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0) # [D, H, W] -> [B=1, C=1, D, H,W]
            
            current_depth = tensor.shape[2]
            temp = F.interpolate(tensor, size=(current_depth, 224, 224), mode='trilinear', align_corners=False)
            tensor = temp.squeeze(0)
            tensor = self._resize_3d(tensor, self.target_shape)
            
            if self.is_train:
                tensor = self._apply_augmentations(tensor)
            
            if tensor.max() == 0:
                logger.warning(f"Warning: Sample {file_path} is empty after processing!")
            
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
            return tensor, label
        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            raise e


def load_dataset(target_shape: tuple[int, int, int], batch_size: int = 4, load_augmented: bool = False):
    """Loads dataset and returns DataLoader objects with list of found classes"""
    try:
        paths = get_dataset_paths()
        
        train_path = paths["train_augmented"] if load_augmented else paths["train"]
        
        train_dataset = Knee3DPathologyDataset(train_path, target_shape=target_shape, is_train=True)
        val_dataset = Knee3DPathologyDataset(paths["val"], target_shape=target_shape, is_train=False)
        test_dataset = Knee3DPathologyDataset(paths["test"], target_shape=target_shape, is_train=False)
        
        class_names = train_dataset.classes
        
        # verify_dataset_processing(train_dataset, sample_idx=967)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Loaded {len(class_names)} classes: {class_names}")

        logger.info("Dataset was loaded successfully")
        return train_loader,val_loader, test_loader, class_names
    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        raise