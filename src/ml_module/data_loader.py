import torchvision.transforms.functional as TF
import cv2
import random
import numpy as np
import torch
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .ml_utils import get_dataset_paths, verify_dataset_processing
from ..logger_module.logger import CustomLogger

logger = CustomLogger("DataLoader_log")

def is_valid_slice(img_array: np.ndarray, std_threshold: float = 10.0):
    """Checks if slice is empty (black)"""
    if img_array is None: return False
    return np.std(img_array.astype(np.float32)) > std_threshold

class KneeDataset(Dataset):
    def __init__(self, root_dir: str | Path, mode: str = "png", stage: int = 1, is_train: bool = False, cache_in_ram: bool = False):
        self.root_dir = Path(root_dir)
        self.mode = mode.lower()
        self.stage = stage
        self.is_train = is_train
        self.cache_in_ram = cache_in_ram
        
        self.samples = []
        self.cached_data = []
        self.labels = []
        
        # Binary classification mapping
        if self.stage == 1:
            self.group_map = {
                'healthy': 0,
                'гонартроз': 1, 
                'хондромаляція виростків': 1, 
                'хондромаляція надколінка': 1,
                'меніски': 1, 
                'часткове пошкодження пхз': 1, 
                'медіапателярна складка': 1
            }
            self.classes = ['Healthy', 'Pathology']
        elif self.stage == 2:
            self.group_map = {
            'гонартроз': 0, 
            'хондромаляція виростків': 1, 
            'хондромаляція надколінка': 2,
            'меніски': 3, 
            'часткове пошкодження пхз': 4, 
            'медіапателярна складка': 5
            }
            self.classes = [
                'Гонартроз', 'Хондромаляція_виростків', 'Хондромаляція_надколінка',
                'Меніски', 'Часткове_пошкодження_пхз', 'Медіапателярна_складка'
                ]
        else:
            logger.error(f"Stage {self.stage} does not exist, there are only 2 stages!")
            raise
        
        if self.is_train and self.mode == "png":
            self.train_transforms = v2.Compose([
                v2.RandomResizedCrop(size=128, scale=(0.9, 1.0), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-10, 10)),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.train_transforms = None
            
        self.normalize = v2.Normalize(mean=[0.449], std=[0.226])
        
        self._build_dataset()
        
        
    def _build_dataset(self):
        """Collects path files and filters out empty slices"""
        if not self.root_dir.exists():
            logger.warning(f"Path not found: {self.root_dir}")
            return

        total_files_found = 0
        for condition_folder in [d for d in self.root_dir.iterdir() if d.is_dir()]:
            condition_name = condition_folder.name.lower()
            
            if condition_name not in self.group_map:
                continue
                
            label = self.group_map[condition_name]
            ext = f"*.{self.mode}"
            all_files = list(condition_folder.rglob(ext))
            
            for f in all_files:
                if self.mode == "png":
                    try:
                        img_data = np.fromfile(str(f), dtype=np.uint8)
                        img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None and is_valid_slice(img):
                            self.samples.append((f, label))
                            total_files_found += 1
                    except Exception as e:
                        continue
                else:
                    self.samples.append((f, label))
                    total_files_found += 1

        logger.info(f"Initialized {self.mode.upper()} dataset: {len(self.samples)} files in {len(self.classes)} groups.")
        
        if self.cache_in_ram and self.samples:
            self._load_to_ram()
    
    def _load_file(self, file_path: Path):
        """Multi format file reading"""
        try:
            if self.mode == 'npy':
                data = np.load(file_path).astype(np.float32)
                tensor = torch.from_numpy(data)
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
                return tensor
            else:
                # PNG: np.fromfile to work with cyrillic
                img_data = np.fromfile(str(file_path), dtype=np.uint8)
                data = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
                
                if data is None: # PIL as a fallback
                    with Image.open(file_path) as img:
                        data = np.array(img.convert('L'))
                
                data = data.astype(np.float32) / 255.0
                return torch.from_numpy(data).unsqueeze(0)
        except Exception as e:
            logger.error(f"Loading error {file_path}: {e}")
            return torch.zeros((1, 128, 128))

    def _load_to_ram(self):
        logger.info(f"Caching {self.mode} data into RAM...")
        for file_path, label in self.samples:
            self.cached_data.append(self._load_file(file_path))
            self.labels.append(label)
        logger.info("RAM caching complete.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.cache_in_ram:
            tensor = self.cached_data[index].clone()
            label = self.labels[index]
        else:
            file_path, label = self.samples[index]
            tensor = self._load_file(file_path)

        # Png augmentation in train mode
        if self.mode == "png":
            if self.is_train and self.train_transforms:
                tensor = self.train_transforms(tensor)
                # Noise
                if torch.rand(1) > 0.5:
                    tensor += torch.randn_like(tensor) * random.uniform(0.001, 0.005)
            tensor = self.normalize(tensor)
            
        return tensor.float(), label

def load_dataset(base_data_path: str | Path, batch_size: int = 16, mode: str = "png", stage: int = 1, cache_in_ram: bool = False):
    """Loads dataset and returns DataLoader objects with list of found classes"""
    try:
        base_path = Path(base_data_path)
        
        train_path = base_path / "train" / mode
        val_path = base_path / "val" / mode
        test_path = base_path / "test" / mode

        train_ds = KneeDataset(train_path, mode=mode, is_train=True, stage=stage, cache_in_ram=cache_in_ram)
        val_ds = KneeDataset(val_path, mode=mode, is_train=False, stage=stage, cache_in_ram=cache_in_ram)
        test_ds = KneeDataset(test_path, mode=mode, is_train=False, stage=stage, cache_in_ram=False)

        if len(train_ds) == 0:
            raise ValueError(f"The training set is empty at {base_path}/train/{mode}. Check split_data!")
        
        # WeightedRandomSampler balancing
        target_list = torch.tensor([s[1] for s in train_ds.samples], dtype=torch.long)
        class_count = torch.bincount(target_list)
        
        class_weights = 1. / (class_count.float() + 1e-6)
        sample_weights = class_weights[target_list]
        
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0 if cache_in_ram else 4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0 if cache_in_ram else 4, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader, train_ds.classes

    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        raise