import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .ml_utils import get_dataset_paths, verify_dataset_processing
from ..logger_module.logger import CustomLogger

logger = CustomLogger("DataLoader_log")

class KneeDataset(Dataset):
    def __init__(self, root_dir: str | Path, is_train: bool = False, cache_in_ram: bool = False):
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.cache_in_ram = cache_in_ram
        
        self.samples = []
        self.cached_data = []
        self.labels = []
        
        self.file_ext = self._detect_extension()
        
        try:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                class_idx = self.class_to_idx[class_name]
                found_files = list(class_dir.rglob(f"*{self.file_ext}"))
                
                for f in found_files:
                    self.samples.append((f, class_idx))
            
            if not self.samples:
                raise FileNotFoundError(f"No {self.file_ext} files found in {self.root_dir}")

            logger.info(f"Dataset initialized: {len(self.samples)} samples found in {len(self.classes)} classes.")
        
            if self.cache_in_ram:
                self._load_to_ram()
                
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {e}")
            raise
        
    def _detect_extension(self):
        """Looks for the first file found in subfolders to understand the format"""
        for f in self.root_dir.rglob('*'):
            if f.is_file() and f.suffix in ['.npy', '.png']:
                return f.suffix
        
        logger.warning(f"Could not find any .npy or .png files in {self.root_dir}")
        return '.npy'
    
    def _load_to_ram(self):
        logger.info(f"Loading {self.file_ext} dataset into RAM...")
        for file_path, label in self.samples:
            tensor = self._load_file(file_path)
            self.cached_data.append(tensor)
            self.labels.append(label)
        logger.info(f"Successfully cached {len(self.cached_data)} samples.")
        
        return torch.clamp(tensor, 0.0, 1.0)
    
    def _load_file(self, file_path: Path):
        """Reading logic depending on the extension"""
        if self.file_ext == '.npy':
            img = np.load(file_path).astype(np.float16)
            tensor = torch.from_numpy(img)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
        else:
            with Image.open(file_path) as img:
                data = np.array(img.convert('L'), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(data).unsqueeze(0)
        
        return torch.clamp(tensor, 0.0, 1.0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.cache_in_ram:
            tensor, label = self.cached_data[index], self.labels[index]
        else:
            file_path, label = self.samples[index]
            tensor = self._load_file(file_path)

        if self.is_train:
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[-1]) # Horizontal
            if torch.rand(1) > 0.5:
                tensor = torch.flip(tensor, dims=[-2]) # Vertical (orientation change)
            
            noise = torch.randn_like(tensor) * 0.002
            tensor = tensor + noise
            
        return tensor.float(), label

def load_dataset(batch_size: int = 16, mode: str = "png", load_augmented: bool = False, cache_in_ram: bool = False):
    """Loads dataset and returns DataLoader objects with list of found classes"""
    try:
        paths = get_dataset_paths()
        
        if mode == "png":
            train_key = "train_augmented_png" if load_augmented else "train_png"
            val_key = "val_png"
            test_key = "test_png"
        else:
            train_key = "train_augmented_npy" if load_augmented else "train_npy"
            val_key = "val_npy"
            test_key = "test_npy"

        train_dataset = KneeDataset(paths[train_key], is_train=True, cache_in_ram=cache_in_ram)
        val_dataset = KneeDataset(paths[val_key], is_train=False, cache_in_ram=cache_in_ram)
        test_dataset = KneeDataset(paths[test_key], is_train=False, cache_in_ram=False)
        
        train_workers = 0 if cache_in_ram else 4
        
        target_list = torch.tensor([sample[1] for sample in train_dataset.samples])
        class_count = [torch.sum(target_list == i).item() for i in range(len(train_dataset.classes))]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        sample_weights = class_weights[target_list]
        
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=train_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=train_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        logger.info(f"Loaded dataset in {mode} mode. Classes: {train_dataset.classes}")
        return train_loader, val_loader, test_loader, train_dataset.classes

    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        raise