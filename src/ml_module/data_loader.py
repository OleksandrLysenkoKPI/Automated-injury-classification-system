from ..logger_module.logger import CustomLogger
from .ml_utils import get_dataset_paths, verify_dataset_processing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import numpy as np
import torch
from pathlib import Path

logger = CustomLogger("DataLoader_log")

class Knee3DPathologyDataset(Dataset):
    def __init__(self, root_dir: str | Path, target_shape: tuple[int, int, int], is_train: bool = False, cache_in_ram: bool = False):
        self.root_dir = Path(root_dir)
        self.target_shape = target_shape
        self.is_train = is_train
        self.cache_in_ram = cache_in_ram
        
        self.samples = []
        self.cached_data = []
        self.labels = []
        
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

            logger.info(f"Dataset initialized: {len(self.samples)} samples. RAM Cache: {self.cache_in_ram}")
        
            if self.cache_in_ram:
                logger.info("Loading dataset into RAM... (Be careful with memory!)")
                for file_path, label in self.samples:
                    img = np.load(file_path).astype(np.float16)
                    tensor = torch.from_numpy(img).unsqueeze(0).clone()
                    self.cached_data.append(torch.clamp(tensor, 0.0, 1.0))
                    self.labels.append(label)
                logger.info(f"Successfully cached {len(self.cached_data)} samples in RAM.")
        except Exception as e:
            logger.error(f"Failed to initialize dataset: {e}")
            raise
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if self.cache_in_ram:
            return self.cached_data[index], self.labels[index]
        else:
            file_path, label = self.samples[index]
            img = np.load(file_path).astype(np.float16)
            tensor = torch.from_numpy(img).unsqueeze(0).clone()
            return torch.clamp(tensor, 0.0, 1.0), label

def load_dataset(target_shape: tuple[int, int, int], batch_size: int = 4, load_augmented: bool = False, verify_processing: bool = False, img_idx: int = 1, cache_in_ram: bool = False):
    """Loads dataset and returns DataLoader objects with list of found classes"""
    try:
        paths = get_dataset_paths()
        
        train_path = paths["train_augmented"] if load_augmented else paths["train"]
        
        train_dataset = Knee3DPathologyDataset(train_path, target_shape, is_train=True, cache_in_ram=cache_in_ram)
        val_dataset = Knee3DPathologyDataset(paths["val"], target_shape, is_train=False, cache_in_ram=cache_in_ram)
        test_dataset = Knee3DPathologyDataset(paths["test"], target_shape, is_train=False, cache_in_ram=False)
        
        if verify_processing: verify_dataset_processing(train_dataset, sample_idx=img_idx)
        
        train_workers = 0 if cache_in_ram else 4
        
        target_list = torch.tensor([sample[1] for sample in train_dataset.samples])
        class_count = [torch.sum(target_list == i).item() for i in range(len(train_dataset.classes))]
        
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        sample_weights = class_weights[target_list]
        
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=train_workers, 
            pin_memory=True,
            shuffle=False
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=train_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        logger.info(f"Loaded {len(train_dataset.classes)} classes: {train_dataset.classes}")

        logger.info("Dataset was loaded successfully")
        return train_loader,val_loader, test_loader, train_dataset.classes
    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        raise