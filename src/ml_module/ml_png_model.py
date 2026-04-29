from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.backends.cudnn as cudnn
from ..logger_module.logger import CustomLogger
from .data_loader import load_dataset
from sklearn.metrics import classification_report

logger = CustomLogger("ML_PNG_model_log")

class KneeResNet(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            
            for param in self.model.conv1.parameters():
                param.requires_grad = True
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential( # type: ignore
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0, max_gap: float = 20.0, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait after the last update. Defaults to 7.
            min_delta (float): The smallest change required for it to be considered an improvement. Defaults to 0.0.
            verbose (bool): Toggle for logging message. Defaults to True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.max_gap = max_gap
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, train_acc, val_acc):
        gap = train_acc - val_acc
        if gap > self.max_gap:
            if self.verbose:
                logger.warning(f"!!! Training stopped: Acc gap is too large ({gap:.2f}% > {self.max_gap}%)")
            self.early_stop = True
            return
        
        
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def unfreeze_layers(model: KneeResNet, stage: int):
    if stage == 1:
        logger.info("Stage 2: Unfreezing Layer 4")
        for param in model.model.layer4.parameters():
            param.requires_grad = True
    elif stage == 2:
        logger.info("Stage 3: Unfreezing Layer 3")
        for param in model.model.layer3.parameters():
            param.requires_grad = True

def train_model(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: FocalLoss,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    epochs: int = 10
):
    scaler = GradScaler("cuda")
    best_val_acc = 0.0
    best_model_state = None
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=25, min_delta=0.01, max_gap=25.0,verbose=True)
    
    current_stage = 0
    
    logger.info(f"Start training Custom 2D KneeNet on {device}")
    
    for epoch in range(epochs):
        
        if early_stopping.counter >= 8 and current_stage < 2:
            current_stage += 1
            unfreeze_layers(model, current_stage)
            early_stopping.counter = 0
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(trainable_params, lr=1e-4, momentum=0.9, weight_decay=0.1, nesterov=True)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch, eta_min=1e-6)
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images: torch.Tensor
            labels: torch.Tensor
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with autocast
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss: torch.Tensor = criterion(outputs, labels)
            
            # Backward pass with gradient scaling to fix underflow problem
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            with autocast(device_type="cuda"):
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step()
        
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model (Acc improved): Epoch {epoch+1} | Acc: {val_acc:.2f}%")
        
        elif val_acc == best_val_acc and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model (Loss improved): Epoch {epoch+1} | Loss: {avg_val_loss:.4f}")
        
        early_stopping(avg_val_loss, train_acc, val_acc)
        
        if early_stopping.early_stop:
            if current_stage < 2:
                logger.info("Early stop prevented - switching to next stage instead.")
                early_stopping.early_stop = False
                early_stopping.counter = 8
            else:
                logger.info("Early stopping triggered. Finishing training.")
                break
        
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

def evaluate_model(
    model: KneeResNet,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str]
):
    logger.info("Starting model evaluation")
    model.eval()
    all_predictions, all_labels = [], []
   
    with torch.no_grad():
        with autocast(device_type="cuda"):
            for images, labels in test_loader:
                images: torch.Tensor
                labels: torch.Tensor
                
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=class_names,
        zero_division=0
    )
    logger.info(f"{report}")
    
    accuracy: float = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    logger.info(f"Overall Test Accuracy: {accuracy:.2f}%")
    logger.info("Evaluation complete.")


def start_png_model_pipeline(
    epochs: int = 30, 
    batch_size: int = 8, 
    mode: str = 'png', 
    save_file_name: str = "knee_3d_pathology_model",
    use_augmented: bool = True,
    cache_in_ram: bool = False
):
    """Starts model training and evaluation pipeline. Saves model at the end.

    Args:
        epochs (int, optional): Defaults to 5.
        batch_size (int, optional): Defaults to 8.
        mode (str): Can be 'png' or 'npy'. Defaults to 'png'.
        save_file_name (str, optional): Defaults to "knee_3d_pathology_model".
    """
    cudnn.benchmark = True
    train_loader, val_loader, test_loader, classes = load_dataset(
        batch_size=batch_size,
        mode=mode, 
        load_augmented=use_augmented, 
        cache_in_ram=cache_in_ram
    )

    model = KneeResNet(num_classes=len(classes), freeze_backbone=True)
    device = torch.device("cuda")
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    weights = torch.tensor([2.0, 2.5, 2.0, 2.5, 2.2, 1.8]).to(device)
    criterion = FocalLoss(weight=weights, gamma=2.0)
    optimizer = optim.SGD(trainable_params, lr=1e-3, momentum=0.9, weight_decay=0.2, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-6)
    
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs)
        evaluate_model(model, test_loader, device, classes) # type: ignore
        
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"Model saved to {save_file_name}.pth")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")