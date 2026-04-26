from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.backends.cudnn as cudnn
from ..logger_module.logger import CustomLogger
from .data_loader import load_dataset
from sklearn.metrics import classification_report

logger = CustomLogger("ML_module_log")

class DetailBlock3D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_c)
            )
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)) + self.shortcut(x))
        
        
class KneeResidualAttentionNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Features extractor based on 3D residual blocks
        self.features = nn.Sequential(
            DetailBlock3D(1, 16, stride=1),
            DetailBlock3D(16, 32, stride=2),
            DetailBlock3D(32, 64, stride=2),
            DetailBlock3D(64, 256, stride=2),
        )
        
        # Spatial-Depth Attention
        self.attn_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.features(x) # [B, 128, 4, 16, 16]
        
        x = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1)).flatten(2) # [B, 128, 4]
        x = x.transpose(1, 2) # [B, 4, 128]
        
        weights = self.attn_layer(x) # [B, 4, 1]
        weights = F.softmax(weights, dim=1)
        
        combined_features = torch.sum(x * weights, dim=1) # [B, 128]
        
        return self.classifier(combined_features)

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


def train_model(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    epochs: int = 10
):
    scaler = GradScaler("cuda")
    best_val_acc = 0.0
    best_model_state = None
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=25, min_delta=0, max_gap=35.0,verbose=True)
    
    logger.info(f"Start training Custom 3D KneeNet on {device}")
    
    for epoch in range(epochs):
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
            logger.info("Early stopping triggered. Finishing training.")
            break
        
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

def evaluate_model(
    model: KneeResidualAttentionNet,
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


def start_model_pipeline(
    epochs: int = 30, 
    batch_size: int = 8, 
    target_shape: tuple[int, int, int] = (32, 128, 128), 
    save_file_name: str = "knee_3d_pathology_model",
    use_augmented: bool = True,
    cache_in_ram: bool = False
):
    """Starts model training and evaluation pipeline. Saves model at the end.

    Args:
        epochs (int, optional): Defaults to 5.
        batch_size (int, optional): Defaults to 8.
        target_shape (tuple[int, int, int], optional): Defaults to (32, 256, 256).
        save_file_name (str, optional): Defaults to "knee_3d_pathology_model".
    """
    cudnn.benchmark = True
    train_loader, val_loader, test_loader, classes = load_dataset(target_shape=target_shape, batch_size=batch_size, load_augmented=use_augmented, cache_in_ram=cache_in_ram)

    model = KneeResidualAttentionNet(num_classes=len(classes))
    device = torch.device("cuda")
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    weights = torch.tensor([1.8, 2.5, 1.5, 1.8, 1.5, 1.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-6)
    
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs)
        evaluate_model(model, test_loader, device, classes) # type: ignore
        
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"Model saved to {save_file_name}.pth")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    