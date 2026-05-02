from typing import cast
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.backends.cudnn as cudnn
from ..logger_module.logger import CustomLogger
from .data_loader import load_dataset
from sklearn.metrics import classification_report

logger = CustomLogger("ML_PNG_model_log")

class KneeResNet(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.4, freeze_backbone: bool = True):
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
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p), # 0.4 for binary
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 0.005, max_gap: float = 20.0):
        self.patience = patience
        self.min_delta = min_delta
        self.max_gap = max_gap
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, train_acc, val_acc):
        gap = train_acc - val_acc
        if gap > self.max_gap:
            logger.warning(f"!!! Training stopped: Acc gap is too large ({gap:.2f}% > {self.max_gap}%)")
            self.early_stop = True
            return
        
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_optimizer_stage2(model, lr_backbone=1e-6, lr_head=1e-4):
    """Creates an optimizer with different learning rates for Stage 2"""
    return optim.AdamW([
        {'params': model.model.conv1.parameters(), 'lr': lr_backbone},
        {'params': model.model.layer4.parameters(), 'lr': lr_backbone},
        {'params': model.model.layer3.parameters(), 'lr': lr_backbone},
        {'params': model.model.fc.parameters(), 'lr': lr_head}
    ], weight_decay=0.2)

def unfreeze_layers(model: KneeResNet, stage: int):
    if stage == 1:
        logger.info("++++ Stage 2: Unfreezing Layer 4 ++++")
        for param in model.model.layer4.parameters():
            param.requires_grad = True
    elif stage == 2:
        logger.info("++++ Stage 3: Unfreezing Layer 3 ++++")
        for param in model.model.layer3.parameters():
            param.requires_grad = True

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler, 
    device, 
    epochs, 
    stage_config
):
    scaler = GradScaler("cuda")
    best_val_acc = 0.0
    best_model_state = None
    early_stopping = EarlyStopping(patience=stage_config['patience'], max_gap=stage_config['max_gap'])
    
    current_stage = 0
    
    logger.info(f"Start training Stage {stage_config['stage_num']} on {device}")
    
    for epoch in range(epochs):        
        
        if stage_config['stage_num'] == 1 and early_stopping.counter >= 5 and current_stage < 2:
            current_stage += 1
            unfreeze_layers(model, current_stage)
            early_stopping.counter = 0
            
            param_groups = [
                {'params': [p for n, p in model.model.named_parameters() if 'layer' in n and p.requires_grad], 'lr': 5e-5, 'weight_decay': 0.01},
                {'params': model.model.conv1.parameters(), 'lr': 5e-5, 'weight_decay': 0.01},
                {'params': model.model.fc.parameters(), 'lr': 2e-4, 'weight_decay': 0.1}
            ]
            optimizer = optim.AdamW(param_groups)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch)
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images: torch.Tensor
            labels: torch.Tensor
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
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
        
        train_acc, val_acc = 100 * train_correct / train_total, 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | Train: {train_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best model: Epoch {epoch+1} | Acc: {val_acc:.2f}%")
            best_model_state = model.state_dict().copy()
        
        early_stopping(avg_val_loss, train_acc, val_acc)
        if early_stopping.early_stop: break
                
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
    logger.info(f"\n{report}")
    
    accuracy: float = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    logger.info(f"Overall Test Accuracy: {accuracy:.2f}%")
    logger.info("Evaluation complete.")


def start_png_model_pipeline(
    base_data_path = "data/prepared_data",
    epochs: int = 40, 
    batch_size: int = 32, 
    mode: str = 'png', 
    save_file_name: str = "knee_2d_binary_model",
    cache_in_ram: bool = False
):
    """Starts model training and evaluation pipeline. Saves model at the end."""
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, classes = load_dataset(
        base_data_path, batch_size, mode, stage=1, cache_in_ram=cache_in_ram
    )
    model = KneeResNet(num_classes=2, dropout_p=0.4, freeze_backbone=True).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # [Healthy, Pathology]
    weights = torch.tensor([0.8, 2.5]).to(device)
    criterion = FocalLoss(weight=weights, gamma=2.0, label_smoothing=0.05)
    
    config = {'stage_num': 1, 'patience': 10, 'max_gap': 25.0}
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, config)
        evaluate_model(model, test_loader, device, classes)
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"Binary model saved.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        
def start_stage2_png_pipeline(
    base_data_path="data/prepared_data",
    binary_model_path="knee_2d_binary_model.pth",
    epochs=50, 
    batch_size=64, 
    save_file_name="knee_stage2_6classes",
    cache_in_ram=False
):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, classes = load_dataset(
        base_data_path, batch_size, mode="png", stage=2, cache_in_ram=cache_in_ram
    )
    
    logger.info("Initializing Stage 2 (Sequential Transfer Learning)")
    
    model = KneeResNet(num_classes=2, dropout_p=0.7, freeze_backbone=False)
    
    try:
        model.load_state_dict(torch.load(binary_model_path))
        logger.info(f"Base weights successfully loaded from {binary_model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    fc_seq = cast(nn.Sequential, model.model.fc)
    last_layer = cast(nn.Linear, fc_seq[-1])
    in_features = last_layer.in_features
    fc_seq[-1] = nn.Linear(in_features, len(classes))
    
    model.to(device)

    # Freeze fc layer
    for param in model.model.parameters(): 
        param.requires_grad = False
    for param in model.model.fc.parameters(): 
        param.requires_grad = True

    optimizer = optim.AdamW(model.model.fc.parameters(), lr=8e-5, weight_decay=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    weights = torch.tensor([1.5, 1.0, 1.0, 1.0, 1.0, 2.0]).to(device)
    criterion = FocalLoss(weight=weights, gamma=3.0, label_smoothing=0.3)
    
    config = {'stage_num': 2, 'patience': 15, 'max_gap': 25.0}
    
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, config)
        evaluate_model(model, test_loader, device, classes)
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"Stage 2 complete.")
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")