from typing import cast
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

logger = CustomLogger("ML_NPY_model_log")

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
    def __init__(self, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            DetailBlock3D(1, 16, stride=1),
            DetailBlock3D(16, 32, stride=2),
            DetailBlock3D(32, 64, stride=2),
            DetailBlock3D(64, 256, stride=2),
        )
        
        self.attn_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.features(x) 
        
        # Attention across depth slices
        # [B, 256, D, H, W] -> [B, 256, D]
        x_pool = F.adaptive_avg_pool3d(x, (x.shape[2], 1, 1)).flatten(2) 
        x_pool = x_pool.transpose(1, 2) # [B, D, 256]
        
        weights = self.attn_layer(x_pool) # [B, D, 1]
        weights = F.softmax(weights, dim=1)
        
        combined_features = torch.sum(x_pool * weights, dim=1) # [B, 256]
        return self.classifier(combined_features)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 0.005, max_gap: float = 20.0):
        self.patience, self.min_delta, self.max_gap = patience, min_delta, max_gap
        self.counter, self.best_loss, self.early_stop = 0, None, False
        
    def __call__(self, val_loss, train_acc, val_acc):
        gap = train_acc - val_acc
        if gap > self.max_gap:
            logger.warning(f"!!! Overfitting: Acc gap {gap:.2f}%")
            self.early_stop = True
            return
        if self.best_loss is None: self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss, self.counter = val_loss, 0


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
    
    logger.info(f"Start training 3D Stage {stage_config['stage_num']} on {device}")
    
    for epoch in range(epochs):
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
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model: {val_acc:.2f}%")
        
        early_stopping(avg_val_loss, train_acc, val_acc)
        if early_stopping.early_stop:
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
    logger.info(f"\n{report}")
    
    accuracy: float = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    logger.info(f"Overall Test Accuracy: {accuracy:.2f}%")
    logger.info("Evaluation complete.")


def start_npy_model_pipeline(
    base_data_path="data/prepared_data",
    epochs: int = 40, 
    batch_size: int = 4,
    mode: str = 'npy', 
    save_file_name: str = "knee_3d_binary_model",
    cache_in_ram: bool = False
):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, classes = load_dataset(base_data_path, batch_size, mode, stage=1, cache_in_ram=cache_in_ram)
    
    # Stage 1: Binary (Healthy vs Pathology)
    model = KneeResidualAttentionNet(num_classes=2, dropout_p=0.4).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # [Healthy, Pathology]
    weights = torch.tensor([1.0, 1.0]).to(device) 
    criterion = FocalLoss(weight=weights, gamma=2.0, label_smoothing=0.1)
    
    config = {'stage_num': 1, 'patience': 15, 'max_gap': 25.0}
    
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, config)
        evaluate_model(model, test_loader, device, classes)
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"3D Binary model saved.")
    except Exception as e:
        logger.error(f"3D Pipeline failed: {e}")

def start_stage2_npy_pipeline(
    base_data_path="data/prepared_data",
    binary_model_path="knee_3d_binary_model.pth",
    epochs=60, 
    batch_size=4, 
    save_file_name="knee_3d_stage2_6classes",
    cache_in_ram=False
):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, classes = load_dataset(base_data_path, batch_size, mode="npy", stage=2, cache_in_ram=cache_in_ram)
    
    model = KneeResidualAttentionNet(num_classes=2, dropout_p=0.6).to(device)
    
    # 3. Binary weights load
    try:
        model.load_state_dict(torch.load(binary_model_path))
        logger.info(f"3D Base weights loaded.")
    except Exception as e:
        logger.error(f"Failed to load binary 3D model: {e}"); return

    # 4. Surgery on classifier
    classifier_seq = cast(nn.Sequential, model.classifier)
    in_features = cast(nn.Linear, classifier_seq[4]).in_features
    classifier_seq[4] = nn.Linear(in_features, len(classes))
    model.to("cuda")

    # Optimizer
    for param in model.features.parameters(): param.requires_grad = False
    for param in model.classifier.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-4, weight_decay=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    weights = torch.tensor([1.5, 2.0, 1.0, 1.5, 1.0, 1.8]).to(device)
    criterion = FocalLoss(weight=weights, gamma=2.0, label_smoothing=0.1)
    
    config = {'stage_num': 2, 'patience': 20, 'max_gap': 25.0}
    
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, config)
        evaluate_model(model, test_loader, device, classes)
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"3D Stage 2 complete.")
    except Exception as e:
        logger.error(f"3D Stage 2 failed: {e}")