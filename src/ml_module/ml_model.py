import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torchvision.models.video import r3d_18, R3D_18_Weights
from ..logger_module.logger import CustomLogger
from .data_loader import load_dataset
from sklearn.metrics import classification_report

logger = CustomLogger("ML_module_log")

# TODO: rewrite utilizing fine-tuned 3D ResNet
class KneeResNet(nn.Module):
    def __init__(self, num_classes: int):
        super(KneeResNet, self).__init__()
        
        weights = R3D_18_Weights.DEFAULT
        self.model = r3d_18(weights=weights)
        
        original_conv = self.model.stem[0] # type: ignore
        self.model.stem[0] = nn.Conv3d( # type: ignore
            in_channels=1, 
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        self.dropout = nn.Dropout(p=0.4)
    
    def forward(self, x: torch.Tensor):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        x = self.model.fc(x)
        
        return x


def train_model(
    model: nn.Module,
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
    logger.info(f"Start of model training (AMP enabled) on device: {device}")
    
    for epoch in range(epochs):
        if epoch == 10:
            logger.info("Unfreezing layers for fine-tuning.")
            for param in model.parameters():
                param.requires_grad = True
                
            optimizer.param_groups[0]['lr'] = 1e-5
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images: torch.Tensor
            labels: torch.Tensor
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
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
            best_model_state = model.state_dict().copy()
            logger.info(f"New best model found at epoch {epoch+1} with Val Acc: {val_acc:.2f}%")
    
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


def start_model_pipeline(
    epochs: int = 30, 
    batch_size: int = 4, 
    target_shape: tuple[int, int, int] = (32, 256, 256), 
    save_file_name: str = "knee_3d_pathology_model",
    use_augmented: bool = True
):
    """Starts model training and evaluation pipeline. Saves model at the end.

    Args:
        epochs (int, optional): Defaults to 5.
        batch_size (int, optional): Defaults to 4.
        target_shape (tuple[int, int, int], optional): Defaults to (32, 256, 256).
        save_file_name (str, optional): Defaults to "knee_3d_pathology_model".
    """
    train_loader, val_loader, test_loader, classes = load_dataset(target_shape=target_shape, batch_size=batch_size, load_augmented=use_augmented)

    model = KneeResNet(num_classes=len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freezing
    for param in model.model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if "fc" in name or "stem.0" in name:
            param.requires_grad = True
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    weights = torch.tensor([1.5, 2.5, 1.5, 1.5, 0.8, 1.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-6)
    
    try:
        model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=epochs)
        evaluate_model(model, test_loader, device, classes) # type: ignore
        
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"Model saved to {save_file_name}.pth")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    