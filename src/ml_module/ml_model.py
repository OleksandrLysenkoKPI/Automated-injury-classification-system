import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from ..logger_module.logger import CustomLogger
from .data_loader import load_dataset
from sklearn.metrics import classification_report

logger = CustomLogger("ML_module_log")

class KneeNet(nn.Module):
    def __init__(self, num_classes: int):
        super(KneeNet, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        
        self.fc1 = nn.Linear(64, 128) # After GAP will remain only 64 channels 
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _conv_layer_set(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
    
    def forward(self, x: torch.Tensor):
        out: torch.Tensor = self.conv_layer1(x)
        out = self.conv_layer2(out)
        
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 10
):
    logger.info(f"Start of model training on device: {device}")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images: torch.Tensor
            labels: torch.Tensor
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) # [Batch, 6]
            loss: torch.Tensor = criterion(outputs, labels) # labels in range [0, 5]
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {accuracy:.2f}%")

def evaluate_model(
    model: KneeNet,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str]
):
    logger.info("Starting model evaluation")
    model.eval()
    all_predictions, all_labels = [], []
   
    with torch.no_grad():
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


# TODO: Write function for running model
def start_model_pipeline(
    epochs: int = 5, 
    batch_size: int = 4, 
    target_shape: tuple[int, int, int] = (32, 256, 256), 
    save_file_name: str = "knee_3d_pathology_model"
):
    """Starts model training and evaluation pipeline. Saves model at the end.

    Args:
        epochs (int, optional): _description_. Defaults to 5.
        batch_size (int, optional): _description_. Defaults to 4.
        target_shape (tuple[int, int, int], optional): _description_. Defaults to (32, 256, 256).
        save_file_name (str, optional): _description_. Defaults to "knee_3d_pathology_model".
    """
    train_dataset, test_dataset, classes = load_dataset(target_shape=target_shape, batch_size=batch_size)

    model = KneeNet(num_classes=len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    try:
        train_model(model, train_dataset, criterion, optimizer, device, epochs=epochs)
        evaluate_model(model, test_dataset, device, classes)
        
        torch.save(model.state_dict(), f"{save_file_name}.pth")
        logger.info(f"Model saved to {save_file_name}.pth")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

# if train_loader and test_dataset:
#     test_loader = get_data_loader(test_dataset, shuffle=False)
    
#     if test_loader:
#         logging.info("Starting training process...")
#         train_model(model, train_loader, epochs=5)
        
#         logging.info("Starting evaluation...")
#         evaluate_model(model, test_loader)
        
#         file_name = "knee_model_refactor_test"
        
#         torch.save(model.state_dict(), f"{file_name}.pth")
#         logging.info(f"Model weights saved to {file_name}.pth")
# else:
#     logging.error("Failed to initialize loaders. Check your dataset paths.")
        