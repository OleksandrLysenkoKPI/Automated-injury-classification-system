import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from ..logger_module.logger import CustomLogger
from data_loader import load_dataset

logger = CustomLogger("ML_module_log")

class KneeNet(nn.Module):
    def __init__(self, num_classes: int):
        super(KneeNet, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32)
        self.conv_layer2 = self._conv_layer_set(32, 64)
        
        self.fc1 = nn.Linear(64 * 6 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)
    
    def _conv_layer_set(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
    
    def forward(self, x: torch.Tensor):
        out: torch.Tensor = self.conv_layer1(x)
        out = self.conv_layer2(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

train_dataset, test_dataset, num_classes = load_dataset(target_shape=(32, 256, 256))

model = KneeNet(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train_model(model: KneeNet, train_loader: DataLoader, epochs: int =10):
    logger.info("Start of model training")
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

# TODO: REWRITE with sklearn metrics
def evaluate_model(model: KneeNet, test_loader: DataLoader):
    logger.info("Start of model evaluation")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # Вимкнення розрахунку градієнтів
        for images, labels in test_loader:
            binary_labels = (labels >= 2).long().to(device)
            images = images.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += binary_labels.size(0)
            correct += (predicted == binary_labels).sum().item()
            
    logging.info(f"Final Test Accuracy: {100 * correct / total:.2f}%")

test_dataset = load_image_dataset(paths["test"], transform)

if train_loader and test_dataset:
    test_loader = get_data_loader(test_dataset, shuffle=False)
    
    if test_loader:
        logging.info("Starting training process...")
        train_model(model, train_loader, epochs=5)
        
        logging.info("Starting evaluation...")
        evaluate_model(model, test_loader)
        
        file_name = "knee_model_refactor_test"
        
        torch.save(model.state_dict(), f"{file_name}.pth")
        logging.info(f"Model weights saved to {file_name}.pth")
else:
    logging.error("Failed to initialize loaders. Check your dataset paths.")
        