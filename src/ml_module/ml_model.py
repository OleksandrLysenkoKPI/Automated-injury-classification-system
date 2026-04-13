import torch
import torch.nn as nn
import torch.optim as optim
import logging
from data_loader import *

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

# TODO: REWRITE
paths = get_dataset_paths()
transform = get_transformations()

train_dataset = load_image_dataset(paths["train"], transform)
train_loader = get_data_loader(train_dataset)

model = KneeNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, epochs=10):
    log_section("Training Model")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Групування 0, 1 --> Healthy (0), 2,3,4 --> (Ill)
            binary_labels = (labels >=2).long().to(device)
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += binary_labels.size(0)
            correct += (predicted == binary_labels).sum().item()
        
        accuracy = 100 * correct / total
        logging.info(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Acc: {accuracy:.2f}%")

def evaluate_model(model, test_loader):
    log_section("Evaluating Model")
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
        