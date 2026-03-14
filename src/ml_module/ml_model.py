import torch
import torch.nn as nn
import torch.optim as optim
import logging
from ml_utils import log_section
from data_loader import *

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class KneeNet(nn.Module):
    def __init__(self):
        super(KneeNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 2 виходи: здорове/хворе
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


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

# Запуск
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
        