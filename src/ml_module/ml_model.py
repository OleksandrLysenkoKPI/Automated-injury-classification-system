import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_dataset_paths():
    """Loads and checks paths from environment"""
    load_dotenv()

    root_env = os.getenv("ROOT_KNEE_OSTEOARTHRITIS_DATASET_PATH")
    name_env = os.getenv("CURRENT_KNEE_OSTEOARTHRITIS_DATASET")

    if not root_env or not name_env:
        raise EnvironmentError("Dataset environment variables are not set properly.")
        
    root_path = Path(root_env)
    dataset_path = root_path / name_env
    return {
        "train": dataset_path / "train",
        "test": dataset_path / "test"
    }

def get_transformations():
    """Returns standard transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Повні значення для 3-х каналів ImageNet
    ])

def load_image_dataset(path: Path, transform):
    """Safely loads image dataset and logs information about it"""
    if not path.exists():
        logging.error(f"Directory not found: {path}")
        return None
    
    try:
        dataset = ImageFolder(root=str(path), transform=transform)
        logging.info(f"Successfully loaded dataset from {path}")
        logging.info(f"Images: {len(dataset)} | Classes: {dataset.classes}")
        return dataset
    except Exception as e:
        logging.error(f"Failed to initialize ImageFolder: {e}")
        return None

def get_data_loader(dataset, batch_size=32, shuffle=True):
    """Creates DataLoader and tests first batch load"""
    if not dataset:
        logging.error("Cannot create DataLoader: Dataset is None")
        return None
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    try:
        images, labels = next(iter(loader))
        
        logging.info(f"Image batch shape: {images.shape} | Label batch shape: {labels.shape}")
        logging.info(f"Image dtype: {images.dtype} | Label dtype: {labels.dtype}")
        return loader
    except StopIteration:
        logging.error("Dataset is empty! Loader has no data to yield.")
        return None
    except Exception as e:
        logging.error(f"Error during DataLoader test: {e}")
        return None


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
        
        torch.save(model.state_dict(), "knee_model_v1.pth")
        logging.info("Model weights saved to knee_model_v1.pth")
else:
    logging.error("Failed to initialize loaders. Check your dataset paths.")
        