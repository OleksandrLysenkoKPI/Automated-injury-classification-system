import os
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

data_root = Path(os.getenv("ROOT_KNEE_OSTEOARTHRITIS_DATASET_PATH"))
dataset_name = os.getenv("CURRENT_KNEE_OSTEOARTHRITIS_DATASET")

dataset_path = data_root / dataset_name

train_path = dataset_path / "train"
test_path = dataset_path / "test"

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number OF GPUs: {torch.cuda.device_count()}")