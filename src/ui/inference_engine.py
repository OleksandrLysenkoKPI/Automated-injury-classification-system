import torch
import numpy as np
from pathlib import Path
from typing import cast
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from ..ml_module.ml_npy_model import KneeResidualAttentionNet
from ..imaging.image_converter import DICOMProcessor
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Interface_Engine")

class KneeInferenceEngine:
    def __init__(self, binary_model_path: str, stage2_model_path: str, device: str = "cuda"):
        try:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.classes_s2 = [
                'Гонартроз', 'Хондромаляція_виростків', 'Хондромаляція_надколінка',
                'Меніски', 'Часткове_пошкодження_пхз', 'Медіапателярна_складка'
            ]
            
            self.model_s1 = KneeResidualAttentionNet(num_classes=2).to(self.device)
            self.model_s1.load_state_dict(torch.load(binary_model_path, map_location=self.device))
            self.model_s1.eval()
            

            self.model_s2 = KneeResidualAttentionNet(num_classes=2)
            
            # Stage 2 head swap
            classifier_seq = cast(nn.Sequential, self.model_s2.classifier)
            in_features = cast(nn.Linear, classifier_seq[4]).in_features
            classifier_seq[4] = nn.Linear(in_features, len(self.classes_s2))
            self.model_s2.to(self.device)
            
            self.model_s2.load_state_dict(torch.load(stage2_model_path, map_location=self.device))
            self.model_s2.eval()
            
            self.dicom_processor = DICOMProcessor()
            logger.info("Inference Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
        
    def preprocess_file(self, file_path: str | Path):
        """Prepares file for neural network and returns weights for visualization"""
        try:
            path = Path(file_path)
            volume = None

            if path.suffix.lower() == '.dcm':
                # DICOM to NPY conversion
                if self.dicom_processor.load_file(str(path)):
                    volume = self.dicom_processor.get_processed_volume(target_shape=(32, 128, 128))
                else:
                    raise ValueError("Не вдалося завантажити DICOM файл.")
            
            elif path.suffix.lower() == '.npy':
                volume = np.load(path).astype(np.float32)
                volume = (volume - volume.mean()) / (volume.std() + 1e-6)
            
            if volume is None:
                raise ValueError("Невідомий формат файлу.")

            # model preparation [Batch, Channel, D, H, W]
            input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # UI preparation
            v_min, v_max = volume.min(), volume.max()
            display_volume = ((volume - v_min) / (v_max - v_min + 1e-6) * 255).astype(np.uint8)
            
            return input_tensor, display_volume
        except Exception as e:
            logger.error(f"Critical error occurred during file preprocessing: {e}")
            raise
        
    def run_inference_only(self, input_tensor):
        """Neural network work on provided preprocessed tensor"""
        logger.info("Start inference (run_inference_only)")
        try:
            with torch.no_grad():
                with autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu"):
                    # Stage 1
                    out_s1 = self.model_s1(input_tensor)
                    prob_s1 = F.softmax(out_s1, dim=1)
                    prediction_s1 = int(torch.argmax(prob_s1, dim=1).item())
                    
                    result = {
                        "is_pathology": prediction_s1 == 1,
                        "stage1_prob": float(prob_s1[0, prediction_s1].item()),
                        "stage2_results": None
                    }
                    
                    if result["is_pathology"]:
                        out_s2 = self.model_s2(input_tensor)
                        prob_s2 = F.softmax(out_s2, dim=1)[0]
                        result["stage2_results"] = {
                            name: float(prob_s2[i].item()) 
                            for i, name in enumerate(self.classes_s2)
                        }
                        
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    return result
        except Exception as e:
            logger.error(f"Error inside run_inference_only: {e}")
            raise e