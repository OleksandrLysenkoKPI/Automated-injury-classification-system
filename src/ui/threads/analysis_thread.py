from PyQt6.QtCore import QThread, pyqtSignal
import torch
from ...logger_module.logger import CustomLogger

logger = CustomLogger("Analysis_Thread")

class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, engine, input_tensor):
        super().__init__()
        self.engine = engine
        self.input_tensor = input_tensor

    def run(self):
        try:
            result = self.engine.run_inference_only(self.input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Thread error: {e}")
            self.error.emit(str(e))