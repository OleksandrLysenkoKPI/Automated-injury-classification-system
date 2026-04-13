import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

class CustomLogger:
    def __init__(self, name: str, base_log_dir: str = "logs", create_folder: bool = True, timestamp_folder: bool = False):
        self.logger = logging.getLogger(name)
        
        if self.logger.hasHandlers():
            return
        
        log_path = Path(base_log_dir)
        
        if create_folder:
            folder_name = name
            if timestamp_folder:
                timestamp = datetime.now().strftime("%d.%m.%y")
                folder_name = f"{timestamp}_{name}"
            
            log_path = log_path / folder_name
        
        self.name_dir = log_path
        self.name_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.name_dir / f"{name}.log"
                
        self.setup_system_logger(log_file)
    
    def setup_system_logger(self, log_file: Path):        
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(funcName)s.%(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log INFO messages"""
        self.logger.info(message, stacklevel=2)
    
    def error(self, message: str):
        """Log ERROR messages"""
        self.logger.error(message, stacklevel=2, exc_info=True)
        
    def warning(self, message: str):
        """Log WARNING messages"""
        self.logger.warning(message, stacklevel=2, exc_info=True)


if __name__ == "__main__":
    custom_logger = CustomLogger(name="Logger_test")
    
    custom_logger.info("This is information log test")
    
    try:
        x = 1/0
    except Exception as e:
        custom_logger.error(f"This is error log test: {e}")
        
    def test_method():
        custom_logger.error("This is log test inside a method")
        
    test_method()