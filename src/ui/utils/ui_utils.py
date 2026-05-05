from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
import logging

class QtLogHandler(logging.Handler, QObject):
    """Handler that redirects logs to a PyQt signal"""
    new_log = pyqtSignal(str)
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        
    def emit(self, record):
        msg = self.format(record)
        self.new_log.emit(msg)

def numpy_to_pixmap(slice_data: np.ndarray, width: int, height: int) -> QPixmap:
    """Converts a NumPy array to a QPixmap for display purposes."""
    slice_data = np.ascontiguousarray(slice_data)
    bytes_per_line = width
    q_img = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(q_img.copy())