from PyQt6.QtGui import QImage, QPixmap
import numpy as np

def numpy_to_pixmap(slice_data: np.ndarray, width: int, height: int) -> QPixmap:
    """Converts a NumPy array to a QPixmap for display purposes."""
    slice_data = np.ascontiguousarray(slice_data)
    bytes_per_line = width
    q_img = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(q_img.copy())