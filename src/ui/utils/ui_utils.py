from PyQt6.QtGui import QImage, QPixmap
import numpy as np

def numpy_to_pixmap(slice_data: np.ndarray, width: int, height: int) -> QPixmap:
    """
    Converts raw numerical image data into a displayable PyQt6 graphic.
    
    Args:
        slice_data (np.ndarray): The 2D array representing an MRI slice.
        width (int): The target display width.
        height (int): The target display height.
        
    Returns:
        QPixmap: A graphical representation of the slice for the UI labels.
    """
    slice_data = np.ascontiguousarray(slice_data)
    bytes_per_line = width
    q_img = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(q_img.copy())