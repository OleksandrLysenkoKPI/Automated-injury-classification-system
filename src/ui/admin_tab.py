from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QProgressBar, QStyle, QApplication
)
from PyQt6.QtCore import Qt


class AdminTab(QWidget):
    def __init__(self, parent_controller):
        super().__init__()
        self.controller = parent_controller # Main window reference for logic call
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        self.temp_text = QLabel("Тут будуть кнопки: Конвертація DICOM, Тренування NPY/PNG, Логи...")
        self.temp_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.temp_text)
        self.setLayout(layout)