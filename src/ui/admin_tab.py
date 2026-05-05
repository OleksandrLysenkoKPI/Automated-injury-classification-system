from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QProgressBar, QTextEdit, QGroupBox, QFileDialog
)
import logging

from .utils.ui_utils import QtLogHandler


class AdminTab(QWidget):
    def __init__(self, parent_controller):
        super().__init__()
        self.controller = parent_controller # Main window reference
        self.setup_ui()
        self.setup_logging()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Global Paths section
        paths_group = QGroupBox("Налаштування шляхів")
        paths_layout = QVBoxLayout()
        
        # Create pickers
        self.raw_data_edit, raw_layout = self.create_path_picker("Шлях до сирих DICOM даних")
        self.dataset_path_edit, ds_layout = self.create_path_picker("Шлях до підготовленого датасету (NPY/PNG)")
        
        paths_layout.addLayout(raw_layout)
        paths_layout.addLayout(ds_layout)
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Data Prep & Conversion section
        data_group = QGroupBox("Підготовка даних та конвертація")
        data_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        self.btn_convert = QPushButton(" Конвертувати DICOM (NPY/PNG)")
        self.btn_process_all = QPushButton(" Конвертувати DICOM з наборів даних")
        row1.addWidget(self.btn_convert)
        row1.addWidget(self.btn_process_all)
        
        row2 = QHBoxLayout()
        self.btn_aug_npy = QPushButton(" Аугментація NPY")
        self.btn_aug_png = QPushButton(" Аугментація PNG")
        self.btn_split = QPushButton(" Split train data")
        row2.addWidget(self.btn_aug_npy)
        row2.addWidget(self.btn_aug_png)
        row2.addWidget(self.btn_split)
        
        data_layout.addLayout(row1)
        data_layout.addLayout(row2)
        data_group.setLayout(data_layout)

        # Model Training section
        train_group = QGroupBox("Model Training Pipeline")
        train_layout = QVBoxLayout()
        
        npy_row = QHBoxLayout()
        self.btn_npy_full = QPushButton(" Start NPY Pipeline (Full)")
        self.btn_npy_s1 = QPushButton(" NPY Stage 1")
        self.btn_npy_s2 = QPushButton(" NPY Stage 2")
        npy_row.addWidget(self.btn_npy_full)
        npy_row.addWidget(self.btn_npy_s1)
        npy_row.addWidget(self.btn_npy_s2)
        
        png_row = QHBoxLayout()
        self.btn_png_full = QPushButton(" Start PNG Pipeline (Full)")
        self.btn_png_s1 = QPushButton(" PNG Stage 1")
        self.btn_png_s2 = QPushButton(" PNG Stage 2")
        png_row.addWidget(self.btn_png_full)
        png_row.addWidget(self.btn_png_s1)
        png_row.addWidget(self.btn_png_s2)
        
        self.train_progress = QProgressBar()
        self.btn_retrain = QPushButton(" Перенавчити обрані моделі")
        self.btn_retrain.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold;")
        
        train_layout.addLayout(npy_row)
        train_layout.addLayout(png_row)
        train_layout.addWidget(self.train_progress)
        train_layout.addWidget(self.btn_retrain)
        train_group.setLayout(train_layout)

        # Logs View section
        log_group = QGroupBox("Системні логи (Real-time)")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            background-color: #1e1e1e; color: #d4d4d4; 
            font-family: 'Consolas', monospace; font-size: 11px;
        """)
        
        self.btn_clear_logs = QPushButton(" Очистити консоль")
        log_layout.addWidget(self.log_output)
        log_layout.addWidget(self.btn_clear_logs)
        log_group.setLayout(log_layout)

        layout.addWidget(data_group)
        layout.addWidget(train_group)
        layout.addWidget(log_group, 1)

    def setup_logging(self):
        """Connects log handler to the system."""
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.log_handler.new_log.connect(self.append_log)
        
        # Add handler to core logger to see all logs
        logging.getLogger().addHandler(self.log_handler)

    def append_log(self, message):
        """Adds text to QTextEdit."""
        self.log_output.append(message)
        self.log_output.ensureCursorVisible()
        
    def cleanuo_logging(self):
        """Removes the handler to prevent an error when closing the application."""
        if hasattr(self, 'log_handler'):
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler.close()
            
    def create_path_picker(self, label_text):
        """Creates a horizontal layout with a field and a folder selection button."""
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(label_text)
        
        btn_browse = QPushButton("Огляд...")
        btn_browse.clicked.connect(lambda: self.browse_folder(line_edit))
        
        layout.addWidget(line_edit)
        layout.addWidget(btn_browse)
        return line_edit, layout

    def browse_folder(self, target_line_edit):
        path = QFileDialog.getExistingDirectory(self, "Оберіть папку")
        if path:
            target_line_edit.setText(path)