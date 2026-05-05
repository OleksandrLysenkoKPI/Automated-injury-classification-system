from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QProgressBar, QStyle, QApplication
)
from PyQt6.QtCore import Qt
from .utils.ui_utils import numpy_to_pixmap

class DiagnosticsTab(QWidget):
    def __init__(self, parent_controller):
        super().__init__()
        self.controller = parent_controller # Main window reference for logic call
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self) # Separate left (control) and right (MRI) parts
        
        layout.setContentsMargins(20, 20, 20, 20)
        
        style = QApplication.style()
        
        # Left panel (Control and Results)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        
        self.btn_load = QPushButton(" Завантажити МРТ (DICOM / NPY)")
        self.btn_load.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)) # type: ignore
        self.btn_load.setFixedHeight(45)
        self.btn_load.clicked.connect(self.controller.load_file)
        
        self.lbl_filepath = QLabel("Файл не обрано")
        self.lbl_filepath.setWordWrap(True)
        
        self.btn_analyze = QPushButton(" Провести аналіз")
        self.btn_analyze.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)) # type: ignore
        self.btn_analyze.setFixedHeight(55)
        self.btn_analyze.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.controller.start_analysis)
        
        # Stage 1 results
        self.lbl_stage1_res = QLabel("Очікування аналізу...")
        self.lbl_stage1_res.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_stage1_res.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        
        # Stage 2 results (creates progress bars for every condition)
        self.stage2_widgets = {}
        self.layout_stage2 = QVBoxLayout()
        classes = ['Гонартроз', 'Хондромаляція виростків', 'Хондромаляція надколінка',
                   'Меніски', 'Часткове пошкодження пхз', 'Медіапателярна складка']
        
        for cls_name in classes:
            lbl = QLabel(f"{cls_name}: 0%")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            self.stage2_widgets[cls_name] = {'label': lbl, 'bar': bar}
            self.layout_stage2.addWidget(lbl)
            self.layout_stage2.addWidget(bar)
            
        left_panel.addWidget(self.btn_load)
        left_panel.addWidget(self.lbl_filepath)
        left_panel.addWidget(self.btn_analyze)
        left_panel.addWidget(self.lbl_stage1_res)
        left_panel.addLayout(self.layout_stage2)
        left_panel.addStretch(1)

        # ---Right panel (MRI visualization) ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        self.lbl_image = QLabel("Зображення МРТ")
        self.lbl_image.setFixedSize(512, 512)
        self.lbl_image.setStyleSheet("background-color: black; color: white;")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(lambda: self.controller.update_slice())
        
        self.lbl_slice_info = QLabel("Зріз: 0 / 0")
        self.lbl_slice_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        right_panel.addStretch(1)
        right_panel.addWidget(self.lbl_image, alignment=Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.slider)
        right_panel.addWidget(self.lbl_slice_info)
        right_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addStretch(1)
        
        layout.addLayout(left_panel, 1) # Left panel takes 1 part
        layout.addLayout(right_panel, 2) # Right panel takes 2 parts

    def update_slice(self, volume, index):
        """Method for MRI image update."""
        slice_data = volume[index]
        height, width = slice_data.shape
        pixmap = numpy_to_pixmap(slice_data, width, height)
        
        pixmap = pixmap.scaled(
            512, 512, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.lbl_image.setPixmap(pixmap)
        
    def set_slider_range(self, max_value):
        """Sets image slider range"""
        self.slider.setMaximum(max_value)
        self.slider.setValue(max_value // 2)
        self.slider.setEnabled(True)

    def set_file_info(self, text):
        """Sets opened file name"""
        self.lbl_filepath.setText(text)

    def update_stage1_ui(self, text, style_sheet):
        """Updates UI for Stage 1"""
        self.lbl_stage1_res.setText(text)
        self.lbl_stage1_res.setStyleSheet(style_sheet)

    def update_progress_bars(self, results):
        """Updates Stage 2 progress bars"""
        for cls_name, cls_prob in results.items():
            prob_int = int(cls_prob * 100)
            self.stage2_widgets[cls_name]['label'].setText(f"{cls_name}: {prob_int}%")
            self.stage2_widgets[cls_name]['bar'].setValue(prob_int)

    def reset_progress_bars(self):
        """Resets Stage 2 progress bars"""
        for cls_name in self.stage2_widgets:
            self.stage2_widgets[cls_name]['label'].setText(f"{cls_name}: 0%")
            self.stage2_widgets[cls_name]['bar'].setValue(0)
    
    def reset_ui(self):
        """Resets UI to a beginning state"""
        style = QApplication.style()
        
        # Button reset
        self.btn_analyze.setText(" Провести аналіз")
        self.btn_analyze.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)) # type: ignore
        self.btn_analyze.setEnabled(True)
        
        # Stage 1 status reset
        self.lbl_stage1_res.setText("Готовий до аналізу.")
        self.lbl_stage1_res.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        
        # Reset Stage 2 progress bars
        self.reset_progress_bars()