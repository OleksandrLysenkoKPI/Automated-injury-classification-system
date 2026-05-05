import sys
import torch
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QSlider, QProgressBar, QMessageBox, QStyle
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap


from src.ui.inference_engine import KneeInferenceEngine
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Main_Window")

class AnalysisThread(QThread):
    """background thread for neural network"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, engine, input_tensor):
        super().__init__()
        self.engine = engine
        self.input_tensor = input_tensor

    def run(self):
        logger.info("Start background analysis thread")
        try:
            result = self.engine.run_inference_only(self.input_tensor)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            logger.info("Thread received the result, sending finished signal")
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Critical error in background thread: {e}")
            self.error.emit(str(e))


class MedicalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knee Pathology Diagnostic System")
        self.resize(1000, 700)
        
        self.engine = KneeInferenceEngine("knee_3d_binary_model.pth", "knee_3d_stage2_6classes.pth") 
        self.current_volume = None # Volume slices
        
        self.setup_ui()

    def setup_ui(self):
        self.tabs = QTabWidget()
        
        self.tab_diagnostics = QWidget()
        self.setup_diagnostics_tab()
        
        self.tab_admin = QWidget()
        self.setup_admin_tab()
        
        self.tabs.addTab(self.tab_diagnostics, "Діагностика (Main)")
        self.tabs.addTab(self.tab_admin, "Панель керування (Admin Tools)")
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def setup_diagnostics_tab(self):
        main_layout = QHBoxLayout() # Separate left (control) and right (MRI) parts
        
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        style = QApplication.style()
        
        # Left panel (Control and Results)
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        
        self.btn_load = QPushButton(" Завантажити МРТ (DICOM / NPY)")
        self.btn_load.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)) # type: ignore
        self.btn_load.setFixedHeight(45)
        self.btn_load.clicked.connect(self.load_file)
        
        self.lbl_filepath = QLabel("Файл не обрано")
        self.lbl_filepath.setWordWrap(True)
        
        self.btn_analyze = QPushButton(" Провести аналіз")
        self.btn_analyze.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)) # type: ignore
        self.btn_analyze.setFixedHeight(55)
        self.btn_analyze.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.start_analysis)
        
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
        self.slider.valueChanged.connect(self.update_slice)
        
        self.lbl_slice_info = QLabel("Зріз: 0 / 0")
        self.lbl_slice_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        right_panel.addStretch(1)
        right_panel.addWidget(self.lbl_image, alignment=Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.slider)
        right_panel.addWidget(self.lbl_slice_info)
        right_panel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addStretch(1)
        
        main_layout.addLayout(left_panel, 1) # Left panel takes 1 part
        main_layout.addLayout(right_panel, 2) # Right panel takes 2 parts
        self.tab_diagnostics.setLayout(main_layout)

    def setup_admin_tab(self):
        # Temporary dud for admin panel 
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Тут будуть кнопки: Конвертація DICOM, Тренування NPY/PNG, Логи..."))
        self.tab_admin.setLayout(layout)

    # WORK LOGIC
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть файл МРТ", "", "DICOM / Numpy (*.dcm *.npy);;All Files (*)"
        )
        style = QApplication.style()
        if file_path:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.file_path = file_path
                self.lbl_filepath.setText(f"Файл: {Path(file_path).name}")
                
                self.input_tensor, self.current_volume = self.engine.preprocess_file(file_path)
                
                # Slider settings
                max_slices = self.current_volume.shape[0] - 1
                self.slider.setMaximum(max_slices)
                self.slider.setValue(max_slices // 2)
                self.slider.setEnabled(True)
                self.update_slice()
                
                self.btn_analyze.setEnabled(True)
                self.btn_analyze.setText(" Провести аналіз")
                play_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay) # type: ignore
                self.btn_analyze.setIcon(play_icon)
                self.lbl_stage1_res.setText("Готовий до аналізу.")
                self.lbl_stage1_res.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
                
            except Exception as e:
                QMessageBox.critical(self, "Помилка завантаження", f"Не вдалося прочитати файл:\n{e}")
            finally:
                QApplication.restoreOverrideCursor()
            
    def start_analysis(self):
        """Starts interface"""
        try:
            if self.input_tensor is None: 
                QMessageBox.warning(self, "Увага", "Спочатку завантажте файл!")
                return
            
            self.btn_analyze.setEnabled(False)
            self.btn_analyze.setText("Аналіз триває...")
            
            if hasattr(self, 'analysis_thread') and self.analysis_thread and self.analysis_thread.isRunning():
                self.analysis_thread.wait()
            
            # Start analysis in background process
            self.analysis_thread = AnalysisThread(self.engine, self.input_tensor)
            
            self.analysis_thread.finished.connect(self.on_analysis_finished)
            self.analysis_thread.error.connect(self.on_analysis_error)
            
            self.analysis_thread.start()
        except Exception as e:
            logger.error(f"Critical error occurred in start_analysis: {e}")

    def on_analysis_finished(self, result):
        if self.analysis_thread:
            self.analysis_thread.wait(10)
        
        logger.info("GUI: Received signal to finish analysis")
        if result is None:
            logger.warning("Received empty analysis result")
            return
        
        style = QApplication.style()
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setText(" Повторити аналіз")
        repeat_icon = style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload) # type: ignore
        self.btn_analyze.setIcon(repeat_icon)
        
        # Show Stage 1 results
        prob = int(result["stage1_prob"] * 100)
        
        low_conf = ""
        if 0.45 < result["stage1_prob"] < 0.65:
            low_conf = "\nНизька впевненість!!!"
        
        if result["is_pathology"]:
            self.lbl_stage1_res.setText(f"ПАТОЛОГІЯ ВИЯВЛЕНА ({prob}% ймовірність)")
            self.lbl_stage1_res.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c; background-color: #fadbd8; border-radius: 5px;")
            
            # Show Stage 2 results
            if result["stage2_results"]:
                for cls_name, cls_prob in result["stage2_results"].items():
                    prob_int = int(cls_prob * 100)
                    self.stage2_widgets[cls_name]['label'].setText(f"{cls_name}: {prob_int}%")
                    self.stage2_widgets[cls_name]['bar'].setValue(prob_int)
        else:
            self.lbl_stage1_res.setText(f"НОРМА ({prob}% ймовірність){low_conf}")
            self.lbl_stage1_res.setStyleSheet("font-size: 18px; font-weight: bold; color: #27ae60; background-color: #d5f5e3; border-radius: 5px;")
            
            for cls_name in self.stage2_widgets:
                self.stage2_widgets[cls_name]['label'].setText(f"{cls_name}: 0%")
                self.stage2_widgets[cls_name]['bar'].setValue(0)
                
        logger.info("GUI: Analysis results showcase is finished")

    def on_analysis_error(self, error_msg):
        style = QApplication.style()
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setText(" Помилка. Спробувати ще")
        error_icon = style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop) # type: ignore
        self.btn_analyze.setIcon(error_icon)
        QMessageBox.critical(self, "Помилка", f"Сталася помилка під час аналізу:\n{error_msg}")

    def update_slice(self):
        """Updates images by slider position"""
        if self.current_volume is None: return
        
        idx = self.slider.value()
        self.lbl_slice_info.setText(f"Зріз: {idx + 1} / {self.current_volume.shape[0]}")
        
        # Get 2d slices array (0-255)
        slice_data = np.ascontiguousarray(self.current_volume[idx])
        height, width = slice_data.shape
        bytes_per_line = width
        
        # Convert NumPy into QImage, and then in QPixmap
        q_img = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img.copy()) 
        
        pixmap = pixmap.scaled(
            512, 512, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_image.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion") 
    
    window = MedicalApp()
    window.show()
    sys.exit(app.exec())