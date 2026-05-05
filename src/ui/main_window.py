import sys
import torch
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout,
    QFileDialog, QMessageBox, QStyle
)
from PyQt6.QtCore import Qt

from .diagnostics_tab import DiagnosticsTab
from .admin_tab import AdminTab
from .threads.analysis_thread import AnalysisThread
from .inference_engine import KneeInferenceEngine

from ..imaging.image_converter import DICOMProcessor
from ..imaging.utils import split_data
from ..imaging.image_augmentation import augment_and_save_npy_dataset, augment_and_save_png_dataset
from ..ml_module.ml_npy_model import start_npy_model_pipeline, start_stage2_npy_pipeline
from ..logger_module.logger import CustomLogger

logger = CustomLogger("Main_Window")

class MedicalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knee Pathology Diagnostic System")
        self.resize(1100, 750)
        
        self.engine = KneeInferenceEngine("models/knee_3d_binary_model.pth", "models/knee_3d_stage2_6classes.pth") 
        self.current_volume = None 
        self.input_tensor = None
        
        self.setup_ui()
        self.connect_admin_signals()

    def setup_ui(self):
        self.tabs = QTabWidget()
        
        self.diag_tab = DiagnosticsTab(self)
        self.admin_tab = AdminTab(self)
        
        self.tabs.addTab(self.diag_tab, "Діагностика (Main)")
        self.tabs.addTab(self.admin_tab, "Панель керування")
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)
        
    def connect_admin_signals(self):
        """Binds buttons from AdminTab to MedicalApp methods."""
        # Data preparation buttons
        self.admin_tab.btn_process_all.clicked.connect(self.handle_process_all_dicoms)
        self.admin_tab.btn_aug_npy.clicked.connect(self.handle_augmentation_npy)
        self.admin_tab.btn_split.clicked.connect(self.handle_data_split)
        
        # Model trainning buttons
        self.admin_tab.btn_npy_full.clicked.connect(lambda: self.run_training("npy_full"))
        self.admin_tab.btn_npy_s2.clicked.connect(lambda: self.run_training("npy_s2"))
        
        # Clear logs button
        self.admin_tab.btn_clear_logs.clicked.connect(lambda: self.admin_tab.log_output.clear())

    # ADMIN TOOLS TAB HANDLERS
    def handle_process_all_dicoms(self):
        """Launches a full conversion of DICOM datasets."""
        source_root = self.admin_tab.raw_data_edit.text()
    
        if not source_root or not Path(source_root).exists():
            QMessageBox.warning(self, "Помилка", "Оберіть коректний шлях до сирих DICOM даних!")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Оберіть папку для виводу результатів")
        if not output_dir:
            return
        
        logger.info(f"Start DICOM processing from: {source_root}")
        try:
            processor = DICOMProcessor()
            processor.process_all_conditions(
                conditions_root_dir=str(Path(source_root) / "conditions"),
                knee_root_dir=str(Path(source_root) / "healthy"),
                output_base_png=str(Path(output_dir) / "png_converted"),
                output_base_npy=str(Path(output_dir) / "npy_converted")
            )
            QMessageBox.information(self, "Успіх", "Конвертація завершена!")
        except Exception as e:
            logger.error(f"Conversion Error: {e}")

    def handle_augmentation_npy(self):
        """Launches NPY data augmentation."""
        
        path = self.admin_tab.dataset_path_edit.text()
    
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Помилка шляху", "Будь ласка, оберіть коректну папку датасету в Admin Tab!")
            return
        
        logger.info(f"Launching NPY dataset augmentation for {path}")
        logger.info(f"Запуск аугментації NPY для: {path}")
        try:
            augment_and_save_npy_dataset(path) 
            QMessageBox.information(self, "Успіх", "Аугментація NPY завершена!")
        except Exception as e:
            logger.error(f"Augmentation Error: {e}")

    def handle_data_split(self):
        """Starts data splitting into Train/Val/Test."""
        base_path = self.admin_tab.dataset_path_edit.text()
    
        if not base_path or not Path(base_path).exists():
            QMessageBox.warning(self, "Помилка", "Вкажіть шлях до папки з конвертованими даними!")
            return
        
        output_base = QFileDialog.getExistingDirectory(self, "Оберіть папку для Final Split")
        if not output_base:
            return

        logger.info(f"Data split from {base_path} into {output_base}")
        try:
            split_data(
                npy_root=str(Path(base_path) / "npy_converted"),
                png_root=str(Path(base_path) / "png_converted"),
                output_base=output_base
            )
            QMessageBox.information(self, "Успіх", "Дані успішно розділені!")
        except Exception as e:
            logger.error(f"Split Error: {e}")

    def run_training(self, task_type):
        """Launches model training pipeline."""
        base_path = self.admin_tab.dataset_path_edit.text()
    
        if not base_path or not Path(base_path).exists():
            QMessageBox.warning(self, "Помилка", "Вкажіть шлях до датасету перед навчанням!")
            return

        self.admin_tab.train_progress.setValue(10)
        
        try:
            if task_type == "npy_full":
                logger.info(f"Навчання на датасеті: {base_path}")
                # Передаємо base_data_path у пайплайн[cite: 15]
                start_npy_model_pipeline(base_data_path=base_path, epochs=30)
            elif task_type == "npy_s2":
                start_stage2_npy_pipeline(base_data_path=base_path, epochs=30)
                
            self.admin_tab.train_progress.setValue(100)
            QMessageBox.information(self, "Успіх", f"Навчання {task_type} завершено!")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.admin_tab.train_progress.setValue(0)
    
    # DIAGNOSTIC TAB LOGIC
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть файл МРТ", "", "DICOM / Numpy (*.dcm *.npy);;All Files (*)"
        )
        
        if not file_path:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.input_tensor, self.current_volume = self.engine.preprocess_file(file_path)
            
            self.diag_tab.reset_ui()
            
            # Update UI through tab methods
            self.diag_tab.set_file_info(f"Файл: {Path(file_path).name}")
            self.diag_tab.set_slider_range(self.current_volume.shape[0] - 1)
            self.diag_tab.btn_analyze.setEnabled(True)
            self.diag_tab.update_stage1_ui("Готовий до аналізу.", "color: white; font-weight: bold;")
            
            self.update_slice()
            
            logger.info(f"New file loaded and UI reset: {file_path}")
        except Exception as e:
            logger.error(f"Load error: {e}")
            QMessageBox.critical(self, "Помилка", f"Не вдалося прочитати файл:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def update_slice(self):
        """Orchestrator: Takes data from the model and tells the tab what to draw."""
        if self.current_volume is not None:
            idx = self.diag_tab.slider.value()
            total_slices = self.current_volume.shape[0]
            self.diag_tab.lbl_slice_info.setText(f"Зріз: {idx + 1} / {total_slices}")
            self.diag_tab.update_slice(self.current_volume, idx)

    def start_analysis(self):
        if self.input_tensor is None: return
        
        self.diag_tab.btn_analyze.setEnabled(False)
        self.diag_tab.btn_analyze.setText("Аналіз триває...")
        
        # Start background thread
        self.analysis_thread = AnalysisThread(self.engine, self.input_tensor)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()

    def on_analysis_finished(self, result):
        style = QApplication.style()

        self.diag_tab.btn_analyze.setEnabled(True)
        self.diag_tab.btn_analyze.setText(" Повторити аналіз")
        self.diag_tab.btn_analyze.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload)) # type: ignore
        
        prob = int(result["stage1_prob"] * 100)
        
        if result["is_pathology"]:
            style = "font-size: 18px; font-weight: bold; color: #e74c3c; background-color: #fadbd8;"
            self.diag_tab.update_stage1_ui(f"ПАТОЛОГІЯ ВИЯВЛЕНА ({prob}%)", style)
            if result["stage2_results"]:
                self.diag_tab.update_progress_bars(result["stage2_results"])
        else:
            style = "font-size: 18px; font-weight: bold; color: #27ae60; background-color: #d5f5e3;"
            self.diag_tab.update_stage1_ui(f"НОРМА ({prob}%)", style)
            self.diag_tab.reset_progress_bars()

    def on_analysis_error(self, error_msg):
        style = QApplication.style()
        self.diag_tab.btn_analyze.setEnabled(True)
        error_icon = style.standardIcon(QStyle.StandardPixmap.SP_BrowserStop) # type: ignore
        self.diag_tab.btn_analyze.setIcon(error_icon)
        QMessageBox.critical(self, "Помилка аналізу", error_msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MedicalApp()
    window.show()
    sys.exit(app.exec())