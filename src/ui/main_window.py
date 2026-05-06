import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout,
    QFileDialog, QMessageBox, QStyle
)
from PyQt6.QtCore import Qt

from .diagnostics_tab import DiagnosticsTab
from .threads.analysis_thread import AnalysisThread
from .inference_engine import KneeInferenceEngine
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

    def setup_ui(self):
        self.tabs = QTabWidget()
        
        self.diag_tab = DiagnosticsTab(self)
        
        self.tabs.addTab(self.diag_tab, "Діагностика (Main)")
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)

    # DIAGNOSTIC LOGIC
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