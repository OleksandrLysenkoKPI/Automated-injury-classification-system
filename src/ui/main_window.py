import sys
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout, QLabel

class TabExample(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tab Plan")
        
        self.tabs = QTabWidget()
        
        self.tab1 = QWidget()
        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("This is the main tab"))
        self.tab1.setLayout(layout1)
        
        self.tab2 = QWidget()
        layout2 = QVBoxLayout()
        layout2.addWidget(QLabel("Settings and options for Admin tab will be here"))
        self.tab2.setLayout(layout2)
        
        self.tabs.addTab(self.tab1, "Main")
        self.tabs.addTab(self.tab2, "Admin tools")
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TabExample()
    ex.show()
    sys.exit(app.exec())