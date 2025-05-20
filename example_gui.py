"""
This is a test script to demonstrate the use of the GUI class.

NOTE: Use the virtual environment 'venvscratchpad' to run this script.
"""

# Import packages
import numpy as np
from PyQt6.QtCore import QSize
from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QGridLayout, QLabel, QRadioButton, QComboBox
from PyQt6.QtGui import QPixmap
from PyQt6 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import os, sys, errno
from pathlib import Path

# ------------ CLASSES ------------


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        #: Set the window title
        self.setWindowTitle("TWAIN | Multi-Objective Control Toolbox (MOCT)")
        # FIXME: This does not appear to work...
        self.setWindowIcon(QtGui.QIcon('assets/icons/icon_twain_app.png'))

        #: Create a vertical layout
        grid = QGridLayout()

        #: Add plot
        self.canvas = MplCanvas(self, width=1, height=1, dpi=300)
        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [np.random.randint(0, 10) for i in range(n_data)]
        self.update_plot()
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setFixedSize(QSize(200, 20))
        grid.addWidget(self.canvas, 0, 0, -1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        grid.addWidget(toolbar, -1, 0, -1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        #: Add dummy image
        # label = QLabel(self)
        # pixmap = QPixmap('assets/simple_flow_field.png')
        # # FROM: https://stackoverflow.com/questions/21802868/how-to-resize-raster-image-with-pyqt  # nopep8
        # pixmap = pixmap.scaled(200, 200, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        # label.setPixmap(pixmap)
        # label.setFixedSize(QSize(200, 200))
        # grid.addWidget(label, 0, 0, -1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        #: Add radio button widgets
        label = QLabel("Selected optimization problem:")
        label.setFixedHeight(50)
        radio_btn_1 = QRadioButton("Yaw steering")
        radio_btn_2 = QRadioButton("Downregulation")
        radio_btn_3 = QRadioButton("Co-design")
        radio_btn_1.setChecked(True)
        radio_btn_1.toggled.connect(self.yaw_is_toggled)
        radio_btn_2.toggled.connect(self.downregulation_is_toggled)
        radio_btn_3.toggled.connect(self.codesign_is_toggled)
        # FROM: https://stackoverflow.com/questions/9532940/how-to-arrange-the-items-in-qgridlayout-as-shown/9533086#9533086  # nopep8
        grid.addWidget(label, 0, 1, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(radio_btn_1, 1, 1, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(radio_btn_2, 1, 1, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(radio_btn_3, 1, 1, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)

        #: Add a model dropdown menu
        self.model_dropdown_menu = QComboBox()
        self.model_dropdown_menu.addItems(["NREL 5MW", "DTU 10.2 Ref", "Vesco 2.0"])
        grid.addWidget(QLabel("Wind-turbine model:"), 2, 0, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self.model_dropdown_menu, 2, 0, -1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        
        #: Create a push-able button
        button = QPushButton("Close")
        button.setFixedSize(QSize(70, 35))
        # FROM: https://discuss.python.org/t/about-pyqt6-how-to-modify-a-button/21214  # nopep8
        # FROM: https://stackoverflow.com/questions/36918016/pyqt-designer-how-to-make-a-buttons-edges-rounder  # nopep8
        button.setStyleSheet("background-color: #A5172F; color: white; border-style: outset; border-width: 1px; border-radius: 2px; border-color: #A5172F;")
        button.clicked.connect(self.close)

        grid.addWidget(button, 3, 1, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        
        #: Set the layout
        widget = QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)
        self.setFixedSize(QSize(600, 400))
        
        #: Show the window
        self.show()

        #: Set a timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def yaw_is_toggled(self, checked):
        if checked:
            print("Yaw is checked")
            self._opt_method = 'yaw_steering'
            #: Enable all turbines
            self.model_dropdown_menu.setItemData(0, True, QtCore.Qt.ItemDataRole.UserRole+1)
            self.model_dropdown_menu.setItemData(1, True, QtCore.Qt.ItemDataRole.UserRole+1)
            self.model_dropdown_menu.setItemData(2, True, QtCore.Qt.ItemDataRole.UserRole+1)

    def downregulation_is_toggled(self, checked):
        if checked:
            print("Downregulation is checked")
            self._opt_method = 'downregulation'
            #: Enable all turbines
            # FIXME: Why is enabling all turbines not working?
            self.model_dropdown_menu.setItemData(0, True, QtCore.Qt.ItemDataRole.UserRole+1)
            self.model_dropdown_menu.setItemData(1, True, QtCore.Qt.ItemDataRole.UserRole+1)
            self.model_dropdown_menu.setItemData(2, True, QtCore.Qt.ItemDataRole.UserRole+1)

    def codesign_is_toggled(self, checked):
        if checked:
            print("Codesign is checked")
            self._opt_method = 'codesign'
            #: Disable two turbines
            # FROM: https://stackoverflow.com/questions/19429609/set-selected-item-for-qcombobox  # nopep8
            self.model_dropdown_menu.setCurrentIndex(0)
            self.model_dropdown_menu.setItemData(0, True, QtCore.Qt.ItemDataRole.UserRole)
            # FROM: https://stackoverflow.com/questions/63997370/qt-how-to-grey-out-options-using-qcombobox-and-make-unselectable  # nopep8
            self.model_dropdown_menu.setItemData(1, False, QtCore.Qt.ItemDataRole.UserRole-1)
            self.model_dropdown_menu.setItemData(2, False, QtCore.Qt.ItemDataRole.UserRole-1)

    def update_plot(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [np.random.randint(0, 10)]
        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(self.xdata, self.ydata, 'r', label="Plot")
        self.canvas.axes.legend(loc='upper right')
        self.canvas.axes.set_title("Dynamic Plot")
        self.canvas.setFixedSize(QSize(200, 200))
        # Trigger the canvas to update and redraw.
        self.canvas.draw()


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=300):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


# ------------ METHODS ------------

# ------------ SCRIPT ------------

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec()