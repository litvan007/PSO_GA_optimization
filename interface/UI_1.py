import sys
sys.path.append("../")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt 
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.animation import Animation

from algorithms.PSO import Common

from PIL import Image
import numpy as np

import yaml

class Initial_interface(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize all variables
        self.function = ""
        self.num_iterations = 0
        self.min_x_input = -10.0
        self.max_x_input = 10.0

        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Optimization")
        self.move(300, 300)
        self.resize(1200, 680)

        # Set up the layout of the window with all the widgets
        layout = QFormLayout()

        # Create the input widgets
        function_label = QLabel("Function to optimize:")
        self.function_input = QLineEdit()
        self.function_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        iterations_label = QLabel("Number of iterations:")
        self.iterations_input = QSpinBox()
        self.iterations_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        x_boundaries_label = QLabel("X Boundaries:")
        self.x_boundaries_input = QLineEdit()
        self.x_boundaries_input.setText(f"({self.min_x_input}, {self.max_x_input})")
        self.x_boundaries_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Add the input widgets to the layout
        layout.addRow(function_label, self.function_input)
        layout.addRow(iterations_label, self.iterations_input)
        layout.addRow(x_boundaries_label, self.x_boundaries_input)

        # Create the Apply button
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.writeConfigFile)  # Connect the button click to the writeConfigFile function

        # Add the Apply button to the layout
        layout.addWidget(apply_button)

        # Set the layout for the main window
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def writeConfigFile(self):
        # Get the inputs from the widgets
        function = self.function_input.text()
        iterations = self.iterations_input.value()
        x_boundaries = self.x_boundaries_input.text()

        # Write the inputs into a YAML file
        config = {
            'function': function,
            'iterations': iterations,
            'x_boundaries': x_boundaries
        }

        with open('configs/config.yaml', 'w') as file:
            yaml.dump(config, file)



# колличество частиц
# границы
# колличество итераций
# инерционный вес


# c1, личного лучшего значения
# c2, глобального лучшего значения


# следование за успешным соседом # появляется c3 соседского лучшего значения
# инерции
# имитация отджига
# вымирание частиц              # с4 масса вымирания
# генетический алгоритм

#выбор метода