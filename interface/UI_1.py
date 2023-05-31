import sys
sys.path.append("../")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt 
from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie
from PyQt5.QtGui import QPixmap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.animation import Animation

from algorithms.PSO import Common

from PIL import Image
import numpy as np

import yaml

from interface.UI_2 import PSO_interface

class Initial_interface(QMainWindow):
    open_ui2 = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize all variables
        self.function = ""
        self.num_iterations = 0
        self.min_x_input = -10.0
        self.max_x_input = 10.0

        self.ui2_instances = []
        self.setupUI()


    def setupUI(self):
        self.setWindowTitle("Optimization")
        self.move(100, 100)
        self.resize(1000, 1000)

        layout = QVBoxLayout()

        # Common parameters

        # Function

        self.function_label = QLabel("Function to optimize:")
        self.function_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.function_input = QLineEdit()
        self.function_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.function_input.setText('5*sin(x[0]) + 5*cos(x[1])')

        # Number of particles

        self.particle_label = QLabel("Number of particles:")
        self.particle_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.particle_input = QLineEdit()
        self.particle_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.particle_input.setText('50')

        # Boundaries

        self.boundary_label = QLabel("Boundaries:")
        self.boundary_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.boundary1_label = QLabel("MIN_X:")
        self.boundary1_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.boundary1_input = QLineEdit()
        self.boundary1_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.boundary1_input.setText('-5.12')

        self.boundary2_label = QLabel("MAX_X:")
        self.boundary2_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.boundary2_input = QLineEdit()
        self.boundary2_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.boundary2_input.setText('5.12')
        

        # Common parameters layout
        self.common_parameters = QWidget()
        self.common_layout = QFormLayout(self.common_parameters)

        self.common_layout.addRow(self.function_label, self.function_input)
        self.common_layout.addRow(self.particle_label, self.particle_input)
        self.common_layout.addRow(self.boundary_label)
        self.common_layout.addRow(self.boundary1_label, self.boundary1_input)
        self.common_layout.addRow(self.boundary2_label, self.boundary2_input)


        # Dropdown menu
        self.dropdown_label = QLabel("Optimization method:")
        self.dropdown_menu = QComboBox()
        self.dropdown_menu.addItems(['PSO', 'GENETIC'])
        self.dropdown_menu.currentIndexChanged.connect(self.on_dropdown_change)
        self.dropdown_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.dropdown_menu.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # PSO Parameters

        # Common PSO parameters
        self.iter_label = QLabel("Number of iterations:")
        self.iter_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.iter_input = QLineEdit()
        self.iter_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.iter_input.setText('100')


        #neighbours parameters
        self.use_neighbours_checkbox = QCheckBox("Use Neighbours")
        self.use_neighbours_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.use_neighbours_checkbox.stateChanged.connect(self.on_neighbours_checkbox_change)

        self.c3_label = QLabel("C3:")
        self.c3_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.c3_label.hide()

        self.c3_input = QLineEdit()
        self.c3_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.c3_input.setText('0.5')
        self.c3_input.hide()

        #inertion parameters
        self.use_inertion_checkbox = QCheckBox("Use inertion")
        self.use_inertion_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.use_inertion_checkbox.stateChanged.connect(self.on_inertion_checkbox_change)

        self.W_label = QLabel("W:")
        self.W_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.W_label.hide()
        
        self.W_input = QLineEdit()
        self.W_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.W_input.setText('0.729')
        self.W_input.hide()

        #simulated_annealing parameters

        self.use_annealing_checkbox = QCheckBox("Use simulated annealing")
        self.use_annealing_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.use_annealing_checkbox.stateChanged.connect(self.on_annealing_checkbox_change)

        #MIN_T
        self.MIN_T_label = QLabel("MIN_T:")
        self.MIN_T_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.MIN_T_label.hide()
        
        self.MIN_T_input = QLineEdit()
        self.MIN_T_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.MIN_T_input.setText('0.1')
        self.MIN_T_input.hide()
        
        #MAX_T
        self.MAX_T_label = QLabel("MAX_T:")
        self.MAX_T_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.MAX_T_label.hide()
        
        self.MAX_T_input = QLineEdit()
        self.MAX_T_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.MAX_T_input.setText('100')
        self.MAX_T_input.hide()

        #alpha
        self.alpha_label = QLabel("alpha:")
        self.alpha_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.alpha_label.hide()
        
        self.alpha_input = QLineEdit()
        self.alpha_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.alpha_input.setText('0.97')
        self.alpha_input.hide()

        #particle extinction parameters

        self.use_extinction_checkbox = QCheckBox("Use Extinction")
        self.use_extinction_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.use_extinction_checkbox.stateChanged.connect(self.on_extinction_checkbox_change)

        self.MIN_MASS_label = QLabel("MIN_MASS:")
        self.MIN_MASS_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.MIN_MASS_label.hide()

        self.MIN_MASS_input = QLineEdit()
        self.MIN_MASS_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.MIN_MASS_input.setText('0.01')
        self.MIN_MASS_input.hide()

        #PSO Layout
        self.pso_parameters = QWidget()
        self.pso_layout = QFormLayout(self.pso_parameters)
        self.pso_layout.addRow(self.iter_label, self.iter_input)

        self.pso_layout.addRow(self.use_neighbours_checkbox)
        self.pso_layout.addRow(self.c3_label, self.c3_input)

        self.pso_layout.addRow(self.use_inertion_checkbox)
        self.pso_layout.addRow(self.W_label, self.W_input)

        self.pso_layout.addRow(self.use_annealing_checkbox)
        self.pso_layout.addRow(self.MIN_T_label, self.MIN_T_input)
        self.pso_layout.addRow(self.MAX_T_label, self.MAX_T_input)
        self.pso_layout.addRow(self.alpha_label, self.alpha_input)

        self.pso_layout.addRow(self.use_extinction_checkbox)
        self.pso_layout.addRow(self.MIN_MASS_label, self.MIN_MASS_input)

        self.pso_parameters.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # self.pso_parameters.hide()

        # GENETIC Parameters
        self.gen_label = QLabel("Number of generations:")
        self.gen_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.gen_input = QLineEdit()
        self.gen_input.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.gen_input.setText('100')

        self.genetic_parameters = QWidget()
        self.gen_layout = QFormLayout(self.genetic_parameters)
        self.gen_layout.addRow(self.gen_label, self.gen_input)
        self.genetic_parameters.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.genetic_parameters.hide()

        # Other buttons
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.writeConfigFile)
        self.apply_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.on_optimize_button_clicked)
        self.optimize_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(self.common_parameters)
        layout.addWidget(self.dropdown_label)
        layout.addWidget(self.dropdown_menu)
        layout.addWidget(self.pso_parameters)
        layout.addWidget(self.genetic_parameters)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.optimize_button)


        main_widget = QWidget()
        main_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        main_widget.setLayout(layout)


        #images
        pic1_label = QLabel()
        pic1 = QPixmap("images/matmodeling_pic3.jpg")
        pic1_label.setPixmap(pic1)

        # pic2_label = QLabel()
        # pic2 = QPixmap("images/matmodeling_pic2.jpg")
        # pic2_label.setPixmap(pic2)

        central_layout = QGridLayout()
        central_layout.addWidget(main_widget, 0,0)
        central_layout.addWidget(pic1_label, 0,1)
        # central_layout.addWidget(pic2_label, 1,1)


        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)


    def on_dropdown_change(self, index):
        if index == 0:  # PSO selected
            self.pso_parameters.show()
            self.genetic_parameters.hide()
        elif index == 1:  # GENETIC selected
            self.pso_parameters.hide()
            self.genetic_parameters.show()

    def on_neighbours_checkbox_change(self, state):
        if state == 2:  # Checkbox checked
            self.c3_label.show()
            self.c3_input.show()
        else:  # Checkbox unchecked
            self.c3_label.hide()
            self.c3_input.hide()

    def on_inertion_checkbox_change(self, state):
        if state == 2:  # Checkbox checked
            self.W_label.show()
            self.W_input.show()
        else:  # Checkbox unchecked
            self.W_label.hide()
            self.W_input.hide()

    def on_annealing_checkbox_change(self, state):
        if state == 2:  # Checkbox checked
            self.MIN_T_label.show()
            self.MIN_T_input.show()

            self.MAX_T_label.show()
            self.MAX_T_input.show()

            self.alpha_label.show()
            self.alpha_input.show()
        else:  # Checkbox unchecked
            self.MIN_T_label.hide()
            self.MIN_T_input.hide()

            self.MAX_T_label.hide()
            self.MAX_T_input.hide()

            self.alpha_label.hide()
            self.alpha_input.hide()
    
    def on_extinction_checkbox_change(self, state):
        if state == 2:  # Checkbox checked
            self.MIN_MASS_label.show()
            self.MIN_MASS_input.show()
        else:  # Checkbox unchecked
            self.MIN_MASS_label.hide()
            self.MIN_MASS_input.hide()

    def on_optimize_button_clicked(self):
        self.writeConfigFile()

        ui2 = PSO_interface()
        ui2.run()
        ui2.show()
        ui2.destroyed.connect(lambda: self.ui2_instances.remove(ui2))
        self.ui2_instances.append(ui2)

    def writeConfigFile(self):
        # Get the inputs from the widgets
        FUNCTION = self.function_input.text()
        PARTICLE_COUNT = self.particle_input.text()
        MAX_ITERATION = self.iter_input.text()
        MIN_X = self.boundary1_input.text()
        MAX_X = self.boundary2_input.text()

        USE_NEIGHBOURS = self.use_neighbours_checkbox.isChecked()
        C3 = self.c3_input.text()

        USE_INERTION = self.use_inertion_checkbox.isChecked()
        W = self.W_input.text()

        USE_ANNEALING = self.use_annealing_checkbox.isChecked()
        MIN_T = self.MIN_T_input.text()
        MAX_T = self.MAX_T_input.text()
        ALPHA = self.alpha_input.text()

        USE_INERTION = self.use_inertion_checkbox.isChecked()
        W = self.W_input.text()

        USE_EXTINCTION = self.use_extinction_checkbox.isChecked()
        MIN_MASS = self.MIN_MASS_input.text()

        # Write the inputs into a YAML file
        config = {
            'FUNCTION': FUNCTION,
            'PARTICLE_COUNT': PARTICLE_COUNT,
            'MAX_ITERATION': MAX_ITERATION,
            'MIN_X': MIN_X,
            'MAX_X': MAX_X,
            'USE_NEIGHBOURS': USE_NEIGHBOURS,
            'C3': C3,
            'USE_INERTION': USE_INERTION,
            'W': W,
            'USE_ANNEALING': USE_ANNEALING,
            'MIN_T': MIN_T,
            'MAX_T': MAX_T,
            'ALPHA': ALPHA,
            'USE_EXTINCTION': USE_EXTINCTION,
            'MIN_MASS': MIN_MASS

        }

        with open('configs/config.yaml', 'w') as file:
            yaml.dump(config, file)


# функция
# колличетсво итераций
# колличество частиц
# границы
# колличество итераций


# c1, личного лучшего значения
# c2, глобального лучшего значения


# следование за успешным соседом # появляется c3 соседского лучшего значения
# инерции                        # появляется вес W
# имитация отджига               # появляется MIN_T, MAX_T, alpha
# вымирание частиц               # появляется MIN_MASS
# генетический алгоритм          # коллиество поколений, вероятность кроссовера, вероятность мутации

#выбор метода

