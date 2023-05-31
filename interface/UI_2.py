import sys
sys.path.append("../")

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from algorithms import GA
from algorithms import PSO

import numpy as np
import yaml

# Добавить флаги для модификации PSO TODO
class PSO_interface(QMainWindow): # главное окно
    def __init__(self, parent=None):
        super().__init__(parent)
        with open('configs/config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        # Количество частиц в рое
        self.PARTICLE_COUNT = int(config['PARTICLE_COUNT'])
        # Количество измерений пространства поиска
        self.SPACE_DIMENSION = 2
        # Максимальное количество итераций алгоритма
        self.MAX_ITERATION = int(config['MAX_ITERATION'])
        # Пороговое значение массы для отмерания
        self.MIN_MASS = float(config['MIN_MASS'])
        # Минимальное и максимальное значение температур для отжига
        self.MIN_T = float(config['MIN_T'])
        self.MAX_T = float(config['MAX_T'])
        # Задаем коэффициент охлаждения SA
        self.ALPHA = float(config['ALPHA'])

        # Параметры алгоритма
        self.W = float(config['W']) # инерционный вес
        self.C1 = 1.49445 # коэффициент личного лучшего значения
        self.C2 = 1.49445 # коэффициент глобального лучшего значения
        self.C3 = float(config['C3']) # коэффицент лучшего соседского значения

        # Границы пространства поиска
        self.MIN_X = float(config['MIN_X'])
        self.MAX_X = float(config['MAX_X'])

        self.GRAPH_MIN_X = float(config['GRAPH_MIN_X'])
        self.GRAPH_MAX_X = float(config['GRAPH_MAX_X'])

        self.X = np.arange(self.GRAPH_MIN_X, self.GRAPH_MAX_X, 0.05)
        self.Y = np.arange(self.GRAPH_MIN_X, self.GRAPH_MAX_X, 0.05)
        self.X, self.Y = np.meshgrid(self.X, self.Y)

        self.FUNCTION = config['FUNCTION']
        self.METHOD = int(config['METHOD']) 

        def func(x):
            return eval(self.FUNCTION, {'x': x, 
                                        'sin': np.sin,
                                        'cos': np.cos,
                                        'pi': np.pi})

        self.func = func
        self.Z = self.func([self.X, self.Y])

        self.MAX_GEN = int(config['MAX_GEN']) 
        self.p_c = float(config['p_c'])
        self.p_m = float(config['p_m'])

        self.USE_ANNEALING = config['USE_ANNEALING']
        self.USE_EXTINCTION = config['USE_EXTINCTION']

        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("PSO") # заголовок окна
        self.move(300, 300) # положение окна
        self.resize(1200, 680) # размер окна

        # создаем фигуру и холст для графика
        self.figure = plt.figure(figsize=(100, 100))
        self.canvas = FigureCanvas(self.figure)
        # создаем панель инструментов для управления графиком
        self.toolbar1 = NavigationToolbar(self.canvas, self)
        # добавляем холст и панель в центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout1 = QVBoxLayout()
        self.layout1.addWidget(self.toolbar1)
        self.layout1.addWidget(self.canvas)

        self.btn_final = QPushButton("Минимум", self)
        self.btn_final.clicked.connect(self.relevant_point_plot)
        
        # создаем горизонтальный лэйаут для кнопок
        self.spinbox = QSpinBox(self)
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(self.MAX_ITERATION)
        self.spinbox.setValue(1)
        self.spinbox.valueChanged.connect(self.main_plot)

        self.label = QLabel(self)
        self.label.setText(f"/ {self.MAX_ITERATION}")

        self.label_point = QLabel(self)
        self.label_point.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.label_fit = QLabel(self)
        self.label_fit.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.label_p_count = QLabel(self)
        self.label_p_count.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_final)
        btn_layout.addWidget(self.spinbox)
        btn_layout.addWidget(self.label)
        btn_layout.addWidget(self.label_p_count)
        btn_layout.addWidget(self.label_point)
        btn_layout.addWidget(self.label_fit)

        # добавляем лэйаут с кнопками в основной лэйаут
        btn_layout.setAlignment(Qt.AlignCenter)
        self.layout1.addLayout(btn_layout)

        # устанавливаем лэйаут в центральный виджет
        self.central_widget.setLayout(self.layout1)

    def run(self):
        if self.METHOD == 0: # PSO
            PSO_alg = PSO.Common(self.GRAPH_MIN_X, self.GRAPH_MAX_X, self.MIN_X, self.MAX_X, self.MIN_T, self.MAX_T, self.ALPHA, self.MIN_MASS, self.SPACE_DIMENSION, self.func, self.W,
                            self.C1, self.C2, self.C3, self.MAX_ITERATION, self.PARTICLE_COUNT, self.USE_EXTINCTION, self.USE_ANNEALING)
            self.particle_count_history, self.position_history, self.best_solution, self.best_value = PSO_alg.run()

        elif self.METHOD == 1: # GENETIC
            GA_alg = GA.Common(self.GRAPH_MIN_X, self.GRAPH_MAX_X, self.MIN_X, self.MAX_X, self.SPACE_DIMENSION, self.PARTICLE_COUNT, self.MAX_GEN, self.func, self.p_c, self.p_m)
            self.count_best_history, self.position_history, self.best_solution, self.best_value = GA_alg.run()

        self.label_point.setText(f"Лучшее решение: X = {self.best_solution[0]}, Y = {self.best_solution[1]}")
        self.label_fit.setText(f"Значение ф-ии: {self.best_value}")
        self.main_plot()

    def main_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        plt.title("PSO")
        ax.plot_surface(self.X, self.Y, self.Z, alpha=0.5)
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')
        ax.set_xlim(self.GRAPH_MIN_X, self.GRAPH_MAX_X)
        ax.set_ylim(self.GRAPH_MIN_X, self.GRAPH_MAX_X)
        ax.set_zlim(0, 100)

        epoch = self.spinbox.value() - 1
        if self.METHOD == 0: # PSO
            self.label_p_count.setText(f"Живых частиц: {self.particle_count_history[epoch]}")
            for particle_position in self.position_history[epoch]:
                ax.plot(particle_position[0], particle_position[1], self.func(particle_position), 'r.')
            self.canvas.draw()
        
        if self.METHOD == 1: # GENETIC
            for particle_position in self.position_history[epoch]:
                self.label_p_count.setText(f"Лучших частиц: {self.count_best_history[epoch]}")
                ax.plot(particle_position[0], particle_position[1], self.func(particle_position), 'r.')
            self.canvas.draw()

    def relevant_point_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        plt.title("PSO")
        ax.plot_surface(self.X, self.Y, self.Z, alpha=0.5)
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')
        ax.set_xlim(self.MIN_X, self.MAX_X)
        ax.set_ylim(self.MIN_X, self.MAX_X)
        ax.set_zlim(0, 100)
        ax.plot(self.best_solution[0], self.best_solution[1], self.best_value, 'g.', marker='o', markersize=7)
        self.canvas.draw()
