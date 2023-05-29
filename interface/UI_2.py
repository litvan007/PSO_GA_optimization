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

class PSO_interface(QMainWindow): # главное окно
    def __init__(self, parent=None):
        super().__init__(parent)
        # Количество частиц в рое
        self.PARTICLE_COUNT = 50
        # Количество измерений пространства поиска
        self.SPACE_DIMENSION = 2
        # Максимальное количество итераций алгоритма
        self.MAX_ITERATION = 5

        # Параметры алгоритма
        self.W = 0.729 # инерционный вес
        self.C1 = 1.49445 # коэффициент личного лучшего значения
        self.C2 = 1.49445 # коэффициент глобального лучшего значения

        # Границы пространства поиска
        self.MIN_X = -5.12
        self.MAX_X = 5.12
        self.X = np.arange(-2, 2, 0.05)
        self.Y = np.arange(-2, 2, 0.05)
        self.X, self.Y = np.meshgrid(self.X, self.Y)

        def func(x): # TODO
            sum = 0
            for i in range(2):
                sum += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
            return 10 * 2 + sum

        self.func = func
        self.Z = self.func([self.X, self.Y])

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

        # # создаем кнопки для анимации и конечного положения графика на одной горизонтальной линии
        # self.btn_animate = QPushButton("Анимация", self)
        # self.btn_animate.clicked.connect(self.show_animation)
        self.btn_final = QPushButton("Минимум", self)
        self.btn_final.clicked.connect(self.relevant_point_plot)
        
        # создаем горизонтальный лэйаут для кнопок
        self.spinbox = QSpinBox(self)
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(self.MAX_ITERATION)
        self.spinbox.setValue(1)
        self.spinbox.valueChanged.connect(self.main_plot)

        self.label = QLabel(self)
        # self.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.label.setText(f"/ {self.MAX_ITERATION}")

        self.label_point = QLabel(self)
        self.label_point.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.label_fit = QLabel(self)
        self.label_fit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        

        btn_layout = QHBoxLayout()
        # btn_layout.addWidget(self.btn_animate)
        btn_layout.addWidget(self.btn_final)
        btn_layout.addWidget(self.spinbox)
        btn_layout.addWidget(self.label)
        btn_layout.addWidget(self.label_point)
        btn_layout.addWidget(self.label_fit)

        # добавляем лэйаут с кнопками в основной лэйаут
        btn_layout.setAlignment(Qt.AlignCenter)
        self.layout1.addLayout(btn_layout)

        # устанавливаем лэйаут в центральный виджет
        self.central_widget.setLayout(self.layout1)

    def run(self):
        PSO = Common(self.X, self.Y, self.Z, self.MIN_X, self.MAX_X, self.SPACE_DIMENSION, self.func, self.W,
                        self.C1, self.C2, self.MAX_ITERATION, self.PARTICLE_COUNT)
        self.position_history, self.best_swarm_x, self.best_swarm_y, self.best_swarm_fitness = PSO.run()
        self.label_point.setText(f"Лучшее решение: X = {self.best_swarm_x}, Y = {self.best_swarm_y}")
        self.label_fit.setText(f"Значение ф-ии: {self.best_swarm_fitness}")
        self.main_plot()

    def main_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        plt.title("PSO")
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, color='b', )
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')
        ax.set_xlim(self.MIN_X, self.MAX_X)
        ax.set_ylim(self.MIN_X, self.MAX_X)
        ax.set_zlim(0, 100)
        for particle_position in self.position_history[self.spinbox.value()-1]:
            ax.plot(particle_position[1], particle_position[1], self.func(particle_position), 'r.')
        self.canvas.draw()

    def relevant_point_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        plt.title("PSO")
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, color='b', )
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')
        ax.set_xlim(self.MIN_X, self.MAX_X)
        ax.set_ylim(self.MIN_X, self.MAX_X)
        ax.set_zlim(0, 100)
        ax.plot(self.best_swarm_x, self.best_swarm_y, self.best_swarm_fitness, 'g.', marker='o', markersize=7,)
        self.canvas.draw()
