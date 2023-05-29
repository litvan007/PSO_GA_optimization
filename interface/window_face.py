from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt 
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from algorithms.PSO import Common

from PIL import Image
import numpy as np

class PSO_interface(QMainWindow): # главное окно
    def __init__(self, parent=None):
        super().__init__(parent)
        # Количество частиц в рое
        self.PARTICLE_COUNT = 100
        # Количество измерений пространства поиска
        self.SPACE_DIMENSION = 2
        # Максимальное количество итераций алгоритма
        self.MAX_ITERATION = 10

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
        self.setWindowTitle("Hello, world") # заголовок окна
        self.move(300, 300) # положение окна
        self.resize(600, 400) # размер окна
        self.lbl = QLabel('Hello, world!!!', self)
        self.lbl.move(30, 30)

        # создаем фигуру и холст для графика
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        # создаем панель инструментов для управления графиком
        self.toolbar = NavigationToolbar(self.canvas, self)
        # добавляем холст и панель в центральный виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # создаем кнопки для анимации и конечного положения графика на одной горизонтальной линии
        self.btn_animate = QPushButton("Анимация", self)
        # self.btn_animate.clicked.connect(self.animate)
        self.btn_final = QPushButton("Конечное положение", self)
        # self.btn_final.clicked.connect(self.final)
        
        # создаем горизонтальный лэйаут для кнопок
        self.spinbox = QSpinBox(self)
        self.spinbox.setMinimum(1)
        self.spinbox.setMaximum(10)
        self.spinbox.setValue(5)
        # self.spinbox.valueChanged.connect(self.plot)

        self.label = QLabel(self)
        self.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.label.setText("Num of itearation")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_animate)
        btn_layout.addWidget(self.btn_final)
        btn_layout.addWidget(self.spinbox)
        btn_layout.addWidget(self.label)

        # добавляем лэйаут с кнопками в основной лэйаут
        btn_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(btn_layout)

        self.central_widget.setLayout(layout)

    def run(self):
        PSO = Common(self.X, self.Y, self.Z, self.MIN_X, self.MAX_X, self.SPACE_DIMENSION, self.func, self.W,
                        self.C1, self.C2, self.MAX_ITERATION, self.PARTICLE_COUNT)
        best_swarm_x, best_swarm_y, best_swarm_fitness = PSO.run()
        self.plot(best_swarm_x, best_swarm_y, best_swarm_fitness)
    
    def plot(self, best_swarm_x, best_swarm_y, best_swarm_fitness):
        images = [Image.open(f"algorithms/temp_pics_iteration/{n}.png") for n in range(self.MAX_ITERATION)]
        images[0].save('PSO_animation.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        plt.title("PSO")
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, color='b', )
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')
        ax.plot(best_swarm_x, best_swarm_y, best_swarm_fitness, 'r.')
        self.canvas.draw()
