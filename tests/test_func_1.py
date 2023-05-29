import sys
sys.path.append("./")

import numpy as np
from algorithms.PSO import Common
import matplotlib.pyplot as plt
import math

# Количество частиц в рое
PARTICLE_COUNT = 100
# Количество измерений пространства поиска
SPACE_DIMENSION = 2
# Максимальное количество итераций алгоритма
MAX_ITERATION = 100

# Параметры алгоритма
W = 0.729 # инерционный вес
C1 = 1.49445 # коэффициент личного лучшего значения
C2 = 1.49445 # коэффициент глобального лучшего значения

# Границы пространства поиска
MIN_X = -5.12
MAX_X = 5.12

def func(x):
    sum = 0
    for i in range(SPACE_DIMENSION):
        sum += x[i]**2 - 10 * np.cos(2 * math.pi * x[i])
    return 10 * SPACE_DIMENSION + sum

X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z = func([X, Y])

PSO = Common(X, Y, Z, MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2, MAX_ITERATION, PARTICLE_COUNT)
best_swarm_x, best_swarm_y, best_swarm_fitness = PSO.run()

PSO.drawPaht(best_swarm_x, best_swarm_y, best_swarm_fitness)