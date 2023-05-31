import sys
sys.path.append("./")

import numpy as np
from algorithms.GA import Common
import matplotlib.pyplot as plt
import math

# Количество частиц в рое
PARTICLE_COUNT = 100
# Количество измерений пространства поиска
SPACE_DIMENSION = 2
# Максимальное количество итераций алгоритма
MAX_GEN = 100

# Границы пространства поиска
MIN_X = -5.12
MAX_X = 5.12

p_c = 0.8
p_m = 0.1

def func(x):
    sum = 0
    for i in range(SPACE_DIMENSION):
        sum += x[i]**2 - 10 * np.cos(2 * math.pi * x[i])
    return 10 * SPACE_DIMENSION + sum

def func_new(x):
    return np.cos(x[0]) * np.sin(x[1])

X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
Z = func([X, Y])

GA = Common(MIN_X, MAX_X, SPACE_DIMENSION, PARTICLE_COUNT, MAX_GEN, func, p_c, p_m)
a, b, best_gen_positions, best_gen_fitness = GA.run()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5)
ax.plot(best_gen_positions[0], best_gen_positions[1], best_gen_fitness, 'g.', marker='o', markersize=7)
plt.show()