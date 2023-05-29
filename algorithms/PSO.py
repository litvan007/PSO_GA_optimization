import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from PIL import Image

class Particle():
    def __init__(self, MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2, C3):
        self.MIN_X = MIN_X
        self.MAX_X = MAX_X
        self.SPACE_DIMENSION = SPACE_DIMENSION
        self.func = func
        self.W, self.C1, self.C2, self.C3 = W, C1, C2, C3
        # Случайным образом инициализируем положение частицы в пространстве
        self.position = [random.uniform(MIN_X, MAX_X) for i in range(self.SPACE_DIMENSION)]
        # Случайным образом инициализируем скорость частицы
        self.velocity = [random.uniform(-1, 1) for i in range(self.SPACE_DIMENSION)]
        # Вычисляем значение функции в текущем положении частицы
        self.fitness = self.func(self.position)
        # Инициализируем лучшее положение частицы текущим положением
        self.best_position = self.position.copy()
        # Инициализируем лучшее значение функции текущим значением функции
        self.best_fitness = self.fitness

    # Метод для обновления скорости частицы
    def update_velocity(self, best_position, best_n_position):
        for i in range(self.SPACE_DIMENSION):
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            cognitive_velocity = self.C1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = self.C2 * r2 * (best_position[i] - self.position[i])
            neighboor_velocity = self.C3 * r3 * (best_n_position[i] - self.position[i])
            self.velocity[i] = self.W * self.velocity[i] + cognitive_velocity + social_velocity + neighboor_velocity

    # Метод для обновления положения частицы
    def update_position(self):
        for i in range(self.SPACE_DIMENSION):
            self.position[i] += self.velocity[i]
            # Если частица вышла за границы пространства поиска, то ограничиваем ее положение
            if self.position[i] < self.MIN_X:
                self.position[i] = self.MIN_X
            elif self.position[i] > self.MAX_X:
                self.position[i] = self.MAX_X

    # Метод для обновления значения функции и лучшего положения частицы
    def update_fitness(self):
        self.fitness = self.func(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class Common():
    def __init__(self, X, Y, Z, MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2, C3, MAX_ITERATION, PARTICLE_COUNT):
        self.MAX_ITERATION, self.PARTICLE_COUNT = MAX_ITERATION, PARTICLE_COUNT
        self.swarm = [Particle(MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2, C3) for _ in range(PARTICLE_COUNT)]
        self.X = X
        self.Y = Y
        self.Z = Z
        self.func = func

    def run(self):
        best_swarm_position = self.swarm[0].position.copy()
        best_swarm_fitness = self.swarm[0].fitness

        for particle in self.swarm:
            if particle.fitness < best_swarm_fitness:
                best_swarm_position = particle.position.copy()
                best_swarm_fitness = particle.fitness

        # Основной цикл алгоритма оптимизации
        iteration = 0 # номер итерации алгоритма
        position_history = []
        while iteration < self.MAX_ITERATION:
            temp_history = []
            for j, particle in enumerate(self.swarm):
                curr_particle_position = particle.position.copy()
                temp_history.append(curr_particle_position)

                left = (j - 1) % self.PARTICLE_COUNT # индекс левого соседа
                right = (j + 1) % self.PARTICLE_COUNT # индекс правого соседа

                p_best_left = self.swarm[left].best_position
                p_best_right = self.swarm[right].best_position
                if self.func(self.swarm[left].best_position) < self.func(self.swarm[right].best_position):
                    best_n_position = p_best_left # левый сосед лучше правого
                else:
                    best_n_position = p_best_right # правый сосед лучше левого

                particle.update_velocity(best_swarm_position, best_n_position)
                particle.update_position()
                particle.update_fitness()

                if particle.fitness < best_swarm_fitness:
                    best_swarm_position = particle.position.copy()
                    best_swarm_fitness = particle.fitness
            
            position_history.append(temp_history)
            iteration += 1 # увеличиваем номер итерации

        # Выводим результат оптимизации на экран
        print("Лучшее решение: ", best_swarm_position)
        print("Значение функции в лучшем решении: ", best_swarm_fitness)

        return position_history, best_swarm_position[0], best_swarm_position[1], best_swarm_fitness
