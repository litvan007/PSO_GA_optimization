import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random
from PIL import Image

class Particle():
    def __init__(self, MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2):
        self.MIN_X = MIN_X
        self.MAX_X = MAX_X
        self.SPACE_DIMENSION = SPACE_DIMENSION
        self.func = func
        self.W, self.C1, self.C2 = W, C1, C2
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
    def update_velocity(self, best_position):
        for i in range(self.SPACE_DIMENSION):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = self.C1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = self.C2 * r2 * (best_position[i] - self.position[i])
            self.velocity[i] = self.W * self.velocity[i] + cognitive_velocity + social_velocity

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
    def __init__(self, X, Y, Z, MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2, MAX_ITERATION, PARTICLE_COUNT):
        self.MAX_ITERATION, self.PARTICLE_COUNT = MAX_ITERATION, PARTICLE_COUNT
        self.swarm = [Particle(MIN_X, MAX_X, SPACE_DIMENSION, func, W, C1, C2) for i in range(PARTICLE_COUNT)]
        self.X = X
        self.Y = Y
        self.Z = Z

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
            # Обновляем скорость, положение и значение функции для каждой частицы в рое
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            plt.title("PSO")
            ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, color='b')
            ax.set_xlabel('x label', color='r')
            ax.set_ylabel('y label', color='g')
            ax.set_zlabel('z label', color='b')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(0, 100)

            temp_history = []
            for particle in self.swarm:
                curr_particle_position = particle.position.copy()
                temp_history.append(curr_particle_position)
                func = particle.func
                ax.plot(curr_particle_position[0], curr_particle_position[1], func(curr_particle_position), 'r.')

                particle.update_velocity(best_swarm_position)
                particle.update_position()
                particle.update_fitness()

                if particle.fitness < best_swarm_fitness:
                    best_swarm_position = particle.position.copy()
                    best_swarm_fitness = particle.fitness
            
            position_history.append(temp_history)
            plt.savefig(f"algorithms/temp_pics_iteration/{iteration}.png")
            plt.close()
            iteration += 1 # увеличиваем номер итерации

        # Выводим результат оптимизации на экран
        print("Лучшее решение: ", best_swarm_position)
        print("Значение функции в лучшем решении: ", best_swarm_fitness)

        images = [Image.open(f"algorithms/temp_pics_iteration/{n}.png") for n in range(self.MAX_ITERATION)]
        images[0].save('PSO_animation.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

        return position_history, best_swarm_position[0], best_swarm_position[1], best_swarm_fitness
