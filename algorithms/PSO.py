import numpy as np
import random

class Particle():
    def __init__(self, GRAPH_MIN_X, GRAPH_MAX_X,  MIN_X, MAX_X, MIN_MASS, SPACE_DIMENSION, func, W, C1, C2, C3):
        self.MIN_X = MIN_X
        self.MAX_X = MAX_X
        self.SPACE_DIMENSION = SPACE_DIMENSION
        self.func = func
        self.W, self.C1, self.C2, self.C3 = W, C1, C2, C3
        self.MIN_MASS = MIN_MASS
        self.position = [random.uniform(GRAPH_MIN_X, GRAPH_MAX_X) for _ in range(self.SPACE_DIMENSION)]
        self.velocity = [random.uniform(-1, 1) for _ in range(self.SPACE_DIMENSION)]
        self.fitness = self.func(self.position)
        self.best_position = self.position.copy()
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

    def update_mass(self, fitness_min, fitness_max):
        self.m = (self.fitness - fitness_min) / (fitness_max - fitness_min)
        return (self.m > self.MIN_MASS)

    def simulated_annealing(self, T):
        delta = np.random.uniform(-0.1, 0.1, size=self.SPACE_DIMENSION)
        temp_position = self.position + delta
        delta_fitness = self.func(temp_position) - self.fitness

        if delta_fitness <= 0:
            self.position = temp_position
        # Вероятность exp(-delta_f / T)
        else:
            p = np.exp(-delta_fitness / T)
            if np.random.uniform() < p:
                self.position = temp_position

class Common():
    def __init__(self, GRAPH_MIN_X, GRAPH_MAX_X, MIN_X, MAX_X, MIN_MASS, MIN_T, MAX_T, ALPHA, SPACE_DIMENSION, func, W, C1, C2, C3, MAX_ITERATION, PARTICLE_COUNT, USE_EXTINCTION, USE_ANNEALING):
        self.MAX_ITERATION, self.PARTICLE_COUNT = MAX_ITERATION, PARTICLE_COUNT
        self.swarm = [Particle(GRAPH_MIN_X, GRAPH_MAX_X, MIN_X, MAX_X, MIN_MASS, SPACE_DIMENSION, func, W, C1, C2, C3) for _ in range(PARTICLE_COUNT)]
        self.func = func
        self.MIN_T, self.MAX_T, self.ALPHA = MIN_T, MAX_T, ALPHA
        self.USE_EXTINCTION, self.USE_ANNEALING = USE_EXTINCTION, USE_ANNEALING

    def run(self):
        best_swarm_position = self.swarm[0].position.copy()
        best_swarm_fitness = self.swarm[0].fitness

        bad_swarm_fitness = self.swarm[-1].fitness

        for particle in self.swarm:
            if particle.fitness < best_swarm_fitness:
                best_swarm_position = particle.position.copy()
                best_swarm_fitness = particle.fitness

        # Основной цикл алгоритма оптимизации
        iteration = 0 
        position_history = []
        particle_count_history = [self.PARTICLE_COUNT]
        while iteration < self.MAX_ITERATION:
            temp_history = []
            # Задаем начальную температуру для эффекта отжига
            T = self.MAX_T
            for j, particle in enumerate(self.swarm):
                curr_particle_position = particle.position.copy()
                temp_history.append(curr_particle_position)

                left = (j - 1) % self.PARTICLE_COUNT 
                right = (j + 1) % self.PARTICLE_COUNT 

                p_best_left = self.swarm[left].best_position
                p_best_right = self.swarm[right].best_position
                if self.func(self.swarm[left].best_position) < self.func(self.swarm[right].best_position):
                    best_n_position = p_best_left 
                else:
                    best_n_position = p_best_right 

                particle.update_velocity(best_swarm_position, best_n_position)
                particle.update_position()
                particle.update_fitness()

                if particle.fitness < best_swarm_fitness:
                    best_swarm_position = particle.position.copy()
                    best_swarm_fitness = particle.fitness
                
                if particle.fitness > bad_swarm_fitness:
                    bad_swarm_fitness = particle.fitness

                # Эффект отжига
                if self.USE_ANNEALING:
                    particle.simulated_annealing(T)    
                    temp = T * self.ALPHA
                    T = self.MIN_T if temp < self.MIN_T else temp

                # Проверка на выживаемость
                if self.USE_EXTINCTION:
                    isAlive = particle.update_mass(best_swarm_fitness, bad_swarm_fitness)
                    if not isAlive:
                        print(f'{j}s particle is dead')
                        del self.swarm[j]
                        self.PARTICLE_COUNT -= 1

            particle_count_history.append(self.PARTICLE_COUNT)
            position_history.append(temp_history)

            # Обновление значения температуры
            iteration += 1 

        print("Лучшее решение: ", best_swarm_position)
        print("Значение функции в лучшем решении: ", best_swarm_fitness)

        return particle_count_history, position_history, best_swarm_position, best_swarm_fitness
