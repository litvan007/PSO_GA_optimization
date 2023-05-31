import numpy as np
import random

class Common():
    def __init__(self, GRAPH_MIN_X, GRAPH_MAX_X, MIN_X, MAX_X, SPACE_DIMENSION, POPULATION_COUNT, MAX_GEN, func, p_c, p_m) -> None:
        self.p_c, self.p_m = p_c, p_m
        self.func = func
        self.SPACE_DIMENSION = SPACE_DIMENSION
        self.MAX_GEN = MAX_GEN
        self.POPULATION_COUNT = POPULATION_COUNT
        self.population = np.array([[random.uniform(GRAPH_MIN_X, GRAPH_MAX_X) for _ in range(self.SPACE_DIMENSION)] for _ in range(self.POPULATION_COUNT)])
        self.fit = np.array([func(self.population[i]) for i in range(self.POPULATION_COUNT)]) # приспособленности

    def run(self):
        epoch = 0
        count_best_history = []
        position_history = []
        while epoch < self.MAX_GEN:
            sel = np.random.choice(self.POPULATION_COUNT, size=self.POPULATION_COUNT, replace=True)
            parents = self.population[sel] 

            # Применение оператора кроссовера
            offspring = []
            num = self.POPULATION_COUNT if self.POPULATION_COUNT % 2 == 0 else self.POPULATION_COUNT-1
            for i in range(0, num, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                if np.random.random() < self.p_c:
                    point = np.random.randint(1, self.SPACE_DIMENSION)
                    c1 = np.concatenate([p1[:point], p2[point:]])
                    c2 = np.concatenate([p2[:point], p1[point:]])
                    offspring.append(c1)
                    offspring.append(c2)
                else:
                    offspring.append(p1)
                    offspring.append(p2)
                
            offspring = np.array(offspring)

            # Применение оператора мутации
            for i in range(num):
                if np.random.random() < self.p_m:
                    point = np.random.randint(self.SPACE_DIMENSION)
                    value = np.random.uniform(-10, 10)
                    offspring[i][point] = value

            count_best = 0
            # Обновление популяции и приспособленности
            for i in range(num):
                if self.func(offspring[i]) < self.func(self.population[i]):
                    self.population[i] = offspring[i]
                    count_best += 1
            count_best_history.append(count_best)
            position_history.append(self.population.copy())

            self.fit = np.array([self.func(self.population[i]) for i in range(num)])

            best_index = np.argmin(self.fit) 
            best_value = self.fit[best_index]
            best_solution = self.population[best_index]
            print(f"Поколение {epoch+1}: best_value = {best_value}, best_solution = {best_solution}")
            epoch += 1

        return count_best_history, position_history, best_solution, best_value
