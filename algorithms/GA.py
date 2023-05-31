import numpy as np
import random

class Common():
    def __init__(self, MIN_X, MAX_X, SPACE_DIMENSION, POPULATION_COUNT, MAX_GEN, func, p_c, p_m) -> None:
        self.p_c, self.p_m = p_c, p_m
        self.func = func
        self.SPACE_DIMENSION = SPACE_DIMENSION
        self.MAX_GEN = MAX_GEN
        self.POPULATION_COUNT = POPULATION_COUNT
        self.population = np.array([[random.uniform(MIN_X, MAX_X) for _ in range(self.SPACE_DIMENSION)] for _ in range(self.POPULATION_COUNT)])
        self.fit = np.array([func(self.population[i]) for i in range(self.POPULATION_COUNT)]) # приспособленности
        print(self.population)

    def run(self):
        epoch = 0
        count_best_history = []
        position_history = []
        while epoch < self.MAX_GEN:
            sel = np.random.choice(self.POPULATION_COUNT, size=self.POPULATION_COUNT, replace=True)
            parents = self.population[sel] # должно меняться по количеству

            # Применение оператора кроссовера
            offspring = [] # список потомков
            num = self.POPULATION_COUNT if self.POPULATION_COUNT % 2 == 0 else self.POPULATION_COUNT-1
            for i in range(0, num, 2): # берем особей парами # TODO
                p1 = parents[i] # первый родитель
                p2 = parents[i+1] # второй родитель
                if np.random.random() < self.p_c: # если выполняется условие кроссовера
                    point = np.random.randint(1, self.SPACE_DIMENSION) # выбираем точку разрыва
                    c1 = np.concatenate([p1[:point], p2[point:]]) # первый потомок
                    c2 = np.concatenate([p2[:point], p1[point:]]) # второй потомок
                    offspring.append(c1) # добавляем первого потомка в список
                    offspring.append(c2) # добавляем второго потомка в список
                else: # если не выполняется условие кроссовера
                    offspring.append(p1) # добавляем первого родителя в список без изменений
                    offspring.append(p2) # добавляем второго родителя в список без изменений
                
            offspring = np.array(offspring) # преобразуем список потомков в массив

            # Применение оператора мутации
            for i in range(num): # для каждой особи в популяции
                if np.random.random() < self.p_m: # если выполняется условие мутации
                    point = np.random.randint(self.SPACE_DIMENSION) # выбираем точку мутации
                    value = np.random.uniform(-10, 10) # выбираем новое значение гена
                    offspring[i][point] = value # заменяем старое значение гена на новое

            count_best = 0
            # Обновление популяции и приспособленности
            for i in range(num): # для каждой особи в популяции
                if self.func(offspring[i]) < self.func(self.population[i]): # если потомок лучше родителя
                    self.population[i] = offspring[i] # заменить родителя на потомка
                    count_best += 1
            count_best_history.append(count_best)
            position_history.append(self.population.copy())

            self.fit = np.array([self.func(self.population[i]) for i in range(num)])# вычисляем приспособленность для новой популяции

            best_index = np.argmin(self.fit) # индекс лучшей особи в популяции
            best_value = self.fit[best_index] # значение целевой функции для лучшей особи
            best_solution = self.population[best_index] # лучшее решение в популяции
            print(f"Поколение {epoch+1}: best_value = {best_value}, best_solution = {best_solution}")
            epoch += 1

        return count_best_history, position_history, best_solution, best_value
