import methods
from numpy import sin, cos, arcsin, arccos, exp
import numpy as np


def input_func(x):
    global func_string
    return eval(func_string)


while True:
    print('select method: \n',
          '1. gradient descent \n',
          '2. gradient projection \n',
          '3. conditional gradient')
    method = int(input())
    if method == 1:
        print(
            'Input function: (note: the way to input variables is x[0], x[1], etc) ')
        func_string = input()
        init_x = x0 = np.array(
            list(map(float, (input("Input initial x: ").split)())))
        print(methods.gradient_descent(input_func, init_x))

    elif method == 2:
        print(
            'Input function: (note: the way to input variables is x[0], x[1], etc) ')
        func_string = input()
        print('select method: \n',
              '1. sphere \n',
              '2. plane \n')

        func_string = input()