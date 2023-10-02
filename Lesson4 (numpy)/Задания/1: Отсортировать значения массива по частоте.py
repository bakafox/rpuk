import numpy as np

arr_input = input('Введите значения массива, разделённые пробелом: ')

arr = np.array(arr_input.strip().split(' '), dtype=int)
arr_unique = np.unique(arr, return_counts=True)
arr_freq = np.flip(np.argsort(arr_unique[1]))

print('Уникальные значения по частоте: ', arr_unique[0][arr_freq])
