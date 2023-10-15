import argparse
import numpy as np


"""
Задаём и парсим аргументы в постпозициональной записи
(не требуют объявления через флаг и всегда обязательны),
после чего парсим и проверяем на корректность входные значения.
"""
parser = argparse.ArgumentParser()
parser.add_argument('file1', type=str)
parser.add_argument('file2', type=str)
parser.add_argument('P', type=float)
# python3 random_select.py 'file_1.txt' 'file_2.txt' 0.2
args = parser.parse_args()

f_dataset_real = open(args.file1, 'r')
f_dataset_synth = open(args.file2, 'r')
dataset_real = np.array(f_dataset_real.read().rstrip().split(' '), dtype=int)
dataset_synth = np.array(f_dataset_synth.read().rstrip().split(' '), dtype=int)
if (len(dataset_real) != len(dataset_synth)):
    print('file1, file2: Датасеты должны иметь одинаковую длину.')

mix_probability = args.P
if (mix_probability < 0 or mix_probability > 1):
    print('P: Вероятность смешения должна лежать в диапазоне от 0 до 1.')


"""
Фукнция для выбора одного из двух элементов с вероятностью prob,
Использует метод random.choice для выбора с вероятностью.
Поэлементно применяется к np.array посредством метода vectorize.
"""
def choose_elm_with_prob(elm_1, elm_2, prob):
    if np.random.choice([0, 1], p=[prob, 1-prob]):
        return elm_1
    else:
        return elm_2

mix_datasets = np.vectorize(choose_elm_with_prob)
dataset_mixed = mix_datasets(dataset_real, dataset_synth, mix_probability)
print(dataset_mixed)
