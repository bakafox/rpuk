import random

from lnp_matrices import *


"""
Класс Бухгалтера. Имеет бесконечное количество денег,
которые он может выдавать работникам в качестве З/П через
метод give_salary (принимает на вход работника для выдачи З/П).
"""
class Accountant:
    def give_salary(self, worker):
        # это в долларах, если что
        if (isinstance(worker, Loopa)):
            return (30000 + random.randrange(0, 5000))
        elif (isinstance(worker, Poopa)):
            return (25000 + random.randrange(0, 15000))
        else:
            print('Accountant: НЕТ ТАКОГО! СМОТРИ ЧТО ВВОДИШЬ, МУД@ЛАЙ!!')
            return 1


"""
Общий класс для всех работников компании.
Содержит атрибут для хранения денег, а также метод take_salary
для получения зарплаты (принимает на вход сумму З/П для выдачи).
"""
class Employee:
    def __init__(self, deneg=0):
        self._deneg = deneg

    def get_deneg(self):
        return self._deneg

    def take_salary(self, dengi):
        self._deneg += dengi


"""
Класс Лупы, имеет метод do_work для выполнения работы
(сложения матриц) -- принимает на вход файл с двумя матрицами
и файл для сохранения результата (также выводит в консоль).
"""
class Loopa(Employee):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do_work(self, filename1, filename2):
        [matrix1, matrix2] = parse_matrices(filename1)

        matrixR = sum_matrices(matrix1, matrix2)
        if not matrixR:
            print('Loopa: Эти матрицы нельзя сложить!')
        else:
            print('Loopa: Матрицы успешно сложены! Результат:')
            output_matrices(filename2, [matrixR])


"""
Класс Пупы, имеет метод do_work для выполнения работы
(вычитания матриц) -- принимает на вход файл с двумя матрицами
и файл для сохранения результата (также выводит в консоль).
"""
class Poopa(Employee):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def do_work(self, filename1, filename2):
        [matrix1, matrix2] = parse_matrices(filename1)

        matrixR = sub_matrices(matrix1, matrix2)
        if not matrixR:
            print('Poopa: Эти матрицы нельзя вычесть!')
        else:
            print('Poopa: Матрицы успешно вычтены! Результат:')
            output_matrices(filename2, [matrixR])





l = Loopa()
p = Poopa()
print(l.get_deneg(), p.get_deneg())

a = Accountant()

a.give_salary('fd')

print()
l.do_work('rpuk/Lesson3 (class)/Лаба 1 (ООП)/matrices1.txt',
          'rpuk/Lesson3 (class)/Лаба 1 (ООП)/result_l1.txt')
l.take_salary(a.give_salary(l))
print(f'Денег у Лупы: {l.get_deneg()}, денег у Пупы: {p.get_deneg()}')

print()
p.do_work('rpuk/Lesson3 (class)/Лаба 1 (ООП)/matrices1.txt',
          'rpuk/Lesson3 (class)/Лаба 1 (ООП)/result_p1.txt')
p.take_salary(a.give_salary(p))
print(f'Денег у Лупы: {l.get_deneg()}, денег у Пупы: {p.get_deneg()}')

print()
l.do_work('rpuk/Lesson3 (class)/Лаба 1 (ООП)/matrices2.txt',
          'rpuk/Lesson3 (class)/Лаба 1 (ООП)/result_l2.txt')
print()
p.do_work('rpuk/Lesson3 (class)/Лаба 1 (ООП)/matrices2.txt',
          'rpuk/Lesson3 (class)/Лаба 1 (ООП)/result_p2.txt')

print("\nAnd now, the moment we all've been waiting for:")
print(f'Денег у Лупы: {l.get_deneg()}, денег у Пупы: {p.get_deneg()}')
l.take_salary(a.give_salary(p))
print(f'Денег у Лупы: {l.get_deneg()}, денег у Пупы: {p.get_deneg()}')
p.take_salary(a.give_salary('behind the loopa'))
print(f'Денег у Лупы: {l.get_deneg()}, денег у Пупы: {p.get_deneg()}')
