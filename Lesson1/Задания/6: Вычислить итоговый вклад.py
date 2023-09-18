import math

X = int(input('Введите сумму вклада: '))
Y = int(input('Введите срок вклада: '))
Z = 0.1 # 10%
print('Процент вклада:', Z*100, '%')

for i in range(0, Y):
    X += X*Z
print('Сумма вклада в конце срока:', X)