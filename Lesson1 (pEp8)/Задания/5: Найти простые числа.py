import math

N = int(input('Поиск в диапазоне: 1—')) # на самом деле поиск идёт с 2, а 0 и 1 сразу =False, но об этом никто никогда не узнает

sieve = []
sieve.append(False)
sieve.append(False)
for num in range(2, N):
    sieve.append(True)

for num in range(2, N):
    if sieve[num] == True:
        for not_prime_num in range(num ** 2, N, num):   # шаг размером num (вычёркиваем все соотв. шагу числа)
            sieve[not_prime_num] = False

for num in range(0, N):
    if sieve[num] == True:
        print(num)