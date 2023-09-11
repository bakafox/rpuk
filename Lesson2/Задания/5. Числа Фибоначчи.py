def fib(n):
    if n < 0:
        print('Число должно быть неотрицательным!')
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

inputnum = int(input('Введите число: '))
print(fib(inputnum))