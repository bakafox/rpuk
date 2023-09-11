gp_iter = int(input('Введите кол-во итераций прогрессии: '))
gp_init = int(input('Ввдеите начальное значение прогрессии: '))
gp_step = int(input('Введите шаг прогрессии: '))

def gp(init, step):
    while True:
        point = init * step**(i)
        yield point

for i in range(gp_iter):
    print(next(gp(gp_init, gp_step)))