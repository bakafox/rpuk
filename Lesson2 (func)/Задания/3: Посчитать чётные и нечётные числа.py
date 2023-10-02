import random

rndnum = random.randint(1, 999999)
print(f'Генерация случайного списка из {rndnum} чисел…')

rndlist = []
for i in range(rndnum):
    rndlist.append(random.randint(1, 999999))

oddeven = [0,0]
for num in rndlist:
    if num % 2 == 0:
        oddeven[0] += 1
    else:
        oddeven[1] += 1
print(f'Чётных: {oddeven[0]}, нечётных: {oddeven[1]}.')