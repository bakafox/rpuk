import argparse
import random

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-N', type=int, required=True)
args = parser.parse_args()


rndarray = []
for i in range(args.N):   # в range можно пропустить начальный аргумент, если он =0
    rndarray.append(random.random())

print('\nНесортированный список:')
for num in rndarray:
    print(num)

for i in range(0, len(rndarray)):
    for j in range(0, len(rndarray)-1):
        if rndarray[j] > rndarray[j+1]:
            tmp = rndarray[j]
            rndarray[j] = rndarray[j+1]
            rndarray[j+1] = tmp

print('\nБахнув бубль сорту:')
for num in rndarray:
    print(num)