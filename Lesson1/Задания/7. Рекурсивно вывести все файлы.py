import os

def scandir(dirpath):
    for file in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, file)):
            scandir(os.path.join(dirpath, file))
        else:
            print(os.path.join(dirpath, file))
            

startdir = input('Стартовая директория (пропуск для текущей): ')
if startdir == '':
    startdir = os.getcwd()

scandir(startdir)