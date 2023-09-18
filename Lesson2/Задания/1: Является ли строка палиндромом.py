inputstr = input('Введите строку для проверки: ')

if len(inputstr) < 2:
    print('Слишком короткая строка.')

for i in range(len(inputstr)//2):
    if inputstr[i] != inputstr[len(inputstr)-1 -i]:
        print('Это не палиндром!')
        break
    
    if i == len(inputstr)//2 -1:
        print('Это палиндром!')