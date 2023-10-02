inputdict = {}

while True:
    inputdict_input_entry = input('Введите слово и его синоним через пробел (пропуск - конец ввода): ')
    if inputdict_input_entry == '':
        break
    inputdict[inputdict_input_entry.split()[0]] = inputdict_input_entry.split()[1]

outputdict = {}
for inputdict_entry in inputdict.items():
    outputdict[inputdict_entry[1]] = inputdict_entry[0]

print(outputdict)