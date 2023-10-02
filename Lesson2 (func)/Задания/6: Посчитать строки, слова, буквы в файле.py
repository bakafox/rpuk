filepath = input('Введите путь до файла (пропуск - исп. образец): ')
if filepath == '':
    filepath = 'rpuk/Lesson2 (func)/Задания/6: sample.txt'

lines = words = letters = 0

with open(filepath, 'r') as f_input:
    for input_line in f_input:
        lines += 1
        words += len(input_line.split(' '))
        letters += len(input_line.replace(' ', '').replace('\n', ''))

print(f'Файл содержит {lines} строк, {words} слов, {letters} букв.')