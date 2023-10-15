import pandas as pd

"""
Прочитываем входные таблицы .csv
"""
cinema_sessions = pd.read_csv('rpuk/Lesson5 (pandas)/Лаба 1+2 (Pandas)/cinema_sessions.csv', sep=' ')
titanic_with_labels = pd.read_csv('rpuk/Lesson5 (pandas)/Лаба 1+2 (Pandas)/titanic_with_labels.csv', sep=' ')

"""
Пол (sex):
отфильтровать строки, где пол не указан,
преобразовать оставшиеся в число 0/1.
"""
titanic_with_labels['sex'] = titanic_with_labels['sex'].str.lower().replace(['m', 'м', 'мужской'], 1).replace(['ж'], 0)
titanic_with_labels = titanic_with_labels[titanic_with_labels['sex'].isin([1, 0])]

"""
Номер ряда в зале (row_number):
заполнить вместо NAN максимальным значением ряда.
"""
titanic_with_labels['row_number'] = titanic_with_labels['row_number'].fillna(titanic_with_labels['row_number'].max())

"""
Количество выпитого в литрах (liters_drunk):
отфильтровать отрицательные значения и нереально большие
значения (выбросы). Вместо них заполнить средним.
"""
titanic_with_labels['liters_drunk'][titanic_with_labels['liters_drunk'] < 0] = titanic_with_labels['liters_drunk'].median()
titanic_with_labels['liters_drunk'][titanic_with_labels['liters_drunk'] > 9] = titanic_with_labels['liters_drunk'].median()

"""
Выводим результаты операций выше в файл results_5.1.csv
"""
titanic_with_labels.to_csv('rpuk/Lesson5 (pandas)/Лаба 1+2 (Pandas)/results_5.1.csv', index=False)





"""
Возраст (age):
разделить на 3 группы: дети (до 18 лет), взрослые (18 - 50),
пожилые (50+). закодировать в виде трёх столбцов
с префиксом age_. Старый столбец с age удалить.
"""
# titanic_with_labels['age'] =

"""
Напиток (drink):
преобразовать в число 0/1, был ли этот напиток хмельным.
"""
# titanic_with_labels['drink'] = titanic_with_labels['drink'].str.contains('beer') * 1

"""
Номер чека (check_number):
надо сопоставить со второй таблицей со временем сеанса
и закодировать в виде трёх столбцов, был ли это
утренний (morining) сеанс, дневной (day) или вечерний (evening).
"""
# cinema_sessions = cinema_sessions.set_index('check_number').sort_index()
# titanic_with_labels = titanic_with_labels.set_index('check_number').sort_index()
# full_table = pd.merge(cinema_sessions, titanic_with_labels, on='check_number')

"""
Записываем итоговую таблицу в файл results_5.2.csv
"""
# full_table.to_csv('rpuk/Lesson5 (pandas)/Лаба 1+2 (Pandas)/results_5.2.csv', index=False)
