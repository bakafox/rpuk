import calendar

year = int(input('Введите год: '))
print(calendar.isleap(year)) # иногда, самое простое решение - лучшее решение

# более скучное решение:
print (((year % 4 == 0) and not (year % 100 == 0)) or (year % 400 == 0))