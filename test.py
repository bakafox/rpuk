print('ok') # эта программа (test.py) выводит в терминал строчку 'ok'. это означает, что программа сработала.

# в случае, если программа не выводит строчку 'ok', либо выдаёт ошибку и не запускается, попробуйте выполнить следующие действия:
# 1. перезагрузите (выключитье и снова включите) ваш компьютер.



# если это не помогло, ознакомьтесь с этой очень краткой справкой по ошибкам в Python:

# Синтаксические ошибки (SyntaxError)

# Синтаксические ошибки часто называют ошибками разбора. Они возникают, когда интерпретатор обнаруживает синтаксическую проблему в коде.

# Рассмотрим на примере.

# a = 8
# b = 10
# c = a b

# File "<ipython-input-8-3b3ffcedf995>", line 3
#  c = a b
#        ^
# SyntaxError: invalid syntax

# Стрелка вверху указывает на место, где интерпретатор получил ошибку при попытке исполнения. Знак перед стрелкой указывает на причину проблемы. Для устранения таких фундаментальных ошибок Python будет делать большую часть работы за программиста, выводя название файла и номер строки, где была обнаружена ошибка.
# Недостаточно памяти (OutofMemoryError)

# Ошибки памяти чаще всего связаны с оперативной памятью компьютера и относятся к структуре данных под названием “Куча” (heap). Если есть крупные объекты (или) ссылки на подобные, то с большой долей вероятности возникнет ошибка OutofMemory. Она может появиться по нескольким причинам:

#     Использование 32-битной архитектуры Python (максимальный объем выделенной памяти невысокий, между 2 и 4 ГБ);
#     Загрузка файла большого размера;
#     Запуск модели машинного обучения/глубокого обучения и много другое;

# Обработать ошибку памяти можно с помощью обработки исключений — резервного исключения. Оно используется, когда у интерпретатора заканчивается память и он должен немедленно остановить текущее исполнение. В редких случаях Python вызывает OutofMemoryError, позволяя скрипту каким-то образом перехватить самого себя, остановить ошибку памяти и восстановиться.

# Но поскольку Python использует архитектуру управления памятью из языка C (функция malloc()), не факт, что все процессы восстановятся — в некоторых случаях MemoryError приведет к остановке. Следовательно, обрабатывать такие ошибки не рекомендуется, и это не считается хорошей практикой.
# Ошибка рекурсии (RecursionError)

# Эта ошибка связана со стеком и происходит при вызове функций. Как и предполагает название, ошибка рекурсии возникает, когда внутри друг друга исполняется много методов (один из которых — с бесконечной рекурсией), но это ограничено размером стека.

# Все локальные переменные и методы размещаются в стеке. Для каждого вызова метода создается стековый кадр (фрейм), внутрь которого помещаются данные переменной или результат вызова метода. Когда исполнение метода завершается, его элемент удаляется.

# Чтобы воспроизвести эту ошибку, определим функцию recursion, которая будет рекурсивной — вызывать сама себя в бесконечном цикле. В результате появится ошибка StackOverflow или ошибка рекурсии, потому что стековый кадр будет заполняться данными метода из каждого вызова, но они не будут освобождаться.

# def recursion():
#     return recursion()

# recursion()

# ---------------------------------------------------------------------------

# RecursionError                            Traceback (most recent call last)

# <ipython-input-3-c6e0f7eb0cde> in <module>
# ----> 1 recursion()


# <ipython-input-2-5395140f7f05> in recursion()
#       1 def recursion():
# ----> 2     return recursion()


# ... last 1 frames repeated, from the frame below ...


# <ipython-input-2-5395140f7f05> in recursion()
#       1 def recursion():
# ----> 2     return recursion()


# RecursionError: maximum recursion depth exceeded

# Ошибка отступа (IndentationError)

# Эта ошибка похожа по духу на синтаксическую и является ее подвидом. Тем не менее она возникает только в случае проблем с отступами.

# Пример:

# for i in range(10):
#     print('Привет Мир!')

#   File "<ipython-input-6-628f419d2da8>", line 2
#     print('Привет Мир!')
#         ^
# IndentationError: expected an indented block

# Исключения

# Даже если синтаксис в инструкции или само выражение верны, они все равно могут вызывать ошибки при исполнении. Исключения Python — это ошибки, обнаруживаемые при исполнении, но не являющиеся критическими. Скоро вы узнаете, как справляться с ними в программах Python. Объект исключения создается при вызове исключения Python. Если скрипт не обрабатывает исключение явно, программа будет остановлена принудительно.

# Программы обычно не обрабатывают исключения, что приводит к подобным сообщениям об ошибке:
# Ошибка типа (TypeError)

# a = 2
# b = 'PythonRu'
# a + b

# ---------------------------------------------------------------------------

# TypeError                                 Traceback (most recent call last)

# <ipython-input-7-86a706a0ffdf> in <module>
#       1 a = 2
#       2 b = 'PythonRu'
# ----> 3 a + b


# TypeError: unsupported operand type(s) for +: 'int' and 'str'

# Ошибка деления на ноль (ZeroDivisionError)

# 10 / 0

# ---------------------------------------------------------------------------

# ZeroDivisionError                         Traceback (most recent call last)

# <ipython-input-43-e9e866a10e2a> in <module>
# ----> 1 10 / 0


# ZeroDivisionError: division by zero

# Есть разные типы исключений в Python и их тип выводится в сообщении: вверху примеры TypeError и ZeroDivisionError. Обе строки в сообщениях об ошибке представляют собой имена встроенных исключений Python.

# Оставшаяся часть строки с ошибкой предлагает подробности о причине ошибки на основе ее типа.

# Теперь рассмотрим встроенные исключения Python.
# Встроенные исключения

# BaseException
#  +-- SystemExit
#  +-- KeyboardInterrupt
#  +-- GeneratorExit
#  +-- Exception
#       +-- StopIteration
#       +-- StopAsyncIteration
#       +-- ArithmeticError
#       |    +-- FloatingPointError
#       |    +-- OverflowError
#       |    +-- ZeroDivisionError
#       +-- AssertionError
#       +-- AttributeError
#       +-- BufferError
#       +-- EOFError
#       +-- ImportError
#       |    +-- ModuleNotFoundError
#       +-- LookupError
#       |    +-- IndexError
#       |    +-- KeyError
#       +-- MemoryError
#       +-- NameError
#       |    +-- UnboundLocalError
#       +-- OSError
#       |    +-- BlockingIOError
#       |    +-- ChildProcessError
#       |    +-- ConnectionError
#       |    |    +-- BrokenPipeError
#       |    |    +-- ConnectionAbortedError
#       |    |    +-- ConnectionRefusedError
#       |    |    +-- ConnectionResetError
#       |    +-- FileExistsError
#       |    +-- FileNotFoundError
#       |    +-- InterruptedError
#       |    +-- IsADirectoryError
#       |    +-- NotADirectoryError
#       |    +-- PermissionError
#       |    +-- ProcessLookupError
#       |    +-- TimeoutError
#       +-- ReferenceError
#       +-- RuntimeError
#       |    +-- NotImplementedError
#       |    +-- RecursionError
#       +-- SyntaxError
#       |    +-- IndentationError
#       |         +-- TabError
#       +-- SystemError
#       +-- TypeError
#       +-- ValueError
#       |    +-- UnicodeError
#       |         +-- UnicodeDecodeError
#       |         +-- UnicodeEncodeError
#       |         +-- UnicodeTranslateError
#       +-- Warning
#            +-- DeprecationWarning
#            +-- PendingDeprecationWarning
#            +-- RuntimeWarning
#            +-- SyntaxWarning
#            +-- UserWarning
#            +-- FutureWarning
#            +-- ImportWarning
#            +-- UnicodeWarning
#            +-- BytesWarning
#            +-- ResourceWarning

# Прежде чем переходить к разбору встроенных исключений быстро вспомним 4 основных компонента обработки исключения, как показано на этой схеме.

#     Try: он запускает блок кода, в котором ожидается ошибка.
#     Except: здесь определяется тип исключения, который ожидается в блоке try (встроенный или созданный).
#     Else: если исключений нет, тогда исполняется этот блок (его можно воспринимать как средство для запуска кода в том случае, если ожидается, что часть кода приведет к исключению).
#     Finally: вне зависимости от того, будет ли исключение или нет, этот блок кода исполняется всегда.

# В следующем разделе руководства больше узнаете об общих типах исключений и научитесь обрабатывать их с помощью инструмента обработки исключения.
# Ошибка прерывания с клавиатуры (KeyboardInterrupt)

# Исключение KeyboardInterrupt вызывается при попытке остановить программу с помощью сочетания Ctrl + C или Ctrl + Z в командной строке или ядре в Jupyter Notebook. Иногда это происходит неумышленно и подобная обработка поможет избежать подобных ситуаций.

# В примере ниже если запустить ячейку и прервать ядро, программа вызовет исключение KeyboardInterrupt. Теперь обработаем исключение KeyboardInterrupt.

# try:
#     inp = input()
#     print('Нажмите Ctrl+C и прервите Kernel:')
# except KeyboardInterrupt:
#     print('Исключение KeyboardInterrupt')
# else:
#     print('Исключений не произошло')

# Исключение KeyboardInterrupt

# Стандартные ошибки (StandardError)

# Рассмотрим некоторые базовые ошибки в программировании.
# Арифметические ошибки (ArithmeticError)

#     Ошибка деления на ноль (Zero Division);
#     Ошибка переполнения (OverFlow);
#     Ошибка плавающей точки (Floating Point);

# Все перечисленные выше исключения относятся к классу Arithmetic и вызываются при ошибках в арифметических операциях.
# Деление на ноль (ZeroDivisionError)

# Когда делитель (второй аргумент операции деления) или знаменатель равны нулю, тогда результатом будет ошибка деления на ноль.

# try:  
#     a = 100 / 0
#     print(a)
# except ZeroDivisionError:  
#     print("Исключение ZeroDivisionError." )
# else:  
#     print("Успех, нет ошибок!")

# Исключение ZeroDivisionError.

# Переполнение (OverflowError)

# Ошибка переполнение вызывается, когда результат операции выходил за пределы диапазона. Она характерна для целых чисел вне диапазона.

# try:  
#     import math
#     print(math.exp(1000))
# except OverflowError:  
#     print("Исключение OverFlow.")
# else:  
#     print("Успех, нет ошибок!")

# Исключение OverFlow.

# Ошибка утверждения (AssertionError)

# Когда инструкция утверждения не верна, вызывается ошибка утверждения.

# Рассмотрим пример. Предположим, есть две переменные: a и b. Их нужно сравнить. Чтобы проверить, равны ли они, необходимо использовать ключевое слово assert, что приведет к вызову исключения Assertion в том случае, если выражение будет ложным.

# try:  
#     a = 100
#     b = "PythonRu"
#     assert a == b
# except AssertionError:  
#     print("Исключение AssertionError.")
# else:  
#     print("Успех, нет ошибок!")

# Исключение AssertionError.

# Ошибка атрибута (AttributeError)

# При попытке сослаться на несуществующий атрибут программа вернет ошибку атрибута. В следующем примере можно увидеть, что у объекта класса Attributes нет атрибута с именем attribute.

# class Attributes(obj):
#     a = 2
#     print(a)

# try:
#     obj = Attributes()
#     print(obj.attribute)
# except AttributeError:
#     print("Исключение AttributeError.")

# 2
# Исключение AttributeError.

# Ошибка импорта (ModuleNotFoundError)

# Ошибка импорта вызывается при попытке импортировать несуществующий (или неспособный загрузиться) модуль в стандартном пути или даже при допущенной ошибке в имени.

# import nibabel

# ---------------------------------------------------------------------------

# ModuleNotFoundError                       Traceback (most recent call last)

# <ipython-input-6-9e567e3ae964> in <module>
# ----> 1 import nibabel


# ModuleNotFoundError: No module named 'nibabel'

# Ошибка поиска (LookupError)

# LockupError выступает базовым классом для исключений, которые происходят, когда key или index используются для связывания или последовательность списка/словаря неверна или не существует.

# Здесь есть два вида исключений:

#     Ошибка индекса (IndexError);
#     Ошибка ключа (KeyError);

# Ошибка ключа

# Если ключа, к которому нужно получить доступ, не оказывается в словаре, вызывается исключение KeyError.

# try:  
#     a = {1:'a', 2:'b', 3:'c'}  
#     print(a[4])  
# except LookupError:  
#     print("Исключение KeyError.")
# else:  
#     print("Успех, нет ошибок!")

# Исключение KeyError.

# Ошибка индекса

# Если пытаться получить доступ к индексу (последовательности) списка, которого не существует в этом списке или находится вне его диапазона, будет вызвана ошибка индекса (IndexError: list index out of range python).

# try:
#     a = ['a', 'b', 'c']  
#     print(a[4])  
# except LookupError:  
#     print("Исключение IndexError, индекс списка вне диапазона.")
# else:  
#     print("Успех, нет ошибок!")

# Исключение IndexError, индекс списка вне диапазона.

# Ошибка памяти (MemoryError)

# Как уже упоминалось, ошибка памяти вызывается, когда операции не хватает памяти для выполнения.
# Ошибка имени (NameError)

# Ошибка имени возникает, когда локальное или глобальное имя не находится.

# В следующем примере переменная ans не определена. Результатом будет ошибка NameError.

# try:
#     print(ans)
# except NameError:  
#     print("NameError: переменная 'ans' не определена")
# else:  
#     print("Успех, нет ошибок!")

# NameError: переменная 'ans' не определена

# Ошибка выполнения (Runtime Error)

# Ошибка «NotImplementedError»
# Ошибка выполнения служит базовым классом для ошибки NotImplemented. Абстрактные методы определенного пользователем класса вызывают это исключение, когда производные методы перезаписывают оригинальный.

# class BaseClass(object):
#     """Опередляем класс"""
#     def __init__(self):
#         super(BaseClass, self).__init__()
#     def do_something(self):
# 	# функция ничего не делает
#         raise NotImplementedError(self.__class__.__name__ + '.do_something')

# class SubClass(BaseClass):
#     """Реализует функцию"""
#     def do_something(self):
#         # действительно что-то делает
#         print(self.__class__.__name__ + ' что-то делает!')

# SubClass().do_something()
# BaseClass().do_something()

# SubClass что-то делает!



# ---------------------------------------------------------------------------

# NotImplementedError                       Traceback (most recent call last)

# <ipython-input-1-57792b6bc7e4> in <module>
#      14
#      15 SubClass().do_something()
# ---> 16 BaseClass().do_something()


# <ipython-input-1-57792b6bc7e4> in do_something(self)
#       5     def do_something(self):
#       6         # функция ничего не делает
# ----> 7         raise NotImplementedError(self.__class__.__name__ + '.do_something')
#       8
#       9 class SubClass(BaseClass):


# NotImplementedError: BaseClass.do_something

# Ошибка типа (TypeError)

# Ошибка типа вызывается при попытке объединить два несовместимых операнда или объекта.

# В примере ниже целое число пытаются добавить к строке, что приводит к ошибке типа.

# try:
#     a = 5
#     b = "PythonRu"
#     c = a + b
# except TypeError:
#     print('Исключение TypeError')
# else:
#     print('Успех, нет ошибок!')

# Исключение TypeError

# Ошибка значения (ValueError)

# Ошибка значения вызывается, когда встроенная операция или функция получают аргумент с корректным типом, но недопустимым значением.

# В этом примере встроенная операция float получат аргумент, представляющий собой последовательность символов (значение), что является недопустимым значением для типа: число с плавающей точкой.

# try:
#     print(float('PythonRu'))
# except ValueError:
#     print('ValueError: не удалось преобразовать строку в float: \'PythonRu\'')
# else:
#     print('Успех, нет ошибок!')

# ValueError: не удалось преобразовать строку в float: 'PythonRu'

# Пользовательские исключения в Python

# В Python есть много встроенных исключений для использования в программе. Но иногда нужно создавать собственные со своими сообщениями для конкретных целей.

# Это можно сделать, создав новый класс, который будет наследовать из класса Exception в Python.

# class UnAcceptedValueError(Exception):   
#     def __init__(self, data):    
#         self.data = data
#     def __str__(self):
#         return repr(self.data)

# Total_Marks = int(input("Введите общее количество баллов: "))
# try:
#     Num_of_Sections = int(input("Введите количество разделов: "))
#     if(Num_of_Sections < 1):
#         raise UnAcceptedValueError("Количество секций не может быть меньше 1")
# except UnAcceptedValueError as e:
#     print("Полученная ошибка:", e.data)

# Введите общее количество баллов: 10
# Введите количество разделов: 0
# Полученная ошибка: Количество секций не может быть меньше 1

# В предыдущем примере если ввести что-либо меньше 1, будет вызвано исключение. Многие стандартные исключения имеют собственные исключения, которые вызываются при возникновении проблем в работе их функций.
# Недостатки обработки исключений в Python

# У использования исключений есть свои побочные эффекты, как, например, то, что программы с блоками try-except работают медленнее, а количество кода возрастает.

# Дальше пример, где модуль Python timeit используется для проверки времени исполнения 2 разных инструкций. В stmt1 для обработки ZeroDivisionError используется try-except, а в stmt2 — if. Затем они выполняются 10000 раз с переменной a=0. Суть в том, чтобы показать разницу во времени исполнения инструкций. Так, stmt1 с обработкой исключений занимает больше времени чем stmt2, который просто проверяет значение и не делает ничего, если условие не выполнено.

# Поэтому стоит ограничить использование обработки исключений в Python и применять его в редких случаях. Например, когда вы не уверены, что будет вводом: целое или число с плавающей точкой, или не уверены, существует ли файл, который нужно открыть.

# import timeit
# setup="a=0"
# stmt1 = '''\
# try:
#     b=10/a
# except ZeroDivisionError:
#     pass'''

# stmt2 = '''\
# if a!=0:
#     b=10/a'''

# print("time=",timeit.timeit(stmt1,setup,number=10000))
# print("time=",timeit.timeit(stmt2,setup,number=10000))

# time= 0.003897680000136461
# time= 0.0002797570000439009

# Выводы!

# Как вы могли увидеть, обработка исключений помогает прервать типичный поток программы с помощью специального механизма, который делает код более отказоустойчивым.

# Обработка исключений — один из основных факторов, который делает код готовым к развертыванию. Это простая концепция, построенная всего на 4 блоках: try выискивает исключения, а except их обрабатывает.

# Очень важно поупражняться в их использовании, чтобы сделать свой код более отказоустойчивым.

# спрасибо за внимание