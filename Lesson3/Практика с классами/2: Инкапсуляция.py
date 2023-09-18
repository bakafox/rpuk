# Атрибут может быть объявлен приватным (внутренним)
# с помощью нижнего подчеркивания перед именем,
# но настоящего скрытия на самом деле не происходит –
# все на уровне соглашений.

class MyClass:
    def __init__(self):
        self._val = 3 # Приватное свойство класса
        self.__priv_val = 8  # Очень приватное свойство класса

    def _factorial(self, x): # Приватный метод класса
        fact = 1
        for i in range(1, x):
            fact *= i
        return fact

    def mult(self, x):  # Метод класса
        return self._val * self._factorial(x)
    
    # Свойство объекта. Не принимает параметров кроме self, вызывается без круглых скобок.
    # Определяется с помощью декоратора property
    @property
    def count(self):
        return self.__priv_val



elem = MyClass()
print(elem._val)  # Доступ к приватному свойству
print(elem._MyClass__priv_val)  # Доступ к очень приватному свойству