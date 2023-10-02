class Item:
    def __init__(self, count=1, max_count=16):
        self._max_count = max_count
        self.update_count(count)

    @classmethod
    def my_name(cls):
        return cls.__name__

    @property
    def count(self):
        return self._count

    def check_count(self, val):
        if val <= self._max_count and val >= 0:
            return val
        else:
            print(f'{self.my_name()}: Выполнение операции невозможно.')
            return -1
    def update_count(self, val):
        if val <= self._max_count and val >= 0:
            self._count = val
            return True
        else:
            print(f'{self.my_name()}: Неверно указано кол-во предметов.')
            return False

    def __call__(self):
        if self.count > 0:
            new_count = self._count -1
            self.update_count(new_count)
            print(f'Вы съели один {self.my_name()}')    # смотря какой fabric
    def __str__(self):
        return f'Набор из {self.count} {self.my_name()}' 
    def __len__(self):
        return self._count

    def __add__(self, num):
        self.check_count(self._count + num)
        return self
    def __mul__(self, num):
        self.check_count(self._count * num)
        return self
    def __sub__(self, num):
        self.check_count(self._count - num)
        return self

    def __iadd__(self, num):
        self.update_count(self._count + num)
        return self
    def __imul__(self, num):
        self.update_count(self._count * num)
        return self
    def __isub__(self, num):
        self.update_count(self._count - num)
        return self

    def __lt__(self, num):
        return self._count < num
    def __le__(self, num):
        return self._count <= num
    def __gt__(self, num):
        return self._count > num
    def __ge__(self, num):
        return self._count >= num
    def __eq__(self, num):
        return self._count == num


class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe

class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    @property
    def eatable(self):
        return self._saturation > 0


class Brick(Item):
    def __init__(self, count=1, max_count=16, color='red'):
        super().__init__(count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='red', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe

class Plum(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=64, saturation=4):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)

    @property
    def eatable(self):
        return super().eatable and self._ripe

class Lemon(Fruit, Food):
    def __init__(self, ripe, sourness=5, count=1, max_count=32, saturation=8):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._sourness = sourness
    @property
    def eatable(self):
        return super().eatable and self._ripe and (self._sourness + 2) < self._saturation

class IceCream(Food):
    def __init__(self, count=1, max_count=16, flavor='vanilla'):
        super().__init__(count=count, max_count=max_count)
        self._flavor = flavor
    @property
    def flavor(self):
        return self._flavor

class PotatoChipsPackages(Food):
    def __init__(self, packages_count=1, chips_in_package=1, max_packages_count=8, max_chips_count=64, saturation=5):
        super().__init__(saturation=saturation, count=packages_count*chips_in_package, max_count=max_packages_count*max_chips_count)

    def __call__(self):
        if self._count >= 8:
            new_count = max(self._count -8, 0)
            self.update_count(new_count)
            print('Вы, довольно урча, зохавали 8 чипсов из пачки.')
        elif self._count > 0:
            self.update_count(0)
            print('Вы, грустно урча, доели остатки чипсов.')
        else:
            print('Увы, пачка совсем пуста. Вы не съели ни одного чипса.')