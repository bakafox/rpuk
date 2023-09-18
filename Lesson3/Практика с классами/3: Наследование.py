class Banana(Item):
    def __init__(self, count=1, max_count=32, color='green'):
        super().__init__(count, max_count)
        self._color = color
    
    @property
    def color(self):
        return self._color



banana = Banana(color='red')
print(banana.count)
print(banana.color)

# isinstance проверяет принадлежит ли объект классу
print(isinstance(banana, Banana))
print(isinstance(banana, Item))



# множественное наследование
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

class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color
    
    @property
    def color(self):
        return self._color



apple = Apple(False, color='green')
print(apple.count)
print(apple.color)
print(apple.eatable)

Apple.__bases__