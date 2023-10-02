# Магические методы python
class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color
    
    @property
    def color(self):
        return self._color
        
    @property
    def eatable(self):
        return super().eatable and self._ripe
    
    def __call__(self):
        """ Вызов как функции """
        if self.eatable:
            new_count = max(self.count - 1, 0)
            self.update_count(new_count)            
    
    def __str__(self):
        """ Вызов как строки """
        return f'Stack of {self.count} {self.color} apples' 
            
    def __len__(self):
        """ Получение длины объекта """
        return self.count
    


apple = Apple(True, count=8, color='red')
print(len(apple))
apple()
print(len(apple))
print(apple)