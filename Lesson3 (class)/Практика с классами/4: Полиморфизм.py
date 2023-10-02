class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color
    
    @property
    def color(self):
        return self._color
        
    @property
    def eatable(self):
        """
        Переопределённая функция класса Food. Добавление проверки на спелость
        """
        return super().eatable and self._ripe



apple = Apple(False, color='green')
print(apple.count)
print(apple.color)
print(apple.eatable)