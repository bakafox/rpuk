m = __import__("1,2,3: Работа с классами еды")


class Inventory:
    def __init__(self, slots=9):
        self._slots = slots
        self._list_of_items = [None] * self._slots

    def _check_slot_index(self, index):
        if index < 0 or index >= self._slots:
            print(f'Неверный номер слота (доступно: от 0 до {self._slots -1}).')
            return False
        else:
            return True

    def get_items_list(self):
        return self._list_of_items

    def get_item(self, index):
        if self._check_slot_index(index):
            return self._list_of_items[index]

    def set_item(self, index, value):
        if self._check_slot_index(index):
            self._list_of_items[index] = value

    def eat_item(self, index):
        if self._check_slot_index(index):
            curr_item = self.get_item(index)
            if curr_item == None:
                print('В этом слоте нет предмета.')

            elif isinstance(curr_item, m.Food):
                if (curr_item.eatable):
                    curr_item()
                    if curr_item.count <= 0:
                        self.set_item(index, None)
                else:
                    print('Этот продукт несъедобен.')

            else:
                print('Этот предмет нельзя съесть!')





test = Inventory()
print(test._list_of_items)
print(test.get_item(0))

test.set_item(0, m.Apple(True, 3))
print(test.get_item(0).count)
print(test.get_item(10))

test.set_item(6, m.PotatoChipsPackages(1, 14))
print(test.get_items_list())

test.eat_item(2)
test.eat_item(13)
test.eat_item(0)
print(test.get_item(0).count)

test.eat_item(0)
print(test.get_items_list())
test.eat_item(0)
print(test.get_items_list())

test.eat_item(6)
print(test.get_item(6).count)
test.eat_item(6)
test.eat_item(6)
print(test.get_items_list())

test.set_item(6, m.Lemon(True, 5))
test.set_item(7, m.Lemon(True, 8))
test.set_item(8, m.Brick(16))
test.eat_item(6)
test.eat_item(7)
test.eat_item(8)
print(test.get_items_list())