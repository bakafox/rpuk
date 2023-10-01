from PIL import Image
import numpy as np

image_path = input('Введите путь изображения (пропуск - BEER.bmp): ')
if image_path == '':
    image_path = 'rpuk/Lesson4/Задания/2: BEER.bmp'

image = Image.open(image_path)
colors = np.array(image.getdata())
colors_unique = np.unique(colors, axis=0)
print('Цвета: \n', colors_unique)
print('Кол-во цветов: ', len(colors_unique))
