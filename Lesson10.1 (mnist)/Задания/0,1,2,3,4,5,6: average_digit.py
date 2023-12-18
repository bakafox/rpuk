import numpy as np
import matplotlib.pyplot as plt
import torchvision # только для извлечения данных
from sklearn.manifold import TSNE


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))])

train_raw = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(), download=True)

test_raw = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(), download=True)


def encode_label(j):
    # преобразует label в вектор из 10 элементов, где j-тый элемент = 1
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784, 1)) for x in data]
    labels = [encode_label(y[1]) for y in data]
    return zip(features, labels)

train = list(shape_data(train_raw))
test = list(shape_data(test_raw))


"""
1. Рассчитать average_digit (матрицу весов)
для каждой цифры от 0 до 9, по аналогии с (avg_eight).
"""
average_digits = [] # будет весами для каждой цифры
for i in range(10):
    filtered_data = [x[0] for x in train if np.argmax(x[1]) == i]
    average_digit = np.average(np.asarray(filtered_data), axis=0)
    #img = np.reshape(average_digit, (28, 28))
    #plt.imshow(img)
    #plt.show()
    average_digits.append(average_digit)


"""
2. Объеденить получившиеся веса в одну модель, которая
на вход принимает картинку, а выдаёт вектор размера 10.
"""
def get_weight_prediction(sample, weight, offset):
    return 1.0/(1.0 + np.exp(-( # сигмоидная функция (ограничение результата в диапазоне 0..1)
        np.dot(weight, sample) + offset # скалярное произведение + смещение
    )))
#x_2 = train[2][0] # 4-ка
#x_17 = train[17][0] # 8-ка
#W = np.transpose(average_digits[7])
#print(get_weight_prediction(x_2, W, -30))
#print(get_weight_prediction(x_17, W, -30))

def predict(sample, weights, offset): # простая сеть из 10 входных и 1 выходного нейрона
    predictions = []
    for j in range(10):
        predictions.append(
            get_weight_prediction(
                sample,
                np.transpose(weights[j]),
                offset
        )[0][0])
    
    return encode_label(np.argmax(predictions))
#print(predict(x_2, average_digits, -30))
#print(predict(x_17, average_digits, -30))


"""
3. Рассчитать точность получившейся модели на тестовом наборе.
"""
def get_accuracy(data, weights, offset):
    correct = 0

    for i in range(len(data)):
        sample, label = data[i]
        prediction = predict(sample, weights, offset)

        if np.argmax(prediction) == np.argmax(label):
            correct += 1
        #else:
            #print(prediction, np.argmax(label))

    return np.round(((correct
                     / len(data)) * 100), 2)

print('Точность на тренировочном датасете:',
      get_accuracy(train, average_digits, -90), '%')
print('Точность на тестовом датасете:',
      get_accuracy(test, average_digits, -90), '%')


"""
4. Визуализировать набор необработанных данных с помощью
алгоритма t-SNE. Взять 30 изображений каждого класса,
каждое изображение перевести в вектор размера (784),
визуализировать полученные вектора с помощью t-SNE.
"""

# work in progress! полурабочий код в коммит не включён.

"""
5. Визуализировать результаты работы вашей модели
(эмбединги) с помощью алгоритма t-SNE. Прогнать
изображения через вашу модель, получившиеся вектора
размера (10) визуализировать с помощью t-SNE.
"""

# пока не успеваю, через часик отправлю коммит с визуализациями :)