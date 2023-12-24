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
    average_digits.append(average_digit)


"""
2. Объеденить получившиеся веса в одну модель, которая
на вход принимает картинку, а выдаёт вектор размера 10.
"""
def get_weight_prediction(sample, weight, offset):
    return 1.0/(1.0 + np.exp(-( # сигмоидная функция (ограничение результата в диапазоне 0..1)
        np.dot(weight, sample) + offset # скалярное произведение + смещение
    )))

def predict(sample, weights, offset): # простая сеть из 10 входных и 1 выходного нейрона
    predictions = []
    for j in range(10):
        predictions.append(
            get_weight_prediction(sample, np.transpose(weights[j]), offset)[0][0]
        )
    return encode_label(np.argmax(predictions))


"""
3. Рассчитать точность получившейся модели на тестовом наборе.
"""
def get_accuracy(data, weights, offset):
    correct_predictions = 0

    for i in range(len(data)):
        sample, label = data[i]
        prediction = predict(sample, weights, offset)

        if np.argmax(prediction) == np.argmax(label):
            correct_predictions += 1

    return np.round(((correct_predictions
                     / len(data)) * 100), 2)

print('\nТочность на тренировочном датасете:',
      get_accuracy(train, average_digits, -90), '%')
print('\nТочность на тестовом датасете:',
      get_accuracy(test, average_digits, -90), '%')


"""
4. Визуализировать набор необработанных данных с помощью
алгоритма t-SNE. Взять 30 изображений каждого класса,
каждое изображение перевести в вектор размера (784),
визуализировать полученные вектора с помощью t-SNE.
"""
sample_digits = []
for i in range(10):
    sample_digits.append([])

for j in range(len(train)):
    sample, label = train[j]
    if (len(sample_digits[np.argmax(label)])) < 30:
        sample_digits[np.argmax(label)].append((np.reshape(sample, (784, 1)), label))

    for k in sample_digits: # проверка на заполнение
        if len(k) == 30:
            break

# преобразуем в непрерынвый список изображений для вывода
sample_digits_prepared = []
for k in sample_digits:
    digits_class, labels_class = zip(*k)
    digits_class = np.concatenate(digits_class, axis=1)
    sample_digits_prepared.append(digits_class)

sample_digits_prepared = np.concatenate(sample_digits_prepared, axis=1)

tsne = TSNE(n_components=2)
tsne_scaled = tsne.fit_transform(np.transpose(sample_digits_prepared))

plt.figure(figsize=(6, 6))
for i in range(10):
    plt.scatter(tsne_scaled[i*30 : (i+1)*30, 0],
                tsne_scaled[i*30 : (i+1)*30, 1],
                label=i)
plt.legend()
plt.show()


"""
5. Визуализировать результаты работы вашей модели
(эмбединги) с помощью алгоритма t-SNE. Прогнать
изображения через вашу модель, получившиеся вектора
размера (10) визуализировать с помощью t-SNE.
"""
sample_embeds = []
for i in range(10):
    sample_embeds.append([])

for i in range(10):
    for j in range(30):
        sample, label = sample_digits[i][j]
        prediction = predict(sample, average_digits, -90)
        sample_embeds[i].append(prediction)

embedding_array_model = np.array(sample_embeds).reshape(-1, 10)

tsne_model = TSNE(n_components=2)
tsne_result_embeddings_model = tsne_model.fit_transform(embedding_array_model)

plt.figure(figsize=(6, 6))
for i in range(10):
    # слегка рандомизиаруем координаты, чтобы точки реже накладывались друг на друга
    random_offsets = np.random.randint(-30, 30, size=(30, 2))
    plt.scatter(tsne_result_embeddings_model[i*30 : (i+1)*30, 0] + random_offsets[:, 0],
                tsne_result_embeddings_model[i*30 : (i+1)*30, 1] + random_offsets[:, 1],
                label=str(i))
plt.legend()
plt.show()
