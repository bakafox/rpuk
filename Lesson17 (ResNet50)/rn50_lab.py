import torch, torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
Задание: Сделать визуализацию работы Resnet50 
"""

"""
1) Установить torch, torchvision, скачать Resnet (pretrained=True)
"""
model = torchvision.models.resnet50(pretrained=True)
model.eval() # переводим модель из режима тренировки в режим тестирования

"""
2) Написать функцию get_features_map() (установить hook).
Которая "возвращает" результаты работы слоя layer4
(находится перед слоем avgpool)
"""
layer4_features = None
def get_features_map(module, inputs, output):
    global layer4_features
    layer4_features = output # np.squeeze(output.data.cpu().numpy(), axis=(2, 3))

# forward_hook -- перехватчик выходных данных
# переднего прохода, в данном случае слоя layer4
model.layer4.register_forward_hook(get_features_map)


"""
Трансформация изображения в тензор для обработки ResNet-ом
"""
transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def visualise_with_heatmap(img_path, pred_len):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_transformed = transform_pipe(img).unsqueeze(0) # установим Batch Size (для ResNet) =1
    with torch.no_grad():
        output = model(img_transformed)

    # получение и вывод результатов предсказания
    pred_top = output.topk(pred_len, sorted=True)[1].numpy()[0]
    pred_top_titles = []
    with open('rpuk/Lesson17 (ResNet50)/classes.txt') as classes_f:
        titles = classes_f.readlines()
        for pred in range(pred_len):
            pred_top_titles.append(titles[pred_top[pred]].strip())

    """
    3) Достать из модели матрицу весов "W" последнего
    (полносвязного) слоя (fc).
    """
    for i in range(pred_len):
        pred = pred_top[i]
        pred_title = pred_top_titles[i]

        fc_weights = model.fc.weight.data.numpy()
        fc_pred_weights = fc_weights[pred]

        """
        4) Сложить "каналы" карты признаков, как показаано в статье
        https://alexisbcook.github.io/posts/global-average-pooling-layers-for-object-localization/
        """
        heatmap = layer4_features.squeeze(0).cpu().numpy() # удалим Batch Size и преобразуем в numpy
        heatmap = np.transpose(heatmap, (1, 2, 0)) # повернём heatmap из 2048*w*h в w*h*2048 (для удобства)

        # для каждого канала в матрице признаков (наш слой 4),
        # выполняем перемножение со значением fc_pred_weights
        for h in range(heatmap.shape[0]):
            for w in range(heatmap.shape[1]):
                heatmap[h][w] = heatmap[h][w] * fc_pred_weights

        heatmap_total = np.sum(heatmap, axis=2) # суммируем каналы
        heatmap_total = cv2.resize(heatmap_total, (img.shape[1], img.shape[0]))

        # собственно, показ визуализации
        plt.title(f'№{str(i+1)} : {pred_title}')
        plt.imshow(img)
        plt.imshow(heatmap_total, alpha=0.4, cmap='jet')

        plt.show()


pred_len = min(max(int(input('Введите длину выводимого топа результатов (1-1000): ')), 1), 1000)
img_path = input('Введите путь к картинке (пропуск - исп. образцы): ')
if not img_path:
    visualise_with_heatmap('rpuk/Lesson17 (ResNet50)/cat.jpeg', pred_len)
    visualise_with_heatmap('rpuk/Lesson17 (ResNet50)/cat.jpg', pred_len)
    visualise_with_heatmap('rpuk/Lesson17 (ResNet50)/cat.png', pred_len)
else:
    visualise_with_heatmap(img_path, pred_len)
