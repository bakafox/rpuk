import torch, torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

"""
Задание: Найти динозавра
"""
img_paths = [
    'rpuk/Lesson18 (ResNet50)/A_B_C.jpg',
    'rpuk/Lesson18 (ResNet50)/Ab_C.jpg',
    'rpuk/Lesson18 (ResNet50)/abc.jpg',
    'rpuk/Lesson18 (ResNet50)/abc.png'
]

imgs = []
for path in img_paths: # превращаем динозавриков в кровавое месиво :3
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img[:, :, 0] = img[:, :, 0] * 0 # нормализуем оттенок цвета
    img[:, :, 1] = img[:, :, 1] * 5 # повысим цветность (увеличит число артефактов)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    imgs.append(img)

"""
1) Выбрать одинаковых динозавров с разных изображений
"""
kents = [
    imgs[0][370:425, 460:610],
    imgs[1][205:260, 315:450],
    imgs[2][360:460, 630:800],
    imgs[3][485:550, 60:225]
]


layer4_features = None
def get_features_map(module, inputs, output):
    global layer4_features
    layer4_features = output # np.squeeze(output.data.cpu().numpy(), axis=(2, 3))

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.eval() # переводим модель из режима тренировки в режим тестирования

layer4 = model.layer4
layer4.register_forward_hook(get_features_map)


transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


while(True):
    print("______")
    try:
        img_no = int(input('Выберите изображение (1-4, пропуск - выход): ')) - 1
        kent_no = int(input('Выберите кента для поиска (1-4, пропуск - выход): ')) - 1
    except ValueError:
        break

    """
    2) 'Закодировать' одно изображение полность с помощью ResNet. (Использовать output с layer4) 
    """
    # преобразуем картинку в тензор и добавляем измерение Batch Size,
    # затем прогоняем через модель и сохраняем веса и 4-й слой
    img_input = transform_pipe(imgs[img_no]).unsqueeze(0)
    with torch.no_grad():
        img_output = model(img_input)
    img_features = layer4_features

    """
    3) Закодировать динозавра с соседнего изображения (использовать output с layer4)
    """
    # (выполняем всё то же самое для кента)
    kent_input = transform_pipe(kents[kent_no]).unsqueeze(0)
    with torch.no_grad():
        kent_output = model(kent_input)
    kent_features = layer4_features


    """
    4) Сравнивая 'фичи' найти динозавра на изобрпжении. Построить тепловую карту
    """
    match = F.conv2d(img_features, kent_features)
    match = match.squeeze().cpu().numpy() # удалим Batch Size и преобразуем в numpy
    kent_found = np.where(match == match.max())
    kent_found[1][0] *= (imgs[img_no].shape[1] / match.shape[1]) 
    kent_found[0][0] *= (imgs[img_no].shape[0] / match.shape[0]) 
    print(f'Кент найден на координатах ({kent_found[1][0]}, {kent_found[0][0]}).')

    img_kent_found = imgs[img_no].copy()
    img_kent_found = cv2.rectangle(
        img_kent_found,
        (kent_found[1][0] - kents[kent_no].shape[1]//2, kent_found[0][0] - kents[kent_no].shape[0]//2),
        (kent_found[1][0] + kents[kent_no].shape[1]//2, kent_found[0][0] + kents[kent_no].shape[0]//2),
        color=(0, 0, 255),
        thickness=5
    )
    plt.subplot(1, 2, 1)
    plt.imshow(img_kent_found)

    img_match = match.copy()
    img_match = cv2.resize(img_match, (imgs[img_no].shape[1], imgs[img_no].shape[0]), interpolation=cv2.INTER_NEAREST)
    img_match = cv2.rectangle(
        img_match,
        (kent_found[1][0] - kents[kent_no].shape[1]//2, kent_found[0][0] - kents[kent_no].shape[0]//2),
        (kent_found[1][0] + kents[kent_no].shape[1]//2, kent_found[0][0] + kents[kent_no].shape[0]//2),
        color=0,
        thickness=5
    )
    plt.subplot(1, 2, 2)
    plt.imshow(img_match)

    plt.show()
