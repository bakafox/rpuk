import torch, torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image

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
    img[:, :, 0] = img[:, :, 0] * 0 # нормализуем оттенок цвета (хз почему, но это помогает)
    img[:, :, 1] = img[:, :, 1] * 7 # повысим цветность (увеличит число артефактов)
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

model = torchvision.models.resnet50(pretrained=True)
model.eval() # переводим модель из режима тренировки в режим тестирования

layer4 = model.layer4
layer4.register_forward_hook(get_features_map)


transform_pipe = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225]
    #)
])

def get_features_heatmap(img, preset_pred=None):
    with torch.no_grad():
        output = model(transform_pipe(img).unsqueeze(0))
        #print(output.max(1, keepdim=True), preset_pred)
    
    # получаем веса "максимального" предсказания
    if not preset_pred:
        pred = output.max(1, keepdim=True)[1]
    else:
        pred = preset_pred
    fc_weights = model.fc.weight.data.numpy()
    fc_pred_weights = fc_weights[pred]

    # получаем тепловую карту
    heatmap = layer4_features.squeeze(0).cpu().numpy() # удалим Batch Size и преобразуем в numpy
    heatmap = np.transpose(heatmap, (1, 2, 0)) # повернём heatmap из 2048*w*h в w*h*2048 (для удобства)

    # для каждого канала в матрице признаков (наш слой 4),
    # выполняем перемножение со значением fc_pred_weights
    for h in range(heatmap.shape[0]):
        for w in range(heatmap.shape[1]):
            heatmap[h][w] = heatmap[h][w] * fc_pred_weights

    heatmap_total = np.sum(heatmap, axis=2) # суммируем каналы
    heatmap_total = cv2.resize(heatmap_total, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return heatmap_total, pred


while (True):
    print("______")
    try:
        img_no = int(input('Выберите изображение (1-4, пропуск - выход): ')) - 1
        kent_no = int(input('Выберите кента для поиска (1-4, пропуск - выход): ')) - 1
    except ValueError:
        break

    """
    3) Закодировать динозавра с соседнего изображения (использовать output с layer4)
    """
    kent_heatmap, kent_pred = get_features_heatmap(kents[kent_no])
    #plt.imshow(kent_heatmap)
    #plt.show()

    """
    2) 'Закодировать' одно изображение полность с помощью ResNet. (Использовать output с layer4) 
    """
    img_heatmap, _ = get_features_heatmap(imgs[img_no], kent_pred)
    #plt.imshow(img_heatmap)
    #plt.show()

    """
    4) Сравнивая 'фичи' найти динозавра на изобрпжении. Построить тепловую карту
    """
    match = cv2.matchTemplate(cv2.cvtColor(kents[kent_no] * cv2.cvtColor(kent_heatmap, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2GRAY),
                              cv2.cvtColor(imgs[img_no] * cv2.cvtColor(img_heatmap, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2GRAY),
                              cv2.TM_CCOEFF_NORMED)
    kent_found = np.where(match == match.max())

    img_kent_found = imgs[img_no].copy()
    img_kent_found = cv2.rectangle(
        img_kent_found,
        (kent_found[1][0], kent_found[0][0]),
        (kent_found[1][0] + kents[kent_no].shape[1], kent_found[0][0] + kents[kent_no].shape[0]),
        color=(0, 0, 255),
        thickness=5
    )
    plt.subplot(1, 2, 1)
    plt.imshow(img_kent_found)

    match = cv2.resize(match, (img_kent_found.shape[1], img_kent_found.shape[0]))
    match = cv2.rectangle(
        match,
        (kent_found[1][0], kent_found[0][0]),
        (kent_found[1][0] + kents[kent_no].shape[1], kent_found[0][0] + kents[kent_no].shape[0]),
        color=1,
        thickness=5
    )
    plt.subplot(1, 2, 2)
    plt.imshow(match)

    plt.show()
