import cv2
import numpy as np
from matplotlib import pyplot as plt


mainimg_path = input("Введите путь до изображения для выполнения поиска, \nили пропустите ввод для выхода: ")
if mainimg_path == "":
    exit(1)
mainimg = cv2.imread(mainimg_path)

dataset_img = []
while True:
    img_path = input("Введите путь до изображения, которое необходимо найти, \nили пропустите ввод для начала поиска: ")
    if img_path == "":
        break

    img = cv2.imread(img_path)
    for i in range(4):
        dataset_img.append(img)
        dataset_img.append(cv2.flip(img, i%2+1))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


ghosts_found = 0
img_no = 0
while img_no < len(dataset_img):
    """
    Поиск ключевых точек с использованием алгоритма SIFT,
    где keypoints -- уникальные ("ключевые") точки изображения
    и descriptors -- запись этой точки в форме n-мерного вектора
    """
    sift = cv2.SIFT_create()

    mainimg_keypoints, mainimg_descriptors = sift.detectAndCompute(mainimg, None)

    dataset_keypoints = []
    dataset_descriptors = []
    for img in dataset_img:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        dataset_keypoints.append(keypoints)
        dataset_descriptors.append(descriptors)

    """
    Сравниваем признаки, используя Brute Force Matching,
    и выводим найденные совпадения ключевых точек
    для каждого изображения из датасета
    """
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    dataset_matches = []

    matches = bf.match(mainimg_descriptors, dataset_descriptors[img_no])
    matches = sorted(matches, key = lambda x:x.distance)

    dataset_matches.append(matches)

    test = cv2.drawMatches(
        mainimg, mainimg_keypoints,
        dataset_img[img_no], dataset_keypoints[img_no],
        matches[:20], None, flags=2)

    """
    Вычисляем среднее расстояние первых 20 совпадений.
    Если оно выше порогового значения, выводим найденное
    изображение, считаем и закрашиваем его на картинке.
    """
    mean_dist = np.mean([match.distance for match in matches[:10]])
    print(mean_dist)

    if mean_dist < 200:
        ghosts_found += 1

        points = np.array([mainimg_keypoints[match.queryIdx].pt for match in matches[:10]], np.int32)
        cv2.polylines(mainimg, [points], True, (255, 0, 0), 200)

        plt.imshow(test)
        plt.show()

    else:
        """
        Перепроверяем, чтобы убедиться, что
        таких же искомых изображений на картинке
        больше не осталось.
        """
        img_no += 1


print("=== Призраков найдено:", ghosts_found)
plt.imshow(mainimg)
plt.show()