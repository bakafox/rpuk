import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Получение и загрузка изображений
"""
mainimg_path = input("Введите путь до изображения для выполнения поиска: ")
mainimg = cv2.imread(mainimg_path)

dataset_img = []
while True:
    img_path = input("Введите путь до изображения, которое необходимо найти, \nили пропустите ввод для начала поиска: ")
    if img_path == "":
        break
    dataset_img.append(cv2.imread(img_path))


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
for i in range(len(dataset_img)):
    matches = bf.match(mainimg_descriptors, dataset_descriptors[i])
    matches = sorted(matches, key = lambda x:x.distance)
    dataset_matches.append(matches)

    test = cv2.drawMatches(
        mainimg, mainimg_keypoints,
        dataset_img[i], dataset_keypoints[i],
        matches[:20], None, flags=2)
    plt.imshow(test)
    plt.show()