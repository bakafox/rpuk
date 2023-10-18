import os
import matplotlib.pyplot as plt
import cv2

imgpath = input("Введите путь до папки изображений (пропуск - по умолчанию): ")
if imgpath == "": imgpath = "rpuk/Lesson6 (cv2)/Задания/nails_segmentation/images"
lblpath = input("Введите путь до папки масок (пропуск - по умолчанию): ")
if lblpath == "": lblpath = "rpuk/Lesson6 (cv2)/Задания/nails_segmentation/labels"


for imgname in os.listdir(imgpath):
    plt.figure(figsize=(14, 4))

    img = plt.imread(os.path.join(imgpath, imgname))
    plt.subplot(1, 3, 1)
    plt.imshow(img)

    lbl = plt.imread(os.path.join(lblpath, imgname))
    plt.subplot(1, 3, 2)
    plt.imshow(lbl)
    
    image_out = img
    image_out = cv2.drawContours(
            image_out,
            cv2.findContours(
                cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
                )[0],
        -1, (0, 255, 0), 2)

    plt.subplot(1, 3, 3)
    plt.imshow(image_out, cmap='gray')

    plt.show()
    key = cv2.waitKey(0)
    plt.close()