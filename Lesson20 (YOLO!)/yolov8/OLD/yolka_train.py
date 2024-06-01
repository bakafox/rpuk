# см. также: Копия блокнота "train-yolov8-object-detection-on-custom-dataset.ipynb"
import ultralytics # pip install ultralytics
from roboflow import Roboflow # pip install roboflow
from torch.utils.data import DataLoader # pip install torch

import matplotlib as plt
import cv2
import os


# скачиваем датасет с RoboFlow
rf = Roboflow(api_key='IJmZ9Rj7mXzHlNQHLSSX') # ключ уже месяц как разделегирован,
                                              # можете даже не пытаться =)
project = rf.workspace('ppe-buxwb').project('edge-academy-ppe')
version = project.version(1)
dataset = version.download('yolov8')

# загружаем чистую ёлу
model = ultralytics.YOLO('yolov8n.pt')


# тренируем модель
results = model.train(
   data=f'{dataset.location}/data.yaml',
   split='train',
   imgsz=640,
   epochs=1, # поменять на побольше
   batch=16,
   name='yolov8n_trained'
)

# сохраняем веса
#model.save('yolov8n_trained.pt')
# (вместо этого мы используем веса из yolov8n_trained/weights/best.pt)

# демонстрации результатов
img = cv2.imread('runs/train/yolov8n_trained/confusion_matrix.png')
plt.imshow(img)
img = cv2.imread('runs/train/yolov8n_trained/confusion_matrix.png')
plt.imshow(img)
