# см. также: Копия блокнота "train-yolov8-object-detection-on-custom-dataset.ipynb"
import ultralytics # pip install ultralytics
from roboflow import Roboflow # pip install roboflow
from torch.utils.data import DataLoader # pip install torch


# скачиваем датасет с RoboFlow
rf = Roboflow(api_key='IJmZ9Rj7mXzHlNQHLSSX')
project = rf.workspace('ppe-buxwb').project('edge-academy-ppe')
version = project.version(1)
dataset = version.download('yolov8')

# загружаем модель с натренированными весами
#model = ultralytics.YOLO('yolov8n_trained.pt')
model = ultralytics.YOLO('yolov8n_trained/weights/best.pt')


# тестируем модель
# val = model.val(
#     data=f'{dataset.location}/data.yaml',
#     split='val',
#     imgsz=640,
#     batch=16,
#     name='yolov8n_val'
# )

predict = model.predict(

)

# демонстрации результатов
