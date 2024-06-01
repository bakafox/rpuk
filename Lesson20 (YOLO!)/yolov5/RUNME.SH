#!/bin/bash

# Задание. 
# ✔️ 1. запустить йоло v5 (или v8) 
# ✔️ 2. выбрать изображение, на котором встречаются объекты разного масштаба 
# ✔️ 3. измерить время инференса для одного изображения
# ✔️ 4. разобраться в post-process-е (ultralytics yolov5) 
# ✔️ 5. выкинуть из постпроцесса обработку предсказаний одного из масштабов (например, чтобы постпроцесс не учитывал мелкие объекты)
# ______
# ✔️ 6. Сохранить выход с нейросети в виде numpy массива
# ✔️ 7. Используя постпроцесс rknn/yolov5, получить изображение с ббоксами

rm -rf __out
rm -rf __postprocess
rm data_0.pt
rm data_1.pt
rm data_2.pt

read -p "KAMEPY BKJIO4AEM? [Y/N]: " yn
if [[ $yn =~ ^[Yy]$ ]]; then
    python3 detect.py --weights yolov5s.pt --source="0" --project="" --name="__out" --exist-ok \
                      --iou-thres=0.7 --conf-thres=0.5
else
    python3 detect.py --weights yolov5s.pt --source="__img/vienna.jpg" --project="" --name="__out" --exist-ok \
                      --iou-thres=0.7 --conf-thres=0.5 #--save-txt --save-conf
fi

python3 post_process.py
