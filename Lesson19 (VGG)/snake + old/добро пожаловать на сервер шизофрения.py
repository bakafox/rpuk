"""
https://github.com/dd2-42/Unloading-the-Cognitive-Overload-in-Machine-Learning/blob/main/Computer_Vision/XAI/CNN_Visualizations/CNN_Visualizations_01.ipynb
"""

"""
Загружаем предтренированную модель
"""
import torch
from torchvision.models import vgg16

weight_vgg16_imagenet_path = "./weights/vgg16-397923af.pth"

model_vgg16 = vgg16(weights=None)
model_vgg16.load_state_dict(torch.load(weight_vgg16_imagenet_path))


"""
Объявление слоёв
"""
layer_num = 10
layer = model_vgg16.features[layer_num]
print(layer)

"""
Настройка хука
"""
layer_output = None
def hook_fn(module, input, output):
    global layer_output
    layer_output = output

handle = layer.register_forward_hook(hook_fn)


"""
Входные данные (случаный шум)
"""
import torch
img_noise = torch.randn(1, 3, 224, 224)

model_vgg16.eval() # Forward pass the image through the model

with torch.inference_mode():
    preds = model_vgg16(img_noise)

#layer_output.shape # torch.Size([1, 256, 56, 56])
layer_output = layer_output.squeeze()
#layer_output.shape # torch.Size([256, 56, 56])









"""
model = vgg16(pretrained=True)

# Сохранение весов модели
torch.save(model_мпп16.state_dict(), 'vgg16_weights.pth')
# model_vgg16.load_state_dict(torch.load(weight_vgg16_imagenet_path))
model.to(device)

Использовать 'optimize_img'
"""
