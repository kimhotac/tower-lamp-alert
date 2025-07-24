import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms

# 모델 설정
MODEL_PATH = r"C:\Users\JH\Documents\GitHub\tower-lamp-alert\py_model\model\MobileNetV2\mobilenetv2_pytorch.pth"
class_labels = ['off', 'green', 'yellow', 'red']
num_classes = len(class_labels)
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 전처리 (정규화 포함)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 감지 함수
def detect(roi_image):
    try:
        if roi_image is None or roi_image.size == 0:
            return "Invalid ROI"

        rgb_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            class_index = torch.argmax(outputs, dim=1).item()
            return class_labels[class_index]
    except Exception as e:
        print(f"[ERROR] detect() failed: {e}")
        return "error"
