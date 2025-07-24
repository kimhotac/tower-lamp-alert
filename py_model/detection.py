import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

# 모델과 관련 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"model\\efficientnet_led_classifier\\efficientnet_led_classifier.pth"

# EfficientNet-B0 모델 불러오기, 클래스 4개로 맞춤
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # 4개의 클래스

# 모델 가중치 불러오기
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 전처리 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # EfficientNet 입력 크기
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 평균
                         [0.229, 0.224, 0.225])  # ImageNet 표준편차
])

# 클래스 이름 정의 (4개)
class_names = ['off', 'green', 'yellow', 'red']

# 감지 함수
def detect(roi_img):
    if roi_img is None or roi_img.size == 0:
        return "Invalid ROI"

    try:
        img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            return class_names[predicted.item()]
    except Exception as e:
        return f"Error: {str(e)}"