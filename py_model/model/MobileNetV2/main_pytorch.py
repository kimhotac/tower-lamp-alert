

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
def main():
# --- 1. 기본 설정값 지정 ---
# 데이터가 있는 폴더의 경로
    data_dir = 'C:/Users/JH/Desktop/web/label'

    # 모델 학습을 위한 파라미터 설정
    batch_size = 32
    img_height = 180
    img_width = 180
    num_epochs_initial = 25  # 초기 학습 에포크
    num_epochs_finetune = 25 # 미세 조정 에포크
    lr_initial = 1e-3
    lr_finetune = 1e-5

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 데이터셋 준비 및 전처리 ---
    print("데이터셋 로딩 및 전처리를 시작합니다...")

    # 데이터 증강 및 정규화를 위한 변환 정의
    # 훈련 데이터용 변환
    data_transforms_train = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 검증 및 테스트 데이터용 변환
    data_transforms_val = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ImageFolder를 사용하여 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms_train)
    # 검증/테스트용 데이터셋은 별도의 변환을 적용하기 위해 다시 로드
    full_dataset_val = datasets.ImageFolder(data_dir, transform=data_transforms_val)


    # 데이터셋 분할 (80% 훈련, 10% 검증, 10% 테스트)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # random_split을 사용하기 전에 generator를 고정하여 재현성을 보장
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # 검증/테스트 데이터셋에 올바른 변환(증강X)을 적용
    val_dataset.dataset = full_dataset_val
    test_dataset.dataset = full_dataset_val


    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"감지된 클래스: {class_names}")

    # --- 3. 모델 구축 (전이 학습) ---
    print("\n전이 학습 모델(MobileNetV2)을 구축합니다...")

    # 사전 학습된 MobileNetV2 모델 로드
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # 초기에는 모든 가중치를 고정
    for param in model.parameters():
        param.requires_grad = False

    # 분류기(classifier)를 새로운 것으로 교체
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.last_channel, num_classes),
    )

    model = model.to(device)

    # --- 4. 모델 학습 함수 정의 ---
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        best_model_wts = model.state_dict()

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            # 각 에포크는 훈련과 검증 단계를 가짐
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 모델을 훈련 모드로 설정
                    dataloader = train_loader
                else:
                    model.eval()   # 모델을 평가 모드로 설정
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # 데이터를 반복
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 옵티마이저의 기울기 초기화
                    optimizer.zero_grad()

                    # 순전파
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 훈련 단계에서만 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                    # 최고 성능 모델 저장
                    if epoch_acc > best_val_acc:
                        best_val_acc = epoch_acc
                        best_model_wts = model.state_dict()
                        print(f'Best validation accuracy: {best_val_acc:.4f}')


        print(f'Best val Acc: {best_val_acc:4f}')
        model.load_state_dict(best_model_wts)
        return model, history

    # --- 5. 초기 학습 (분류기만) ---
    print("\n모델의 분류기만 초기 학습을 시작합니다...")
    criterion = nn.CrossEntropyLoss()
    # 새로 추가한 분류기의 파라미터만 학습하도록 옵티마이저 설정
    optimizer_initial = optim.Adam(model.classifier.parameters(), lr=lr_initial)

    model, history_initial = train_model(model, train_loader, val_loader, criterion, optimizer_initial, num_epochs=num_epochs_initial)

    # --- 6. 미세 조정 (Fine-Tuning) ---
    print("\n모델 미세 조정을 시작합니다...")
    # 이제 전체 모델의 가중치를 학습 가능하게 설정
    for param in model.parameters():
        param.requires_grad = True

    # 매우 낮은 학습률로 다시 컴파일
    optimizer_finetune = optim.Adam(model.parameters(), lr=lr_finetune)

    model, history_finetune = train_model(model, train_loader, val_loader, criterion, optimizer_finetune, num_epochs=num_epochs_finetune)

    # --- 7. 최종 평가 및 결과 시각화 ---
    def evaluate_model(model, test_loader):
        model.eval()
        running_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = running_corrects.double() / len(test_loader.dataset)
        print(f'\n최종 테스트 정확도: {accuracy:.4f}')
        return all_labels, all_preds

    true_labels, predictions = evaluate_model(model, test_loader)

    # 혼동 행렬(Confusion Matrix) 시각화
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_pytorch.png')
    print("\n혼동 행렬을 'confusion_matrix_pytorch.png' 파일로 저장했습니다.")
    plt.show()


    # 학습 과정 시각화
    def plot_history(history_initial, history_finetune, num_epochs_initial):
        acc = history_initial['train_acc'] + history_finetune['train_acc']
        val_acc = history_initial['val_acc'] + history_finetune['val_acc']
        loss = history_initial['train_loss'] + history_finetune['train_loss']
        val_loss = history_initial['val_loss'] + history_finetune['val_loss']

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.axvline(x=num_epochs_initial - 1, color='r', linestyle='--', label='Fine-tuning start')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.axvline(x=num_epochs_initial - 1, color='r', linestyle='--', label='Fine-tuning start')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.suptitle("Model Training History (PyTorch)")
        plt.savefig('training_history_pytorch.png')
        print("학습 과정 그래프를 'training_history_pytorch.png' 파일로 저장했습니다.")
        plt.show()

    plot_history(history_initial, history_finetune, num_epochs_initial)


    # --- 8. 모델 저장 ---
    model_save_path = 'mobilenetv2_pytorch.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"\n학습된 모델을 '{model_save_path}' 파일로 저장했습니다.")

    print("\n모든 작업이 완료되었습니다.")
if __name__ == '__main__':
    main()