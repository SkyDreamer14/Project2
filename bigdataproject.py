import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from torchvision.models import efficientnet_b0

# 시드 고정 (재현 가능한 결과를 위해)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 시드 고정

# GPU 설정 및 확인
available_gpus = torch.cuda.device_count()
print(f"사용 가능한 GPU 개수: {available_gpus}")

if available_gpus >= 2:
    # 처음 2개의 GPU 사용
    gpu_ids = [0, 1]
    print(f"사용할 GPU: {gpu_ids}")
    for gpu_id in gpu_ids:
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU {gpu_id} 메모리 사용량:")
        print(f"할당된 메모리: {torch.cuda.memory_allocated(gpu_id)/1024**2:.2f} MB")
        print(f"캐시된 메모리: {torch.cuda.memory_reserved(gpu_id)/1024**2:.2f} MB")
else:
    print("2개 이상의 GPU가 필요합니다.")
    exit()

# 데이터 경로 설정
train_dir = "./data/train"  # 훈련 데이터 경로
val_dir = "./data/validation" # 검증 데이터 경로

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 10
learning_rate = 0.003
num_classes = 2  # 생성된 이미지와 real 두 가지 클래스

# 데이터 전처리: 이미지 크기 조정 및 정규화
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 입력 크기
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 데이터셋의 평균 및 표준편차로 정규화
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터셋 로드
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

# DataLoader 설정
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
}

"""
# ResNet 모델 불러오기 (pre-trained)
model = models.resnet50(pretrained=True)

# 마지막 레이어를 재설정하여 2개의 클래스를 분류할 수 있게 설정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 모델을 GPU로 옮기기
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
"""

model = efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# DataParallel을 사용하여 멀티 GPU 설정
if available_gpus >= 2:
    model = nn.DataParallel(model, device_ids=gpu_ids)
    print("DataParallel 모드로 실행 중")

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증 함수
def train_model_with_history(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 epoch에서 훈련 및 검증 단계 실행
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 훈련 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터에 대한 반복문
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 옵티마이저 초기화
                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파 및 최적화 (훈련 단계에서만)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 메모리 해제
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 기록 저장
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            # 최적의 모델을 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if isinstance(model, nn.DataParallel):
                    best_model_wts = model.module.state_dict()
                else:
                    best_model_wts = model.state_dict()

        # 에포크가 끝날 때마다 메모리 정리
        torch.cuda.empty_cache()

    print(f'Best val Acc: {best_acc:.4f}')

    # 최적의 가중치로 모델 로드
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(best_model_wts)
    else:
        model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# 학습 및 검증에 대한 손실과 정확도 그래프 그리기
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, num_epochs):
    epochs_range = range(1, num_epochs+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('loss_accuracy.png')


# 모델 학습
model2, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model_with_history(
    model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

torch.save(model2.state_dict(), 'best_resnet_model.pth')
plot_loss_accuracy(train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs)



# email: mj_lee@korea.ac.kr
# if you have any questions please contact me by email.
