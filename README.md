# Tower Lamp Alert (CNC 타워램프 상태 감지·분류·알림)

## 프로젝트 개요
카메라 영상만으로 타워램프 상태(Off/Green/Yellow/Red)를 분류하고, 실시간 알림을 제공하는 프로젝트입니다. CCTV/웹캠/영상파일/RTSP 입력을 지원하며, OpenCV+ML, MobileNetV2, EfficientNetB0, YOLOv8n 기반 자동 ROI 기능을 포함합니다.

## 버전별 주요 기능
아래 표는 각 브랜치(버전)에 구현된 주요 기능을 요약한 것입니다. 상세 내용 및 코드 변경 내역은 브랜치별로 확인할 수 있습니다.

| 브랜치명 | 버전 | 주요 기능 |
|---|---|---|
| version/1.1-opencv-ml | Ver 1.1 | OpenCV 기반 전처리(CLAHE, GaussianBlur, HSV boost) + LightGBM ML 분류, 수동 ROI, 이메일 알림, 가동률 계산 |
| version/1.2-deeplearning-mobilenetv2 | Ver 1.2 | MobileNetV2 이미지 분류, 전이학습/파인튜닝, GUI 연동 |
| version/1.3-deeplearning-efficientNet | Ver 1.3 | EfficientNetB0 분류(빠른 추론), 생산 스케줄 기반 가동률 표기 |
| version/2.0-auto_add_roi_yolo | Ver 2.0 | YOLOv8n 기반 타워램프 객체 자동 탐지(ROI 자동 추가), 분류/알림 통합, 실시간성 강화 |

> 각 버전별 상세 기능 및 코드 변경 내역은 브랜치에서 직접 확인 가능합니다.

## 개발 및 구현 현황
- **MyObjectDetection**: C# WPF 기반 프로토타입(실제 동작 미구현, UI/설계 목적)
- **py_model**: Python 기반 실제 동작 구현(모델 추론, GUI, 알림 등)
	- 주요 기능: 타워램프 상태 분류, ROI 등록/자동추출, 이메일 알림, 가동률 계산, GUI

## 한계 및 가정
- 실제 공장 데이터가 아닌 시뮬레이션 데이터(LED 모듈) 사용
- 4개 클래스(Off, Green, Yellow, Red)
- 다양한 조건에서 직접 촬영한 이미지(3060장)로 학습
- 실제 현장 적용 시 추가 학습/보정 필요

## 디렉토리 구조
```
tower-lamp-alert/
├─ MyObjectDetection/
│  └─ tower_lamp_alart/   # C# WPF 프로토타입
├─ py_model/              # Python 실제 동작 코드
│  ├─ main_window.py      # GUI
│  ├─ detection.py        # 상태 분류
│  ├─ roi_list_widget.py  # ROI 관리
│  ├─ video_capture.py    # 영상 입력
│  ├─ util/               # 이메일 알림 등
│  └─ model/              # 분류/탐지 모델
└─ README.md
```

## 설치 및 실행 방법
```bash
git clone https://github.com/kimhotac/tower-lamp-alert.git
cd tower-lamp-alert
git checkout version/2.0-auto_add_roi_yolo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
# requirements.txt가 있다면
# pip install -r requirements.txt
```

## 실행 예시
```bash
# 웹캠 입력
python py_model/main_window.py --source 0
# 동영상 파일 입력
python py_model/main_window.py --source path/to/video.mp4
# RTSP 입력
python py_model/main_window.py --source "rtsp://..."
```

## 주요 기능
- 타워램프 상태 분류(Off/Green/Yellow/Red)
- ROI 등록(수동/자동), 다중 램프 모니터링
- 이메일 알림(쿨다운)
- 가동률 계산
- GUI 제공

## 데이터셋 및 학습 팁
- 클래스: 0_off / 1_green / 2_yellow / 3_red
- 데이터 분할: Train 80% / Val 10% / Test 10% (3060장)
- 전처리: CLAHE, GaussianBlur, HSV boost 등
- ML: LightGBM, Optuna 최적화
- DL: MobileNetV2, EfficientNetB0
- YOLO: YOLOv8n, Optuna 탐색