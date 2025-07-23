import os
import csv
import cv2
from model.opencv.led_detector import LedDetector # 우리가 만든 클래스를 import

# --- 설정 ---
# 라벨링된 이미지가 있는 기본 폴더
LABEL_BASE_DIR = 'model/opencv/label'
# 생성될 CSV 파일 이름
CSV_FILENAME = 'led_data.csv'
# 모든 이미지를 통일할 크기 (가로, 세로)
IMAGE_SIZE = (100, 100)

# --- 메인 로직 ---
def create_csv_dataset():
    """
    LABEL_BASE_DIR 안의 모든 이미지를 처리하여 특징을 추출하고,
    CSV_FILENAME으로 저장합니다.
    """
    # 1. OpenCV 특징 추출기 초기화
    detector = LedDetector(use_preprocessing=False)  # 여기서 리사이징하므로 비활성화
    
    # 2. CSV 파일에 기록할 헤더(컬럼명) 정의
    # extract_features_for_ml가 반환하는 키 + 파일명, 라벨
    header = [
        'filename', 'label', 'opencv_decision', 'count', 
        'hue', 'value', 'saturation', 'area'
    ]
    
    # 모든 이미지의 특징을 저장할 리스트
    all_rows = []

    print("데이터셋 생성을 시작합니다...")
    
    # 3. 'label' 폴더 안의 각 하위 폴더(red, green 등)를 순회
    for label_name in os.listdir(LABEL_BASE_DIR):
        label_dir = os.path.join(LABEL_BASE_DIR, label_name)
        
        # 폴더가 아니면 건너뜀
        if not os.path.isdir(label_dir):
            continue
            
        print(f"'{label_name}' 상태의 이미지를 처리 중...")
        
        # 4. 각 하위 폴더 안의 모든 이미지 파일을 순회
        for filename in os.listdir(label_dir):
            # jpg, png 파일만 대상으로 함
            if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
                continue

            image_path = os.path.join(label_dir, filename)
            
            # 이미지를 읽고 리사이징
            image = cv2.imread(image_path)
            if image is None:
                print(f"  - 경고: {filename}을 읽을 수 없습니다.")
                continue
            
            resized_image = cv2.resize(image, IMAGE_SIZE)
            
            # OpenCV로 특징 추출
            features = detector.extract_features_for_ml(resized_image)
            
            # 파일명과 실제 정답(라벨) 정보 추가
            features['filename'] = filename
            features['label'] = label_name
            
            # 리스트에 최종 데이터 추가
            all_rows.append(features)
            print(f"  - {filename}: {features['opencv_decision']} (특징 추출 완료)")

    # 5. 수집된 모든 데이터를 CSV 파일에 한 번에 저장
    if not all_rows:
        print("처리할 이미지가 없습니다. 'label' 폴더 구조를 확인하세요.")
        return

    print(f"\n총 {len(all_rows)}개의 이미지 데이터를 CSV 파일로 저장합니다...")
    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()  # 헤더 쓰기
        writer.writerows(all_rows) # 모든 데이터 쓰기

    print(f"성공적으로 '{CSV_FILENAME}' 파일을 생성했습니다.")
    
    # 간단한 통계 출력
    print("\n=== 데이터셋 통계 ===")
    label_counts = {}
    for row in all_rows:
        label = row['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"{label}: {count}개")


if __name__ == '__main__':
    create_csv_dataset()
