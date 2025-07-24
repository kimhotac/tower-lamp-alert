import os
import cv2
import numpy as np
import joblib

def detect(roi_image):
    """
    입력: 사진 프레임 하나 (numpy array)
    출력: LED 상태 ('off', 'green', 'yellow', 'red')
    """
    if roi_image is None:
        return 'off'
    
    # 1. 140개 특징 추출 (4가지 전처리 병렬)
    features = extract_140_features(roi_image)
    
    # 2. 모델 로드 및 예측 (필수)
    model = load_lgbm_model()
    if model is None:
        raise RuntimeError("❌ ML 모델을 로드할 수 없습니다. 모델 파일이 존재하는지 확인하세요.")
    
    prediction = model.predict([features])[0]
    label_map = {0: 'off', 1: 'green', 2: 'yellow', 3: 'red'}
    return label_map.get(prediction, 'off')


def extract_140_features(image):
    """
    노트북의 apply_parallel_preprocessing과 100% 동일한 4가지 전처리 특징 추출
    순서: 원본 → CLAHE → 블러 → HSV (각 35개씩, 총 140개)
    """
    all_features = []
    
    # 1. 원본 (리사이징만) - 35개 특징
    all_features.extend(extract_35_features(image))
    
    # 2. CLAHE 전처리 (원본 이미지에 적용) - 35개 특징  
    clahe_img = apply_clahe(image)  # 원본 이미지에 적용
    all_features.extend(extract_35_features(clahe_img))
    
    # 3. 가우시안 블러 전처리 (원본 이미지에 적용) - 35개 특징
    blur_img = apply_gaussian_blur(image)  # 원본 이미지에 적용
    all_features.extend(extract_35_features(blur_img))
    
    # 4. HSV 부스트 전처리 (원본 이미지에 적용) - 35개 특징
    hsv_img = apply_hsv_boost(image, delta=10)  # 원본 이미지에 적용
    all_features.extend(extract_35_features(hsv_img))
    
    return all_features


def extract_35_features(image):
    """
    노트북의 extract_features와 100% 동일한 35개 특징 추출
    순서 절대 변경 금지 - CSV 생성과 완전히 일치해야 함
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    features = []
    
    # === 기존 14개 특징 (호환성 유지) ===
    # 밝기 관련 특징 (3개)
    avg_brightness = np.mean(gray)
    max_brightness = np.max(gray)
    std_brightness = np.std(gray)
    features.extend([avg_brightness, max_brightness, std_brightness])

    # RGB 평균값 (3개)
    avg_b, avg_g, avg_r = cv2.mean(image)[:3]
    features.extend([avg_r, avg_g, avg_b])

    # 색상 비율 계산 (3개)
    def mask_ratio(hsv_img, lower, upper):
        mask = cv2.inRange(hsv_img, lower, upper)
        return np.count_nonzero(mask) / (mask.size + 1e-6)

    green_ratio = mask_ratio(hsv, (40, 50, 50), (85, 255, 255))
    yellow_ratio = mask_ratio(hsv, (20, 50, 50), (35, 255, 255))
    red_ratio = (
        mask_ratio(hsv, (0, 50, 50), (10, 255, 255)) +
        mask_ratio(hsv, (160, 50, 50), (180, 255, 255))
    )
    features.extend([green_ratio, yellow_ratio, red_ratio])

    # 엣지 관련 특징 (2개)
    edges = cv2.Canny(gray, 100, 200)
    edge_count = np.count_nonzero(edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = sum(cv2.contourArea(c) for c in contours)
    features.extend([edge_count, contour_area])

    # HSV 평균값 (3개)
    hue_mean = np.mean(hsv[:, :, 0])
    saturation_mean = np.mean(hsv[:, :, 1])
    value_mean = np.mean(hsv[:, :, 2])
    features.extend([hue_mean, saturation_mean, value_mean])
    
    # === LED 특화 추가 특징들 (21개) ===
    
    # 밝기 분포 특징 (3개)
    min_brightness = np.min(gray)
    brightness_range = max_brightness - min_brightness
    brightness_median = np.median(gray)
    features.extend([min_brightness, brightness_range, brightness_median])
    
    # RGB 최대값 및 비율 (6개)
    max_red = np.max(image[:,:,2])
    max_green = np.max(image[:,:,1])
    max_blue = np.max(image[:,:,0])
    rgb_dominance = np.argmax([avg_r, avg_g, avg_b])  # 0=R, 1=G, 2=B
    rg_ratio = avg_r / (avg_g + 1e-6)
    rb_ratio = avg_r / (avg_b + 1e-6)
    features.extend([max_red, max_green, max_blue, rgb_dominance, rg_ratio, rb_ratio])
    
    # HSV 확장 특징 (3개)
    hue_std = np.std(hsv[:, :, 0])
    saturation_std = np.std(hsv[:, :, 1])
    saturation_max = np.max(hsv[:, :, 1])
    features.extend([hue_std, saturation_std, saturation_max])
    
    # 색상 마스킹 확장 (3개)
    bright_pixel_ratio = mask_ratio(hsv, (0, 0, 200), (180, 255, 255))
    dark_pixel_ratio = mask_ratio(hsv, (0, 0, 0), (180, 255, 50))
    high_sat_ratio = mask_ratio(hsv, (0, 150, 50), (180, 255, 255))
    features.extend([bright_pixel_ratio, dark_pixel_ratio, high_sat_ratio])
    
    # LED 중심/가장자리 특징 (3개)
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    edge_region = np.concatenate([gray[:h//4, :].flatten(), gray[3*h//4:, :].flatten(),
                                 gray[:, :w//4].flatten(), gray[:, 3*w//4:].flatten()])
    
    center_brightness = np.mean(center)
    edge_brightness = np.mean(edge_region)
    center_edge_ratio = center_brightness / (edge_brightness + 1e-6)
    features.extend([center_brightness, edge_brightness, center_edge_ratio])
    
    # 핫스팟 감지 (2개)
    hotspot_mask = gray > (np.max(gray) * 0.8)
    hotspot_count = np.count_nonzero(hotspot_mask)
    hotspot_ratio = hotspot_count / (gray.size + 1e-6)
    features.extend([hotspot_count, hotspot_ratio])
    
    # 색상 균일성 (1개)
    color_uniformity = 1.0 / (1.0 + np.std(hsv[:,:,0]) + np.std(hsv[:,:,1]))
    features.append(color_uniformity)
    
    return features


def apply_clahe(image):
    """CLAHE 전처리 (노트북과 100% 동일)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def apply_gaussian_blur(image):
    """가우시안 블러 전처리 (노트북과 100% 동일)"""
    return cv2.GaussianBlur(image, (5, 5), 0.5)


def apply_hsv_boost(image, delta=10):
    """HSV 밝기 증가 전처리 (노트북과 100% 동일)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def load_lgbm_model():
    """LGBM 모델 로드 (캐싱)"""
    global _cached_model
    
    if '_cached_model' not in globals():
        try:
            # 새로운 모델 경로: ML_newdata/best_lgbm_model.pkl
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'ML_newdata', 'best_lgbm_model.pkl')
            _cached_model = joblib.load(model_path)
            print(f"✅ LGBM 모델 로드 성공: ML_newdata/best_lgbm_model.pkl")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            # 백업 모델 시도 (기존 경로)
            try:
                backup_path = os.path.join(os.path.dirname(__file__), 'model', 'ml', 'best_lgbm_model_140features.pkl')
                _cached_model = joblib.load(backup_path)
                print(f"⚠️ 백업 모델 사용: ml/best_lgbm_model_140features.pkl")
            except:
                _cached_model = None
    
    return _cached_model