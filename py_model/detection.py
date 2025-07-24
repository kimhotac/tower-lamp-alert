import os
import pickle
import numpy as np
import cv2

def detect(roi_image):
    """
    기존 LGBM 모델을 사용한 LED 상태 감지
    
    Args:
        roi_image: ROI 영역 이미지 (numpy array)
        
    Returns:
        str: 예측된 LED 상태 ('red', 'green', 'yellow', 'off')
    """
    if roi_image is None:
        return 'off'
    
    try:
        # 1. 특징 추출
        features = _extract_features(roi_image)
        
        # 2. 모델 로드 시도
        model = _load_model()
        
        if model is not None:
            # 3. LGBM 모델 예측
            prediction = model.predict([features])[0]
            label_map = {0: 'off', 1: 'red', 2: 'yellow', 3: 'green'}
            result = label_map.get(prediction, 'off')
            print(f"🤖 LGBM 모델 예측: {result}")
            return result
        else:
            # 4. 모델 로드 실패 시 규칙 기반 판단
            print("⚠️ 모델 사용 불가 - 규칙 기반 판단 사용")
            return _rule_based_detection(features)
        
    except Exception as e:
        print(f"LED 감지 오류: {e}")
        return 'off'


def _rule_based_detection(features):
    """규칙 기반 LED 상태 판단 (모델 대체용)"""
    # 특징값 분해
    avg_brightness = features[0]
    max_brightness = features[1] 
    avg_red = features[3]
    avg_green = features[4]
    avg_blue = features[5]
    green_ratio = features[6]
    yellow_ratio = features[7]
    red_ratio = features[8]
    saturation_mean = features[12]
    value_mean = features[13]
    
    print(f"🔍 규칙 기반 분석:")
    print(f"   밝기: {avg_brightness:.1f}, 최대: {max_brightness:.1f}")
    print(f"   RGB: R={avg_red:.1f}, G={avg_green:.1f}, B={avg_blue:.1f}")
    print(f"   색상비율: R={red_ratio:.3f}, G={green_ratio:.3f}, Y={yellow_ratio:.3f}")
    print(f"   채도: {saturation_mean:.1f}, 명도: {value_mean:.1f}")
    
    # 1. 너무 어두우면 off
    if avg_brightness < 60 or max_brightness < 120:
        print("   판단: 너무 어두움 → off")
        return 'off'
    
    # 2. 채도가 너무 낮으면 off (흑백에 가까움)
    if saturation_mean < 40:
        print("   판단: 채도 낮음 → off")
        return 'off'
    
    # 3. 색상 비율로 1차 판단
    if green_ratio > 0.12:  # 12% 이상이면 녹색
        print("   판단: 녹색 비율 높음 → green")
        return 'green'
    elif red_ratio > 0.10:  # 10% 이상이면 빨강
        print("   판단: 빨간색 비율 높음 → red")
        return 'red'
    elif yellow_ratio > 0.15:  # 15% 이상이면 노랑
        print("   판단: 노란색 비율 높음 → yellow")
        return 'yellow'
    
    # 4. RGB 값으로 2차 판단
    if avg_green > avg_red + 5 and avg_green > avg_blue + 10:
        print("   판단: 녹색 RGB 우세 → green")
        return 'green'
    elif avg_red > avg_green + 5 and avg_red > avg_blue + 10:
        print("   판단: 빨간색 RGB 우세 → red")
        return 'red'
    elif avg_red > 85 and avg_green > 85 and avg_blue < 80:  # 빨강+녹색=노랑
        print("   판단: 노란색 RGB 패턴 → yellow")
        return 'yellow'
    
    # 5. 밝기와 채도로 최종 판단
    if avg_brightness > 90 and saturation_mean > 60:
        print("   판단: 밝고 채도 높음 - 색상 불분명하지만 켜져있음")
        # 가장 높은 RGB 값으로 추정
        if max(avg_red, avg_green, avg_blue) == avg_green:
            return 'green'
        elif max(avg_red, avg_green, avg_blue) == avg_red:
            return 'red'
        else:
            return 'yellow'
    
    print("   판단: 조건 미충족 → off")
    return 'off'


def _load_model():
    """LGBM 모델 로드 (캐싱)"""
    global _cached_model
    
    if '_cached_model' not in globals():
        try:
            import joblib  # 추가
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'ml', 'best_lgbm_model.pkl')
            _cached_model = joblib.load(model_path)  # pickle.load → joblib.load로 변경
            print(f"✅ LGBM 모델 로드: {model_path}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            _cached_model = None
    
    return _cached_model


def _apply_preprocessing_combination(image):
    """
    ML 모델 훈련 시 사용된 최적 전처리 조합 적용:
    1. 100x100 리사이징
    2. CLAHE 적용 
    3. 가우시안 블러 적용
    4. HSV 밝기 증가 적용
    """
    # 1. 100x100으로 리사이징 (ML 훈련 시와 동일)
    img = cv2.resize(image, (100, 100))
    
    # 2. CLAHE 적용
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 3. 가우시안 블러 적용
    img = cv2.GaussianBlur(img, (5, 5), 1)
    
    # 4. HSV 밝기 증가 적용 (delta=40)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 40, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img


def _extract_features(roi_image):
    """14개 특징 추출 (노트북의 extract_features 함수와 동일)"""
    
    # ML 훈련 시 사용된 전처리 조합 적용
    processed_image = _apply_preprocessing_combination(roi_image)
    
    # 특징 추출을 위한 변환
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

    # 1. 밝기 관련 특징
    avg_brightness = float(np.mean(gray))
    max_brightness = float(np.max(gray))
    std_brightness = float(np.std(gray))

    # 2. RGB 평균값 (노트북과 동일한 방식)
    avg_b, avg_g, avg_r = cv2.mean(processed_image)[:3]

    # 3. 색상 비율 계산 (HSV +40 전처리 후 조정된 임계값)
    def mask_ratio(hsv_img, lower, upper):
        mask = cv2.inRange(hsv_img, lower, upper)
        return np.count_nonzero(mask) / (mask.size + 1e-6)

    # HSV +40 증가 후 색상 범위 재조정
    green_ratio = mask_ratio(hsv, (35, 40, 50), (90, 255, 255))    # 녹색 범위 확장
    yellow_ratio = mask_ratio(hsv, (15, 40, 50), (40, 255, 255))   # 노랑 범위 확장
    red_ratio = (
        mask_ratio(hsv, (0, 40, 50), (15, 255, 255)) +             # 빨강 범위 확장
        mask_ratio(hsv, (155, 40, 50), (180, 255, 255))
    )

    # 4. 엣지 관련 특징
    edges = cv2.Canny(gray, 100, 200)  # 노트북과 동일한 파라미터
    edge_count = float(np.count_nonzero(edges))

    # 5. 윤곽선 특징
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_area = float(sum(cv2.contourArea(c) for c in contours))

    # 6. HSV 평균값
    hue_mean = float(np.mean(hsv[:, :, 0]))
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    value_mean = float(np.mean(hsv[:, :, 2]))

    # 14개 특징 순서 (노트북과 정확히 동일)
    features = [
        avg_brightness, max_brightness, std_brightness,
        avg_r, avg_g, avg_b,
        green_ratio, yellow_ratio, red_ratio,
        edge_count, contour_area,
        hue_mean, saturation_mean, value_mean
    ]
    
    return features

