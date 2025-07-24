"""
정적 이미지 파일로 LED Detection 테스트
test1.png, test2.png를 input으로 받아서 detection.py로 LED 상태 판별
"""
import os
import cv2
import numpy as np
from detection import detect, extract_140_features, load_lgbm_model

def test_image_detection(image_path):
    """단일 이미지 파일로 LED 감지 테스트"""
    
    print(f"\n📸 이미지 Detection 테스트")
    print(f"입력 파일: {os.path.basename(image_path)}")
    print("-" * 50)
    
    # 1. 이미지 로드
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일이 없습니다: {image_path}")
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        return None
    
    print(f"✅ 이미지 로드 성공: {image.shape} (H×W×C)")
    
    # 2. LED 감지 (전체 플로우)
    try:
        print(f"\n🔍 LED Detection 시작...")
        result = detect(image)
        print(f"🎯 최종 감지 결과: '{result}'")
        
        # 3. 상세 분석 정보
        print(f"\n📊 상세 분석:")
        
        # 140개 특징 추출 과정 보기
        features = extract_140_features(image)
        print(f"   총 특징 개수: {len(features)}")
        
        # 각 전처리별 평균 밝기 확인
        brightness_info = {
            '원본': features[0],      # original_avg_brightness
            'CLAHE': features[35],    # clahe_avg_brightness  
            '블러': features[70],     # blur_avg_brightness
            'HSV': features[105]      # hsv_avg_brightness
        }
        
        for name, brightness in brightness_info.items():
            print(f"   {name:6} 평균밝기: {brightness:.2f}")
        
        total_avg = sum(brightness_info.values()) / 4
        print(f"   전체 평균밝기: {total_avg:.2f}")
        
        # 4. 모델 정보
        model = load_lgbm_model()
        if model is not None:
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
            confidence = max(probabilities)
            
            print(f"   모델 예측값: {prediction}")
            print(f"   예측 신뢰도: {confidence:.3f}")
            print(f"   클래스별 확률: [off:{probabilities[0]:.3f}, green:{probabilities[1]:.3f}, yellow:{probabilities[2]:.3f}, red:{probabilities[3]:.3f}]")
        else:
            print(f"   ⚠️ 모델 로드 실패 - 규칙 기반 판단 사용됨")
            
        return result
        
    except Exception as e:
        print(f"❌ Detection 실행 에러: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_all_images():
    """모든 테스트 이미지 검사"""
    
    print("🚀 LED Image Detection 테스트 시작")
    print("=" * 60)
    
    # 테스트할 이미지 파일들
    test_files = ['test1.png', 'test2.png']
    
    results = {}
    
    for img_file in test_files:
        img_path = os.path.join(os.path.dirname(__file__), img_file)
        result = test_image_detection(img_path)
        results[img_file] = result
    
    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("📋 전체 테스트 결과 요약:")
    for img_file, result in results.items():
        status = result if result else "실패"
        print(f"   {img_file:12} → {status}")
    
    print("\n✅ 모든 테스트 완료!")
    return results

def test_custom_image(image_path):
    """사용자 지정 이미지 테스트"""
    if not os.path.isabs(image_path):
        # 상대 경로인 경우 현재 디렉토리 기준으로 변환
        image_path = os.path.join(os.path.dirname(__file__), image_path)
    
    return test_image_detection(image_path)

if __name__ == "__main__":
    # 기본 테스트 실행
    test_all_images()
    
    print(f"\n💡 사용법:")
    print(f"   python image_detection_test.py")
    print(f"   또는 다른 이미지 테스트:")
    print(f"   test_custom_image('your_image.jpg')")
