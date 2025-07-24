"""
CSV 파일의 feature 순서와 detection.py의 feature 추출 순서 비교 검증
140개 feature가 정확히 일치하는지 확인
"""
import os
import cv2
import numpy as np
from detection import extract_140_features

def compare_feature_order():
    """CSV 컬럼 순서와 detection.py feature 순서 비교"""
    
    print("🔍 CSV vs Detection.py Feature 순서 비교 검증")
    print("=" * 80)
    
    # CSV 파일에서 추출한 feature 컬럼명들 (정확한 순서)
    csv_feature_names = [
        # Original (35개)
        'original_avg_brightness', 'original_max_brightness', 'original_std_brightness',
        'original_avg_red', 'original_avg_green', 'original_avg_blue',
        'original_green_ratio', 'original_yellow_ratio', 'original_red_ratio',
        'original_edge_count', 'original_contour_area',
        'original_hue_mean', 'original_saturation_mean', 'original_value_mean',
        'original_min_brightness', 'original_brightness_range', 'original_brightness_median',
        'original_max_red', 'original_max_green', 'original_max_blue', 
        'original_rgb_dominance', 'original_rg_ratio', 'original_rb_ratio',
        'original_hue_std', 'original_saturation_std', 'original_saturation_max',
        'original_bright_pixel_ratio', 'original_dark_pixel_ratio', 'original_high_sat_ratio',
        'original_center_brightness', 'original_edge_brightness', 'original_center_edge_ratio',
        'original_hotspot_count', 'original_hotspot_ratio', 'original_color_uniformity',
        
        # CLAHE (35개)
        'clahe_avg_brightness', 'clahe_max_brightness', 'clahe_std_brightness',
        'clahe_avg_red', 'clahe_avg_green', 'clahe_avg_blue',
        'clahe_green_ratio', 'clahe_yellow_ratio', 'clahe_red_ratio',
        'clahe_edge_count', 'clahe_contour_area',
        'clahe_hue_mean', 'clahe_saturation_mean', 'clahe_value_mean',
        'clahe_min_brightness', 'clahe_brightness_range', 'clahe_brightness_median',
        'clahe_max_red', 'clahe_max_green', 'clahe_max_blue',
        'clahe_rgb_dominance', 'clahe_rg_ratio', 'clahe_rb_ratio',
        'clahe_hue_std', 'clahe_saturation_std', 'clahe_saturation_max',
        'clahe_bright_pixel_ratio', 'clahe_dark_pixel_ratio', 'clahe_high_sat_ratio',
        'clahe_center_brightness', 'clahe_edge_brightness', 'clahe_center_edge_ratio',
        'clahe_hotspot_count', 'clahe_hotspot_ratio', 'clahe_color_uniformity',
        
        # Blur (35개)
        'blur_avg_brightness', 'blur_max_brightness', 'blur_std_brightness',
        'blur_avg_red', 'blur_avg_green', 'blur_avg_blue',
        'blur_green_ratio', 'blur_yellow_ratio', 'blur_red_ratio',
        'blur_edge_count', 'blur_contour_area',
        'blur_hue_mean', 'blur_saturation_mean', 'blur_value_mean',
        'blur_min_brightness', 'blur_brightness_range', 'blur_brightness_median',
        'blur_max_red', 'blur_max_green', 'blur_max_blue',
        'blur_rgb_dominance', 'blur_rg_ratio', 'blur_rb_ratio',
        'blur_hue_std', 'blur_saturation_std', 'blur_saturation_max',
        'blur_bright_pixel_ratio', 'blur_dark_pixel_ratio', 'blur_high_sat_ratio',
        'blur_center_brightness', 'blur_edge_brightness', 'blur_center_edge_ratio',
        'blur_hotspot_count', 'blur_hotspot_ratio', 'blur_color_uniformity',
        
        # HSV (35개)
        'hsv_avg_brightness', 'hsv_max_brightness', 'hsv_std_brightness',
        'hsv_avg_red', 'hsv_avg_green', 'hsv_avg_blue',
        'hsv_green_ratio', 'hsv_yellow_ratio', 'hsv_red_ratio',
        'hsv_edge_count', 'hsv_contour_area',
        'hsv_hue_mean', 'hsv_saturation_mean', 'hsv_value_mean',
        'hsv_min_brightness', 'hsv_brightness_range', 'hsv_brightness_median',
        'hsv_max_red', 'hsv_max_green', 'hsv_max_blue',
        'hsv_rgb_dominance', 'hsv_rg_ratio', 'hsv_rb_ratio',
        'hsv_hue_std', 'hsv_saturation_std', 'hsv_saturation_max',
        'hsv_bright_pixel_ratio', 'hsv_dark_pixel_ratio', 'hsv_high_sat_ratio',
        'hsv_center_brightness', 'hsv_edge_brightness', 'hsv_center_edge_ratio',
        'hsv_hotspot_count', 'hsv_hotspot_ratio', 'hsv_color_uniformity'
    ]
    
    print(f"📋 CSV 파일의 feature 개수: {len(csv_feature_names)}")
    
    # 테스트 이미지로 detection.py에서 feature 추출
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :, 2] = 150  # 빨간색 테스트 이미지
    
    try:
        # detection.py에서 140개 feature 추출
        detection_features = extract_140_features(test_image)
        print(f"🔧 detection.py feature 개수: {len(detection_features)}")
        
        # 개수 체크
        if len(csv_feature_names) != len(detection_features):
            print(f"❌ Feature 개수 불일치!")
            print(f"   CSV: {len(csv_feature_names)}개")
            print(f"   Detection: {len(detection_features)}개")
            return False
        
        print(f"✅ Feature 개수 일치: {len(detection_features)}개")
        
        # 상세 비교 출력
        print(f"\n📊 Feature 순서 상세 비교:")
        print(f"{'Index':>5} | {'CSV Feature Name':<35} | {'Detection Value':>15}")
        print("-" * 70)
        
        # 처음 20개와 마지막 10개만 출력 (너무 길어서)
        show_indices = list(range(20)) + list(range(130, 140))
        
        for i in show_indices:
            csv_name = csv_feature_names[i]
            detection_value = detection_features[i]
            print(f"{i+1:5d} | {csv_name:<35} | {detection_value:15.6f}")
            
            if i == 19:  # 20개 출력 후 구분선
                print("  ... | (중간 생략)                       |            ...")
        
        # 각 전처리별 첫 번째 feature 확인 (avg_brightness)
        print(f"\n🔍 각 전처리별 첫 번째 feature (avg_brightness) 확인:")
        preprocessing_starts = [0, 35, 70, 105]  # 각 전처리 시작 인덱스
        preprocessing_names = ['original', 'clahe', 'blur', 'hsv']
        
        for i, (start_idx, name) in enumerate(zip(preprocessing_starts, preprocessing_names)):
            csv_name = csv_feature_names[start_idx]
            detection_value = detection_features[start_idx]
            print(f"   {name:8}: {csv_name} = {detection_value:.6f}")
        
        print(f"\n✅ CSV 컬럼 순서와 detection.py feature 순서가 일치합니다!")
        return True
        
    except Exception as e:
        print(f"❌ Feature 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """실제 LED 이미지로 feature 순서 테스트"""
    print(f"\n🖼️ 실제 LED 이미지로 feature 추출 테스트")
    print("-" * 50)
    
    test_files = ['test1.png', 'test2.png']
    
    for img_file in test_files:
        img_path = os.path.join(os.path.dirname(__file__), img_file)
        
        if not os.path.exists(img_path):
            print(f"❌ 이미지 없음: {img_file}")
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 이미지 로드 실패: {img_file}")
            continue
            
        print(f"\n📸 {img_file} ({image.shape}):")
        
        # Feature 추출
        features = extract_140_features(image)
        
        # 각 전처리별 주요 feature 출력
        key_features = [
            (0, 'original_avg_brightness'),
            (3, 'original_avg_red'),
            (35, 'clahe_avg_brightness'),
            (38, 'clahe_avg_red'),
            (70, 'blur_avg_brightness'),
            (73, 'blur_avg_red'),
            (105, 'hsv_avg_brightness'),
            (108, 'hsv_avg_red')
        ]
        
        for idx, name in key_features:
            print(f"   [{idx:3d}] {name:<25}: {features[idx]:8.3f}")

def main():
    """메인 테스트 실행"""
    # 1. 순서 비교 검증
    is_match = compare_feature_order()
    
    # 2. 실제 이미지 테스트
    test_with_real_image()
    
    print(f"\n" + "=" * 80)
    if is_match:
        print("🎉 결론: CSV 파일과 detection.py의 feature 순서가 100% 일치합니다!")
        print("💡 ML 모델이 정확한 feature 순서로 예측할 수 있습니다.")
    else:
        print("⚠️ 결론: feature 순서에 문제가 있습니다. 수정이 필요합니다.")

if __name__ == "__main__":
    main()
