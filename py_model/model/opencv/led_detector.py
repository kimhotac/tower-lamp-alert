import cv2
import numpy as np
import sys
import os

# util 모듈을 import하기 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.image_utils import ImageProcessor

class LedDetector:
    """
    원형 LED 검출 및 상태 판별 클래스
    1. 원형 LED 위치 검출
    2. LED 켜짐/꺼짐 상태 판별
    3. 켜진 LED의 색상 판별
    """
    def __init__(self, use_preprocessing=True, target_size=(100, 100)):
        self.use_preprocessing = use_preprocessing
        self.target_size = target_size
        self.image_processor = ImageProcessor()
        
        # LED 상태 판별을 위한 임계값 (전처리 효과 고려하여 조정)
        self.saturation_threshold = 80   # 채도 임계값 (전처리로 채도가 높아져서 낮춤)
        self.value_threshold = 120       # 명도 임계값 (전처리로 밝기가 개선되어 낮춤)
        
        # 색상 범위 정의 (HSV의 Hue 값) - 기본값 유지, determine_color에서 조정
        self.color_ranges = {
            'red': [(0, 15), (170, 180)],  # 빨간색은 양 끝 범위
            'yellow': (20, 40),
            'green': (45, 85)
        }

    def preprocess_image(self, image):
        """
        이미지 전처리 (리사이징 + 색상 보정 + 노이즈 제거)
        """
        processed = image.copy()
        
        if self.use_preprocessing:
            # 1. 리사이징
            processed = self.image_processor.resize_image(
                processed, 
                self.target_size, 
                maintain_aspect_ratio=True
            )
            
            # 2. 색상 보정 및 대비 향상
            processed = self.apply_color_correction(processed)
            
            # 3. 노이즈 제거
            processed = self.apply_noise_reduction(processed)
            
        return processed

    def apply_color_correction(self, image):
        """
        색상 보정 및 대비 향상
        """
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # L 채널에 CLAHE 적용하여 지역적 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # 채널 합치기
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. 감마 보정 (LED의 밝기 특성 고려)
        gamma_corrected = self.apply_gamma_correction(enhanced_bgr, gamma=1.2)
        
        # 3. 색상 채도 향상 (HSV에서 채도 증가)
        hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 채도를 1.3배 증가 (LED 색상 더 선명하게)
        s = cv2.multiply(s, 1.3)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        enhanced_hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        
        return result

    def apply_gamma_correction(self, image, gamma=1.0):
        """감마 보정"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_noise_reduction(self, image):
        """
        노이즈 제거 및 이미지 개선
        """
        # 1. 가우시안 블러로 노이즈 제거 (너무 강하지 않게)
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        
        # 2. 언샵 마스킹으로 엣지 보강
        blurred = cv2.GaussianBlur(denoised, (9, 9), 0)
        unsharp_mask = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
        
        # 3. 모폴로지 연산으로 작은 노이즈 제거
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(unsharp_mask, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def apply_led_specific_enhancement(self, image):
        """LED 영역에 특화된 향상 처리"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # LED 색상 영역별 마스크 생성
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        # 모든 LED 마스크 결합
        led_mask = cv2.bitwise_or(red_mask1, red_mask2)
        led_mask = cv2.bitwise_or(led_mask, yellow_mask)
        led_mask = cv2.bitwise_or(led_mask, green_mask)
        
        # 마스크 확장 (약간의 여유 공간)
        kernel = np.ones((3, 3), np.uint8)
        led_mask = cv2.dilate(led_mask, kernel, iterations=1)
        
        # LED 영역의 채도와 명도 향상 (올바른 방법)
        s_enhanced = s.copy().astype(np.float32)
        v_enhanced = v.copy().astype(np.float32)
        
        # LED 영역만 향상 처리
        s_enhanced[led_mask > 0] = np.clip(s_enhanced[led_mask > 0] * 1.4, 0, 255)
        v_enhanced[led_mask > 0] = np.clip(v_enhanced[led_mask > 0] * 1.2, 0, 255)
        
        # 다시 uint8로 변환
        s_enhanced = s_enhanced.astype(np.uint8)
        v_enhanced = v_enhanced.astype(np.uint8)
        
        # HSV 채널 재결합
        enhanced_hsv = cv2.merge([h, s_enhanced, v_enhanced])
        
        # BGR로 변환하여 반환
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    def find_led_circles(self, image):
        """
        원형 LED 검출 (개선된 전처리 적용)
        """
        # LED 특성에 맞는 추가 전처리
        enhanced_image = self.apply_led_specific_enhancement(image)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        # 적응적 히스토그램 평활화로 대비 개선
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 노이즈 제거 (median blur가 원형 특성 보존에 좋음)
        gray = cv2.medianBlur(gray, 5)
        
        # 원 검출 (파라미터 조정)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,  # 원 중심 간 최소 거리 (더 가깝게)
            param1=50,   # Canny edge 검출 임계값
            param2=25,   # 원 중심 검출 임계값 (더 민감하게)
            minRadius=3, # 최소 반지름 (더 작은 LED도 검출)
            maxRadius=60 # 최대 반지름 (더 큰 LED도 검출)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            
            # 원 품질 검증 (너무 겹치거나 이상한 원 제거)
            filtered_circles = self.filter_circles(circles, gray)
            return filtered_circles
            
        return None

    def filter_circles(self, circles, gray_image):
        """
        검출된 원들의 품질을 검증하고 필터링
        """
        if len(circles) == 0:
            return None
            
        valid_circles = []
        
        for circle in circles:
            x, y, r = circle
            
            # 이미지 경계 확인
            if (x - r < 0 or x + r >= gray_image.shape[1] or 
                y - r < 0 or y + r >= gray_image.shape[0]):
                continue
                
            # 원 영역의 품질 검증
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # 원 영역의 표준편차 계산 (균일한 영역인지 확인)
            roi_pixels = gray_image[mask > 0]
            if len(roi_pixels) > 0:
                std_dev = np.std(roi_pixels)
                mean_val = np.mean(roi_pixels)
                
                # 너무 균일하거나 너무 불균일한 원 제거
                if 10 < std_dev < 80 and mean_val > 30:
                    valid_circles.append(circle)
        
        return np.array(valid_circles) if valid_circles else None

    def analyze_led_state(self, image, circle):
        """
        개별 LED의 상태 분석 (개선된 색상 분석)
        """
        x, y, r = circle
        
        # 원형 마스크 생성
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # 중심부 마스크 생성 (더 정확한 색상 분석을 위해)
        center_mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(center_mask, (x, y), max(1, r//2), 255, -1)
        
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 전체 원 영역의 평균값
        mean_hsv_full = cv2.mean(hsv, mask=mask)
        
        # 중심부 영역의 평균값 (더 신뢰성 있는 색상 정보)
        mean_hsv_center = cv2.mean(hsv, mask=center_mask)
        
        # 중심부와 전체 영역의 가중 평균 (중심부에 더 높은 가중치)
        weight_center = 0.7
        weight_full = 0.3
        
        final_hue = (mean_hsv_center[0] * weight_center + 
                    mean_hsv_full[0] * weight_full)
        final_saturation = (mean_hsv_center[1] * weight_center + 
                           mean_hsv_full[1] * weight_full)
        final_value = (mean_hsv_center[2] * weight_center + 
                      mean_hsv_full[2] * weight_full)
        
        # LED 상태 분석 결과
        return {
            'position': (x, y),
            'radius': r,
            'hue': final_hue,
            'saturation': final_saturation,
            'value': final_value
        }

    def determine_color(self, hue):
        """색상(Hue) 값으로 LED 색상 판별 (색상 보정된 이미지에 맞게 조정)"""
        # 색상 보정으로 인해 색상 범위를 약간 넓게 조정
        color_ranges_adjusted = {
            'red': [(0, 18), (165, 180)],    # 빨간색 범위 확장
            'yellow': (18, 45),              # 노란색 범위 확장
            'green': (40, 90)                # 초록색 범위 확장
        }
        
        # 빨간색 (양 끝 범위 확인)
        for range_start, range_end in color_ranges_adjusted['red']:
            if range_start <= hue <= range_end:
                return 'red'
        
        # 노란색
        yellow_start, yellow_end = color_ranges_adjusted['yellow']
        if yellow_start <= hue <= yellow_end:
            return 'yellow'
            
        # 초록색
        green_start, green_end = color_ranges_adjusted['green']
        if green_start <= hue <= green_end:
            return 'green'
            
        return None

    def detect(self, frame):
        """
        LED 검출 및 상태 판별 메인 함수
        return: 'red', 'yellow', 'green', 'off' 중 하나
        """
        # 전처리 적용
        processed_frame = self.preprocess_image(frame)
        
        # 1. LED 원형 검출
        circles = self.find_led_circles(processed_frame)
        if circles is None:
            return 'off'

        # 2. 각 원의 상태 분석
        led_states = []
        for circle in circles:
            state = self.analyze_led_state(processed_frame, circle)
            
            # 켜진 LED 판별 (채도와 명도 모두 고려)
            if (state['saturation'] > self.saturation_threshold and 
                state['value'] > self.value_threshold):
                
                color = self.determine_color(state['hue'])
                if color:
                    led_states.append({
                        'color': color,
                        'brightness': state['value'],
                        'saturation': state['saturation']
                    })
        
        # 3. 최종 상태 판별
        if not led_states:
            return 'off'
            
        # 가장 선명한(채도가 높은) LED의 색상 반환
        brightest_led = max(led_states, key=lambda x: x['saturation'])
        return brightest_led['color']
    def debug_info(self, frame):
        """디버깅용 상세 정보 출력 및 시각화된 이미지 반환"""
        print("\n=== 전처리 과정 ===")
        print("1. 원본 이미지 분석...")
        
        # 전처리 적용
        processed_frame = self.preprocess_image(frame)
        debug_frame = processed_frame.copy()
        
        print("2. 색상 보정 및 노이즈 제거 완료")
        
        circles = self.find_led_circles(processed_frame)
        if circles is None:
            print("3. LED를 찾을 수 없습니다.")
            return debug_frame

        print(f"3. {len(circles)}개의 원형 LED 후보 발견")
        print("\n=== LED 분석 결과 ===")
        print("-" * 50)
        
        for i, circle in enumerate(circles):
            state = self.analyze_led_state(processed_frame, circle)
            x, y, r = circle
            
            print(f"LED {i+1}:")
            print(f"위치: ({x}, {y})")
            print(f"반지름: {r}")
            print(f"HSV: H={state['hue']:.1f}, S={state['saturation']:.1f}, V={state['value']:.1f}")
            
            # 임계값 검사 상세 출력
            sat_pass = state['saturation'] > self.saturation_threshold
            val_pass = state['value'] > self.value_threshold
            print(f"채도 검사: {state['saturation']:.1f} > {self.saturation_threshold} = {'통과' if sat_pass else '실패'}")
            print(f"명도 검사: {state['value']:.1f} > {self.value_threshold} = {'통과' if val_pass else '실패'}")
            
            # 원 그리기
            if sat_pass and val_pass:
                color = self.determine_color(state['hue'])
                print(f"판정: 켜짐 (색상: {color})")
                # 켜진 LED는 빨간색 원으로 표시
                cv2.circle(debug_frame, (x, y), r, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"{color}" if color else "unknown", (x-15, y-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(debug_frame, f"S:{state['saturation']:.0f}", (x-15, y+r+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            else:
                print("판정: 꺼짐")
                # 꺼진 LED는 회색 원으로 표시
                cv2.circle(debug_frame, (x, y), r, (128, 128, 128), 2)
                cv2.putText(debug_frame, "off", (x-10, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            print("-" * 30)
            
        return debug_frame

    def get_preprocessing_steps(self, frame):
        """전처리 각 단계별 결과를 반환 (디버깅용)"""
        steps = {}
        
        # 1. 원본
        steps['original'] = frame.copy()
        
        if self.use_preprocessing:
            # 2. 리사이징
            resized = self.image_processor.resize_image(
                frame, self.target_size, maintain_aspect_ratio=True
            )
            steps['resized'] = resized
            
            # 3. 색상 보정
            color_corrected = self.apply_color_correction(resized)
            steps['color_corrected'] = color_corrected
            
            # 4. 노이즈 제거
            noise_reduced = self.apply_noise_reduction(color_corrected)
            steps['noise_reduced'] = noise_reduced
            
            # 5. LED 특화 전처리
            led_enhanced = self.apply_led_specific_enhancement(noise_reduced)
            steps['led_enhanced'] = led_enhanced
            
            steps['final'] = led_enhanced
        else:
            steps['final'] = frame
            
        return steps

    def extract_features_for_ml(self, frame):
        """
        머신러닝을 위한 특징 추출
        OpenCV 기반 분석으로 6가지 핵심 특징을 추출합니다.
        
        Returns:
            dict: {
                'opencv_decision': str,  # OpenCV만으로 판정한 결과
                'count': int,           # 검출된 LED 개수
                'hue': float,          # 평균 색상값
                'value': float,        # 평균 명도값
                'saturation': float,   # 평균 채도값
                'area': float          # 검출된 LED의 평균 면적
            }
        """
        # 전처리 적용
        processed_frame = self.preprocess_image(frame)
        
        # OpenCV 판정 결과
        opencv_decision = self.detect(frame)
        
        # LED 원형 검출
        circles = self.find_led_circles(processed_frame)
        
        if circles is None:
            return {
                'opencv_decision': opencv_decision,
                'count': 0,
                'hue': 0.0,
                'value': 0.0,  
                'saturation': 0.0,
                'area': 0.0
            }
        
        # 각 원의 특징 분석
        led_features = []
        total_area = 0
        
        for circle in circles:
            state = self.analyze_led_state(processed_frame, circle)
            x, y, r = circle
            
            # 켜진 LED만 특징으로 사용
            if (state['saturation'] > self.saturation_threshold and 
                state['value'] > self.value_threshold):
                
                area = np.pi * r * r  # 원의 면적
                led_features.append({
                    'hue': state['hue'],
                    'saturation': state['saturation'],
                    'value': state['value'],
                    'area': area
                })
                total_area += area
        
        # 특징이 없는 경우 (모든 LED가 꺼진 경우)
        if not led_features:
            return {
                'opencv_decision': opencv_decision,
                'count': len(circles),  # 검출된 원의 개수는 기록
                'hue': 0.0,
                'value': 0.0,
                'saturation': 0.0,
                'area': 0.0
            }
        
        # 평균값 계산
        avg_hue = np.mean([f['hue'] for f in led_features])
        avg_saturation = np.mean([f['saturation'] for f in led_features])
        avg_value = np.mean([f['value'] for f in led_features])
        avg_area = total_area / len(led_features)
        
        return {
            'opencv_decision': opencv_decision,
            'count': len(led_features),  # 켜진 LED의 개수
            'hue': round(avg_hue, 2),
            'value': round(avg_value, 2),
            'saturation': round(avg_saturation, 2),
            'area': round(avg_area, 2)
        }