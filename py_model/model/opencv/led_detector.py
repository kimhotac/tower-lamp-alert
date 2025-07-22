import cv2
import numpy as np

class LedDetector:
    def __init__(self):
        # '켜진 빛'을 찾기 위한 HSV 임계값 (V값을 대폭 상향!)
        self.saturation_threshold = 100
        self.value_threshold = 220 # 진짜 '빛'은 매우 밝다는 점을 이용
        
        self.min_contour_area = 100 # 노이즈 제거용 최소 면적 증가
        
        # 색상 범위 확장 (더 너그럽게)
        self.color_ranges = {
            'red': [(0, 12), (170, 180)],
            'yellow': (15, 40), # 노란색 범위 확장
            'green': (40, 90)  # 초록색 범위 확장
        }

    def find_lit_led_contours(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, 
                           np.array([0, self.saturation_threshold, self.value_threshold]), 
                           np.array([180, 255, 255]))
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        
        return valid_contours

    def determine_color(self, hue):
        for r_start, r_end in self.color_ranges['red']:
            if r_start <= hue <= r_end:
                return 'red'
        
        y_start, y_end = self.color_ranges['yellow']
        if y_start <= hue <= y_end:
            return 'yellow'
            
        g_start, g_end = self.color_ranges['green']
        if g_start <= hue <= g_end:
            return 'green'
            
        return None

    def detect(self, frame):
        contours = self.find_lit_led_contours(frame)
        
        if not contours:
            return 'off'
            
        lit_leds = []
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for cnt in contours:
            # === 핵심 개선 로직: Contour 내 가장 밝은 픽셀만 분석 ===
            mask = np.zeros(hsv_frame.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Contour 내의 모든 픽셀을 가져옴
            points = hsv_frame[mask == 255]
            
            # 밝기(V) 기준으로 정렬하여 상위 10% 픽셀만 선택
            points = sorted(points, key=lambda x: x[2], reverse=True)
            brightest_points = points[:len(points)//10 + 1] # 0으로 나눠지는 경우 방지
            
            # 상위 픽셀들의 평균 Hue와 V 계산
            avg_hue = np.mean([p[0] for p in brightest_points])
            avg_value = np.mean([p[2] for p in brightest_points])

            color = self.determine_color(avg_hue)
            if color:
                lit_leds.append({'color': color, 'brightness': avg_value})
        
        if not lit_leds:
            return 'off'
        
        brightest_led = max(lit_leds, key=lambda x: x['brightness'])
        return brightest_led['color']

    # debug_info 함수는 이전과 동일하게 사용 가능합니다.
    def debug_info(self, frame):
        # ... (이전 코드와 동일)
        contours = self.find_lit_led_contours(frame)
        
        print("\nContour 분석 결과:")
        print("-" * 50)

        if not contours:
            print("켜진 것으로 판단되는 LED를 찾을 수 없습니다.")
            return frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        debug_frame = frame.copy()

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_hsv = cv2.mean(hsv, mask=mask)

            print(f"Contour {i+1}:")
            print(f"위치(x,y,w,h): ({x}, {y}, {w}, {h})")
            print(f"면적: {cv2.contourArea(cnt):.1f}")
            print(f"평균 HSV: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")
            
            points = hsv[mask == 255]
            points = sorted(points, key=lambda x: x[2], reverse=True)
            brightest_points = points[:len(points)//10 + 1]
            avg_hue = np.mean([p[0] for p in brightest_points])
            
            color = self.determine_color(avg_hue)
            print(f"핵심부 판정 색상: {color} (평균 Hue: {avg_hue:.1f})")
            print("-" * 30)
            
        return debug_frame