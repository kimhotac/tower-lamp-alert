import cv2
import numpy as np
from model.opencv.led_detector import LedDetector

def test_led_detection(image_path):
    """단일 이미지에 대한 LED 탐지 테스트"""
    print(f"\n테스트 이미지: {image_path}")
    print("=" * 50)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return
    
    detector = LedDetector()
    
    print("이미지 정보:")
    print(f"크기: {img.shape}")
    
    # 1. debug_info()를 한 번만 호출하여 결과 프레임을 받아옵니다.
    #    (함수 내부에서 print는 이미 실행됩니다)
    debug_frame = detector.debug_info(img) 
    
    # 2. detect() 함수로 최종 결과를 얻습니다.
    result = detector.detect(img)
    print("\n최종 판정 결과:", result)
    print("=" * 50)

    # 1. 이미지를 화면에 보여줍니다.
    cv2.imshow(f"Debug - {image_path}", debug_frame)

    # 2. 여기서 프로그램이 멈추고 '아무 키' 입력을 무한정 기다립니다.
    cv2.waitKey(0) 

    # 3. 키가 입력되면, 열려있는 모든 창을 닫습니다.
    cv2.destroyAllWindows() 

def test_all_images():
    """모든 테스트 이미지에 대해 LED 탐지 실행"""
    import glob
    import os
    
    # 현재 디렉토리의 모든 테스트 이미지 찾기
    image_files = glob.glob("*.png")
    
    if not image_files:
        print("현재 디렉토리에 테스트 이미지가 없습니다.")
        return
        
    print(f"발견된 테스트 이미지: {len(image_files)}개")
    print(image_files)
    print("\n테스트 시작...")
    
    # 각 이미지에 대해 테스트 실행
    for image_path in sorted(image_files):
        test_led_detection(image_path)

if __name__ == "__main__":
    test_all_images()
