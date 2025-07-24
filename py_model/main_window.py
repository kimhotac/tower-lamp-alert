import random
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer
import time

from video_widget import VideoWidget
from roi_list_widget import ROIListWidget
from video_capture import VideoCaptureHandler
from detection import detect
from util.email_notifier import EmailNotifier

last_alert_time = 0  # 최초 알림 시간 (0초부터 시작)
ALERT_COOLDOWN = 600  # 10분 (초 단위)

notifier = EmailNotifier(
    sender_email='asho27695193@gmail.com',           # 실제 Gmail 주소
    sender_password=''            # 2단계 인증이 끝난 실제 Gmail 앱 비밀번호를 생성 후 해당하는 비밀번호 입력
)                                                 # 실제 계정 비밀번호와는 다르다.


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI 선택, 이동, 삭제 + 예측 표시")
        self.resize(1000, 500)

        self.video_widget = VideoWidget()
        self.roi_list_widget = ROIListWidget()
        self.capture_handler = VideoCaptureHandler()

        self.setup_ui()
        self.connect_signals()

        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self.detect_all_rois)
        self.detect_timer.start(1000)

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.addWidget(self.video_widget)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.roi_list_widget)
        layout.addLayout(side_layout)

    def connect_signals(self):
        self.capture_handler.frame_captured.connect(self.video_widget.set_frame)
        self.video_widget.roi_created.connect(self.roi_list_widget.add_roi)
        self.video_widget.roi_updated.connect(self.roi_list_widget.update_roi)
        self.roi_list_widget.roi_deleted.connect(self.video_widget.delete_roi)

    def detect_all_rois(self):
        global last_alert_time

        frame = self.video_widget.frame
        if frame is None:
            return

        for i, rect in enumerate(self.video_widget.roi_list):
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            if x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            roi_img = frame[y:y + h, x:x + w]
            result = detect(roi_img)

            # 알림 쿨타임 확인
            current_time = time.time()
            if result == 'red' and (current_time - last_alert_time) > ALERT_COOLDOWN:
                notifier.send_alert()
                last_alert_time = current_time  # 마지막 알림 시간 갱신

            self.roi_list_widget.update_prediction(i, result)

    def closeEvent(self, event):
        self.capture_handler.release()
        self.detect_timer.stop()
        super().closeEvent(event)
