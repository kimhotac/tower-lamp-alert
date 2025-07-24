import time
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer, QRect
from PyQt5.QtGui import QPixmap, QIcon, QFont

from video_widget import VideoWidget
from roi_list_widget import ROIListWidget
from video_capture import VideoCaptureHandler
from detection import detect
from util.email_notifier import EmailNotifier

# 전역 변수
last_alert_time = 0
ALERT_COOLDOWN = 600  # 초 단위 (10분)

# 이메일 알림 설정
notifier = EmailNotifier(
    sender_email='your_email@gmail.com',
    sender_password='your_app_password'  # 앱 비밀번호 (실제 계정 비번 아님)
)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("번쩍이 3000")
        self.setWindowIcon(QIcon('icon.png'))
        self.resize(1000, 500)

        self.video_widget = VideoWidget()
        self.roi_list_widget = ROIListWidget()
        self.capture_handler = VideoCaptureHandler()

        # ⏱ 가동률 계산용 변수 초기화
        self.total_runtime = 0
        self.green_runtime = 0
        self.last_check_time = time.time()

        # 📊 상태 표시용 라벨
        self.status_label = QLabel("상태: 초기화 중")
        self.uptime_label = QLabel("가동률: 0.00%")
        font = QFont("Arial", 12)
        self.status_label.setFont(font)
        self.uptime_label.setFont(font)

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

        # 가동률 및 상태 라벨 추가
        side_layout.addWidget(self.status_label)
        side_layout.addWidget(self.uptime_label)

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

        # ⏱ 시간 누적
        now = time.time()
        dt = now - self.last_check_time
        self.last_check_time = now
        self.total_runtime += dt

        green_detected_any = False  # 여러 ROI 중 하나라도 green이면 True

        for i, rect in enumerate(self.video_widget.roi_list):
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            if x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            roi_img = frame[y:y + h, x:x + w]
            result = detect(roi_img)

            # 알림 처리 (쿨타임)
            current_time = time.time()
            if result == 'red' and (current_time - last_alert_time) > ALERT_COOLDOWN:
                notifier.send_alert()
                last_alert_time = current_time

            if result == 'green':
                green_detected_any = True

            self.roi_list_widget.update_prediction(i, result)

        # ✅ 상태 표시
        if green_detected_any:
            self.green_runtime += dt
            self.status_label.setText("상태: ✅ 가동 중 (Green)")
        else:
            self.status_label.setText("상태: ⛔ 비가동")

        # 📊 가동률 계산 및 표시
        if self.total_runtime > 0:
            uptime_ratio = (self.green_runtime / self.total_runtime) * 100
            self.uptime_label.setText(f"가동률: {uptime_ratio:.2f}%")

    def closeEvent(self, event):
        self.capture_handler.release()
        self.detect_timer.stop()
        super().closeEvent(event)
