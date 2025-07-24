import cv2
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

class VideoCaptureHandler(QObject):
    frame_captured = pyqtSignal(object)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_index)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.query_frame)
        self.timer.start(30)

    def query_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_captured.emit(frame)

    def release(self):
        self.timer.stop()
        self.cap.release()
