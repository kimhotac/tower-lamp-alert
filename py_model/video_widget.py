from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRect, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from ultralytics import YOLO
class VideoWidget(QLabel):
    roi_created = pyqtSignal(QRect)
    roi_updated = pyqtSignal(int, QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.roi_list = []
        self.drawing = False
        self.moving = False
        self.selected_index = -1
        self.offset = None
        self.start_point = None
        self.end_point = None
        self.frame = None
        self.model = YOLO("model/YoLo/model.pt")

    def set_frame(self, frame):
        self.frame = frame.copy()
        self.repaint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if not self.drawing:
                self.start_point = event.pos()
                self.current_point = event.pos()
                self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.create_roi()
            
    def delete_roi(self, index: int):
        if 0 <= index < len(self.roi_list):
            del self.roi_list[index]
            self.repaint()

    def create_roi(self):
        """ROI 생성 메서드"""
        if self.start_point and self.current_point:
            x1 = min(self.start_point.x(), self.current_point.x())
            y1 = min(self.start_point.y(), self.current_point.y())
            x2 = max(self.start_point.x(), self.current_point.x())
            y2 = max(self.start_point.y(), self.current_point.y())
            
            # ROI가 너무 작지 않은지 확인
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                roi = QRect(x1, y1, x2 - x1, y2 - y1)
                self.roi_list.append(roi)
                self.roi_created.emit(roi)  # ROI 생성 시그널 발생
                
        # 그리기 상태 초기화
        self.start_point = None
        self.current_point = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.frame is None:
            return

        h, w, ch = self.frame.shape
        q_img = QImage(self.frame.data, w, h, 3 * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        painter = QPainter(pixmap)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)

        for rect in self.roi_list:
            painter.drawRect(rect)

        if self.drawing and self.start_point and self.current_point:
            painter.drawRect(QRect(self.start_point, self.current_point).normalized())

        painter.end()
        self.setPixmap(pixmap)

    def auto_add_roi(self):
        """YOLOv8로 감지된 모든 객체를 ROI로 자동 추가"""
        if self.frame is None:
            return

        # BGR → RGB 변환
        rgb_frame = self.frame[..., ::-1]

        # YOLOv8 예측 실행 (결과는 list, 첫 번째 결과 사용)
        results = self.model.predict(rgb_frame, verbose=False)[0]

        # 감지된 모든 바운딩 박스에 대해 처리
        added = 0
        for box in results.boxes:
            # 바운딩 박스 좌표 (xyxy): tensor([[x1, y1, x2, y2]])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1

            if w < 10 or h < 10:
                continue  # 너무 작은 ROI는 무시

            rect = QRect(x1, y1, w, h)
            self.roi_list.append(rect)
            self.roi_created.emit(rect)
            added += 1

        self.update()