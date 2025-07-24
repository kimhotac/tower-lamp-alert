import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QListWidget, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import QTimer, Qt, QRect, pyqtSignal, QObject


class VideoCaptureThread(QObject):
    frame_captured = pyqtSignal(object)

    def __init__(self, cam_index=0):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_index)
        self.timer = QTimer()
        self.timer.timeout.connect(self.query_frame)
        self.timer.start(30)

    def query_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_captured.emit(frame)

    def release(self):
        self.timer.stop()
        self.cap.release()


class ROIListWidget(QListWidget):
    roi_deleted = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

    def add_roi(self, rect: QRect):
        self.addItem(f"ROI - x:{rect.x()}, y:{rect.y()}, w:{rect.width()}, h:{rect.height()}")

    def update_roi(self, index: int, rect: QRect):
        self.item(index).setText(f"ROI - x:{rect.x()}, y:{rect.y()}, w:{rect.width()}, h:{rect.height()}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            selected = self.currentRow()
            if selected >= 0:
                self.takeItem(selected)
                self.roi_deleted.emit(selected)
        else:
            super().keyPressEvent(event)


class VideoWidget(QLabel):
    roi_created = pyqtSignal(QRect)
    roi_updated = pyqtSignal(int, QRect)
    roi_deleted = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_list = []
        self.drawing = False
        self.moving = False
        self.selected_index = -1
        self.offset = None
        self.setMouseTracking(True)
        self.frame = None

    def set_frame(self, frame):
        self.frame = frame.copy()
        self.repaint()

    def mousePressEvent(self, event):
        pos = event.pos()

        # ROI 이동 모드 체크
        for i, rect in enumerate(self.roi_list):
            if rect.contains(pos):
                self.selected_index = i
                self.moving = True
                self.offset = pos - rect.topLeft()
                return

        # 새 ROI 그리기
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = pos
            self.end_point = pos

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if self.drawing:
            self.end_point = pos
            self.repaint()
        elif self.moving and self.selected_index >= 0:
            new_top_left = pos - self.offset
            old_rect = self.roi_list[self.selected_index]
            new_rect = QRect(new_top_left, old_rect.size()).normalized()
            self.roi_list[self.selected_index] = new_rect
            self.roi_updated.emit(self.selected_index, new_rect)
            self.repaint()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.roi_list.append(rect)
                self.roi_created.emit(rect)
            self.repaint()
        elif self.moving:
            self.moving = False
            self.selected_index = -1

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.frame is not None:
            height, width, channel = self.frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)

            painter = QPainter(pixmap)
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)

            for rect in self.roi_list:
                painter.drawRect(rect)

            if self.drawing and self.start_point and self.end_point:
                temp_rect = QRect(self.start_point, self.end_point).normalized()
                painter.drawRect(temp_rect)

            painter.end()
            self.setPixmap(pixmap)

    def delete_roi(self, index: int):
        if 0 <= index < len(self.roi_list):
            del self.roi_list[index]
            self.repaint()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI 선택, 이동, 삭제 GUI")

        self.video_widget = VideoWidget()
        self.roi_list_widget = ROIListWidget()
        self.capture_thread = VideoCaptureThread()

        # 시그널 연결
        self.capture_thread.frame_captured.connect(self.video_widget.set_frame)
        self.video_widget.roi_created.connect(self.roi_list_widget.add_roi)
        self.video_widget.roi_updated.connect(self.roi_list_widget.update_roi)
        self.roi_list_widget.roi_deleted.connect(self.video_widget.delete_roi)

        # 레이아웃 구성
        layout = QHBoxLayout()
        layout.addWidget(self.video_widget)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.roi_list_widget)
        layout.addLayout(right_layout)

        self.setLayout(layout)

    def closeEvent(self, event):
        self.capture_thread.release()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 600)
    window.show()
    sys.exit(app.exec_())
