from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRect, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen

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

    def set_frame(self, frame):
        self.frame = frame.copy()
        self.repaint()

    def mousePressEvent(self, event):
        pos = event.pos()
        for i, rect in enumerate(self.roi_list):
            if rect.contains(pos):
                self.selected_index = i
                self.moving = True
                self.offset = pos - rect.topLeft()
                return
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
            new_rect = QRect(pos - self.offset, self.roi_list[self.selected_index].size()).normalized()
            self.roi_list[self.selected_index] = new_rect
            self.roi_updated.emit(self.selected_index, new_rect)
            self.repaint()

    def mouseReleaseEvent(self, event):
        if self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, event.pos()).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.roi_list.append(rect)
                self.roi_created.emit(rect)
            self.repaint()
        elif self.moving:
            self.moving = False
            self.selected_index = -1

    def delete_roi(self, index: int):
        if 0 <= index < len(self.roi_list):
            del self.roi_list[index]
            self.repaint()

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

        if self.drawing and self.start_point and self.end_point:
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())

        painter.end()
        self.setPixmap(pixmap)
