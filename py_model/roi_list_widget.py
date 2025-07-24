from PyQt5.QtWidgets import QListWidget
from PyQt5.QtCore import Qt, pyqtSignal, QRect

class ROIListWidget(QListWidget):
    roi_deleted = pyqtSignal(int)

    def add_roi(self, rect: QRect):
        self.addItem(self._format_roi(rect))

    def update_roi(self, index: int, rect: QRect):
        self.item(index).setText(self._format_roi(rect))

    def update_prediction(self, index: int, result: int):
        item = self.item(index)
        if item:
            text = item.text().split("→")[0].strip()
            item.setText(f"{text} → 예측: {result}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            selected = self.currentRow()
            if selected >= 0:
                self.takeItem(selected)
                self.roi_deleted.emit(selected)
        else:
            super().keyPressEvent(event)

    @staticmethod
    def _format_roi(rect: QRect):
        return f"ROI - x:{rect.x()}, y:{rect.y()}, w:{rect.width()}, h:{rect.height()}"
