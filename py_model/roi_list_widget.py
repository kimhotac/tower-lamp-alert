from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QRect

class ROIListWidget(QListWidget):
    roi_deleted = pyqtSignal(int)

    def add_roi(self, rect: QRect):
        item = QListWidgetItem()
        widget = self._create_item_widget(rect, item)
        self.addItem(item)
        self.setItemWidget(item, widget)

    def update_roi(self, index: int, rect: QRect):
        item = self.item(index)
        widget = self.itemWidget(item)
        if widget:
            widget.label.setText(self._format_roi(rect))

    def update_prediction(self, index: int, result: int):
        item = self.item(index)
        widget = self.itemWidget(item)
        if widget:
            base_text = widget.label.text().split("→")[0].strip()
            widget.label.setText(f"{base_text} → 예측: {result}")

    def _create_item_widget(self, rect: QRect, list_item: QListWidgetItem):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)

        label = QLabel(self._format_roi(rect))
        delete_btn = QPushButton("삭제")
        delete_btn.setFixedWidth(50)

        # 삭제 버튼 클릭 시 해당 항목 삭제
        def on_delete():
            index = self.row(list_item)
            self.takeItem(index)
            self.roi_deleted.emit(index)

        delete_btn.clicked.connect(on_delete)

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(delete_btn)

        widget.setLayout(layout)
        widget.label = label  # 후속 접근용 저장
        return widget

    @staticmethod
    def _format_roi(rect: QRect):
        return f"ROI - x:{rect.x()}, y:{rect.y()}, w:{rect.width()}, h:{rect.height()}"
