import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow,QWidget,QGridLayout,QLabel,QScrollArea,QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt

class ScrollableImageGrid(QMainWindow):
    def __init__(self, images, window_name="Scrollable Face Recognition Results"):
        super().__init__()
        self.setWindowTitle(window_name)
        self.showMaximized()
        self.images = images
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.display_images_in_grid()

        self.scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll_area)

    def display_images_in_grid(self, padding=10, target_size=(700, 700)):
        if not self.images:
            return

        num_images = len(self.images)
        num_cols = int(np.ceil(np.sqrt(num_images)))

        for i, img in enumerate(self.images):
            pixmap = self.create_uniform_pixmap(img, target_size)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setFixedSize(target_size[0], target_size[1])
            image_label.setStyleSheet("border: 1px solid #ddd; margin: 5px;")

            row = i // num_cols
            col = i % num_cols

            self.grid_layout.addWidget(image_label, row, col)

    @staticmethod
    def create_uniform_pixmap(cv_image, target_size):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        scale_w = target_size[0] / w
        scale_h = target_size[1] / h
        scale_factor = min(scale_w, scale_h)

        if scale_factor < 1.0:
            resized_w = int(w * scale_factor)
            resized_h = int(h * scale_factor)
            resized_image = cv2.resize(rgb_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        else:
            resized_image = rgb_image
            resized_w = w
            resized_h = h

        qt_image = QImage(resized_image.data, resized_w, resized_h, resized_w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        final_pixmap = QPixmap(target_size[0], target_size[1])
        final_pixmap.fill(Qt.white)
        painter = QPainter(final_pixmap)
        x_offset = (target_size[0] - resized_w) // 2
        y_offset = (target_size[1] - resized_h) // 2
        painter.drawPixmap(x_offset, y_offset, pixmap)
        painter.end()
        return final_pixmap