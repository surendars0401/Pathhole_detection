import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QBrush, QPen

from ultralyticsplus import YOLO, render_result


class VideoProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.model = YOLO('path_hole.pt')
        self.model.overrides['conf'] = 0.25  
        self.model.overrides['iou'] = 0.45 
        self.model.overrides['agnostic_nms'] = False  
        self.model.overrides['max_det'] = 1000  

        self.video_path = ''
        self.image_path = ''

    def initUI(self):
        self.setWindowTitle('YOLO Video Processor')
        self.setFixedSize(1000, 745)

        self.layout = QVBoxLayout()

        self.label = QLabel('Drag and drop a video or image file or click to browse')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Helvetica Neue", 14))
        self.label.setStyleSheet("color: #333; padding: 20px;")
        self.layout.addWidget(self.label)

        self.button_layout = QHBoxLayout()

        self.browse_video_button = QPushButton('Browse Video')
        self.browse_video_button.setFont(QFont("Helvetica Neue", 12))
        self.browse_video_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(10px);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
        """)
        self.browse_video_button.clicked.connect(self.browse_video)
        self.button_layout.addWidget(self.browse_video_button)

        self.browse_image_button = QPushButton('Browse Image')
        self.browse_image_button.setFont(QFont("Helvetica Neue", 12))
        self.browse_image_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(10px);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
        """)
        self.browse_image_button.clicked.connect(self.browse_image)
        self.button_layout.addWidget(self.browse_image_button)

        self.run_button = QPushButton('Run')
        self.run_button.setFont(QFont("Helvetica Neue", 12))
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(10px);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
        """)
        self.run_button.clicked.connect(self.run_model)
        self.button_layout.addWidget(self.run_button)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.setAcceptDrops(True)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                backdrop-filter: blur(10px);
            }
            QLabel {
                background-color: rgba(255, 255, 255, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
                self.video_path = file_path
                self.show_thumbnail(self.video_path)
            elif file_path.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                self.image_path = file_path
                self.show_thumbnail(self.image_path, is_video=False)

    def browse_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "All Files (*);;MP4 Files (*.mp4)",
                                                   options=options)
        if file_name:
            self.video_path = file_name
            self.show_thumbnail(self.video_path)

    def browse_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                   "All Files (*);;Image Files (*.jpg;*.jpeg;*.png;*.bmp)",
                                                   options=options)
        if file_name:
            self.image_path = file_name
            self.show_thumbnail(self.image_path, is_video=False)

    def show_thumbnail(self, file_path, is_video=True):
        if is_video:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, _ = frame.shape
                    qimg = QImage(frame.data, width, height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                cap.release()
        else:
            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, _ = image.shape
                qimg = QImage(image.data, width, height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_model(self):
        if self.video_path:
            self.process_video()
        elif self.image_path:
            self.process_image()
        else:
            self.label.setText('No video or image loaded. Please load a video or image first.')

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.label.setText('Error: Could not open video.')
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = 'output.mp4'
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame)
            render = render_result(model=self.model, image=frame, result=results[0])
            output_frame = np.array(render)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            out.write(output_frame)

            qimg = QImage(output_frame.data, output_frame.shape[1], output_frame.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.label.setText(f"Processed video saved as {output_path}")

    def process_image(self):
        image = cv2.imread(self.image_path)
        if image is None:
            self.label.setText('Error: Could not open image.')
            return

        results = self.model.predict(image)
        render = render_result(model=self.model, image=image, result=results[0])
        output_image = np.array(render)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        qimg = QImage(output_image.data, output_image.shape[1], output_image.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        output_path = 'output.jpg'
        cv2.imwrite(output_path, output_image)

        self.label.setText(f"Processed image saved as {output_path}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())
