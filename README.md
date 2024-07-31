# Pathhole_detection

# YOLO Video and Image Processor

This application is a video and image processing tool using PyQt5 for the graphical user interface and YOLO for Pathhole detection. The application allows users to drag and drop or browse video and image files for processing, and then it runs the YOLO model to detect objects in the media.

## Features

- Drag and drop or browse to select video and image files.
- Process selected video or image using YOLO object detection.
- Display the processed media with detected objects highlighted.
- Save the processed video or image to an output file.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.6+
- OpenCV
- numpy
- PyQt5
- ultralyticsplus

You can install the necessary packages using pip:

```sh
pip install opencv-python-headless numpy PyQt5 ultralyticsplus
