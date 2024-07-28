# License Plate Detection using YOLOv4

This project aims to detect Moroccan vehicle license plates using the YOLOv4 model. The Python code provided reads an image, processes it using the YOLOv4 model, and identifies and marks the license plates present in the image.

## Dataset

The dataset consists of images of Moroccan vehicle license plates. Each image is accompanied by a file that specifies the position of the plate in YOLO format. The images were labeled using LabelImg, a graphical image annotation tool.

## YOLO Model

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. YOLOv4, the model used in this project, is an improvement over previous versions and offers enhanced accuracy and speed.

## Code Explanation

### 1. Importing Libraries

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
```
This cell imports necessary libraries: numpy for numerical operations, cv2 for image processing, matplotlib for plotting, and time for measuring execution time.

### 2. Reading Class Labels
```python
labels = open("obj.names").read().strip().split('\n')  # list of class names
print(labels)
```
Reads class names from the obj.names file and prints them. Output:
```css
['license_plate']
```
