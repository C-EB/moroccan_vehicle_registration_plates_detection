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

### 3. Defining Paths and Parameters
```python
weights_path = "yolov4-obj.weights"
configuration_path = "yolov4-obj.cfg"
probability_minimum = 0.5
threshold = 0.3
```
Sets the paths to the YOLO weights and configuration files and defines the minimum probability and threshold for non-maximum suppression.

### 4. Loading YOLO Model
```python
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
```
Loads the YOLO model with the specified configuration and weights.

### 5. Getting Layer Names
```python
layers_names_all = network.getLayerNames()  # list of layers names
print(layers_names_all)
```
Gets and prints all layer names in the YOLO model.
### 6. Output Layers
```python
print(network.getUnconnectedOutLayers())
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
print(layers_names_output)
```
Prints the indices of unconnected output layers and extracts their names. Output:
```css
[327 353 379]
['yolo_139', 'yolo_150', 'yolo_161']
```

