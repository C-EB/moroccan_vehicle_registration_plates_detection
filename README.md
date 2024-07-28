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
### 7. Reading and Displaying the Image

```python
image_input = cv2.imread("WhatsApp Image 2023-10-28 at 18.15.55_8d3d7646.jpg")
image_input_shape = image_input.shape
print(image_input_shape)
plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
plt.show()
```
Reads the input image and displays its shape. Then, it displays the image. Output shape:
```css
(1280, 1024, 3)
```
### 8. Creating Blob from Image
```python
blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
print(image_input.shape)
print(blob.shape)
```
Creates a blob from the image to pass to the YOLO model and prints the shapes. Output:
```css
(1280, 1024, 3)
(1, 3, 416, 416)
```
### 9. Displaying the Blob
```python
blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
print(blob_to_show.shape)
plt.imshow(blob_to_show)
plt.show()
```
Displays the blob. Output shape:
```css
(416, 416, 3)
```

### 10. Forward Pass through YOLO
```python
network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()
print('YOLO v4 took {:.3f} seconds'.format(end - start))
```
Performs a forward pass through the YOLO network and measures the time taken. Output:
```css
YOLO v4 took 1.935 seconds
```
### 11. Processing YOLO Output
```python
print(type(output_from_network))
print(type(output_from_network[0]))
```
Prints the types of the outputs. Output:
```css
<class 'tuple'>
<class 'numpy.ndarray'>
```
### 12. Generating Colors for Classes
```python
np.random.seed(42)
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
print(colours.shape)
print(colours[0])
```
Generates random colors for bounding boxes. Output:
```css
(1, 3)
[102 220 225]
```

### 13. Detecting Objects
```python
bounding_boxes = []
confidences = []
class_numbers = []
h, w = image_input_shape[:2]
print(h, w)
```
Initializes lists for bounding boxes, confidences, and class numbers and prints the image dimensions. 
Output:
```css
1280 1024
```
### 14. Extracting Bounding Boxes and Confidences
```python
for result in output_from_network:
    for detection in result:
        scores = detection[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]
        if confidence_current > probability_minimum:
            box_current = detection[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current.astype('int')
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)
```
Extracts bounding boxes and confidences for detected objects.
### 15. Applying Non-Maximum Suppression
```python
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
```
Applies Non-Maximum Suppression to filter out overlapping bounding boxes.
### 16. Displaying Detected Labels
```python
for i in range(len(class_numbers)):
    print(labels[int(class_numbers[i])])
with open('found_labels.txt', 'w') as f:
    for i in range(len(class_numbers)):
        f.write(labels[int(class_numbers[i])])
```
Prints and saves detected labels. 
Output:
```css
license_plate
license_plate
```
### 17. Drawing Bounding Boxes
```python
if len(results) > 0:
    for i in results.flatten():
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        colour_box_current = [int(j) for j in colours[class_numbers[i]]]
        cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)
        text_box_current = '{} : {:.2f}%'.format(labels[int(class_numbers[i])], confidences[i])
        cv2.putText(image_input, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```
Draws bounding boxes and labels on the image and displays it.

