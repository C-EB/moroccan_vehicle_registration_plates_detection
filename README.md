# License Plate Detection using YOLOv4

This project aims to detect Moroccan vehicle license plates using the YOLOv4 model. The Python code provided reads an image, processes it using the YOLOv4 model, and identifies and marks the license plates present in the image.

## Dataset

The dataset consists of images of Moroccan vehicle license plates. Each image is accompanied by a file that specifies the position of the plate in YOLO format. The images were labeled using LabelImg, a graphical image annotation tool.

## YOLO Model

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. YOLOv4, the model used in this project, is an improvement over previous versions and offers enhanced accuracy and speed.

### YOLO Model Comparison

| Model Version | Description | Speed (FPS) | mAP (Mean Average Precision) |
|---------------|-------------|-------------|------------------------------|
| YOLOv1        | Initial version with basic features | ~45 FPS | ~63.4% |
| YOLOv2        | Improved accuracy and speed, multi-scale training | ~40 FPS | ~76.8% |
| YOLOv3        | Further improvements in accuracy, deeper architecture | ~30 FPS | ~80.0% |
| YOLOv4        | Latest version with advanced features like CSPDarknet53 | ~50 FPS | ~89.7% |


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
Reads class names from the obj.names file and prints them. 
Output:
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
Output:
```css
('conv_0', 'bn_0', 'mish_1', 'conv_1', 'bn_1', 'mish_2', 'conv_2', 'bn_2', 'mish_3', 'identity_3', 'conv_4', 'bn_4', 'mish_5', 'conv_5', 'bn_5', 'mish_6', 'conv_6', 'bn_6', 'mish_7', 'shortcut_7', 'conv_8', 'bn_8', 'mish_9', 'concat_9', 'conv_10', 'bn_10', 'mish_11', 'conv_11', 'bn_11', 'mish_12', 'conv_12', 'bn_12', 'mish_13', 'identity_13', 'conv_14', 'bn_14', 'mish_15', 'conv_15', 'bn_15', 'mish_16', 'conv_16', 'bn_16', 'mish_17', 'shortcut_17', 'conv_18', 'bn_18', 'mish_19', 'conv_19', 'bn_19', 'mish_20', 'shortcut_20', 'conv_21', 'bn_21', 'mish_22', 'concat_22', 'conv_23', 'bn_23', 'mish_24', 'conv_24', 'bn_24', 'mish_25', 'conv_25', 'bn_25', 'mish_26', 'identity_26', 'conv_27', 'bn_27', 'mish_28', 'conv_28', 'bn_28', 'mish_29', 'conv_29', 'bn_29', 'mish_30', 'shortcut_30', 'conv_31', 'bn_31', 'mish_32', 'conv_32', 'bn_32', 'mish_33', 'shortcut_33', 'conv_34', 'bn_34', 'mish_35', 'conv_35', 'bn_35', 'mish_36', 'shortcut_36', 'conv_37', 'bn_37', 'mish_38', 'conv_38', 'bn_38', 'mish_39', 'shortcut_39', 'conv_40', 'bn_40', 'mish_41', 'conv_41', 'bn_41', 'mish_42', 'shortcut_42', 'conv_43', 'bn_43', 'mish_44', 'conv_44', 'bn_44', 'mish_45', 'shortcut_45', 'conv_46', 'bn_46', 'mish_47', 'conv_47', 'bn_47', 'mish_48', 'shortcut_48', 'conv_49', 'bn_49', 'mish_50', 'conv_50', 'bn_50', 'mish_51', 'shortcut_51', 'conv_52', 'bn_52', 'mish_53', 'concat_53', 'conv_54', 'bn_54', 'mish_55', 'conv_55', 'bn_55', 'mish_56', 'conv_56', 'bn_56', 'mish_57', 'identity_57', 'conv_58', 'bn_58', 'mish_59', 'conv_59', 'bn_59', 'mish_60', 'conv_60', 'bn_60', 'mish_61', 'shortcut_61', 'conv_62', 'bn_62', 'mish_63', 'conv_63', 'bn_63', 'mish_64', 'shortcut_64', 'conv_65', 'bn_65', 'mish_66', 'conv_66', 'bn_66', 'mish_67', 'shortcut_67', 'conv_68', 'bn_68', 'mish_69', 'conv_69', 'bn_69', 'mish_70', 'shortcut_70', 'conv_71', 'bn_71', 'mish_72', 'conv_72', 'bn_72', 'mish_73', 'shortcut_73', 'conv_74', 'bn_74', 'mish_75', 'conv_75', 'bn_75', 'mish_76', 'shortcut_76', 'conv_77', 'bn_77', 'mish_78', 'conv_78', 'bn_78', 'mish_79', 'shortcut_79', 'conv_80', 'bn_80', 'mish_81', 'conv_81', 'bn_81', 'mish_82', 'shortcut_82', 'conv_83', 'bn_83', 'mish_84', 'concat_84', 'conv_85', 'bn_85', 'mish_86', 'conv_86', 'bn_86', 'mish_87', 'conv_87', 'bn_87', 'mish_88', 'identity_88', 'conv_89', 'bn_89', 'mish_90', 'conv_90', 'bn_90', 'mish_91', 'conv_91', 'bn_91', 'mish_92', 'shortcut_92', 'conv_93', 'bn_93', 'mish_94', 'conv_94', 'bn_94', 'mish_95', 'shortcut_95', 'conv_96', 'bn_96', 'mish_97', 'conv_97', 'bn_97', 'mish_98', 'shortcut_98', 'conv_99', 'bn_99', 'mish_100', 'conv_100', 'bn_100', 'mish_101', 'shortcut_101', 'conv_102', 'bn_102', 'mish_103', 'concat_103', 'conv_104', 'bn_104', 'mish_105', 'conv_105', 'bn_105', 'leaky_106', 'conv_106', 'bn_106', 'leaky_107', 'conv_107', 'bn_107', 'leaky_108', 'pool_108', 'identity_109', 'pool_110', 'identity_111', 'pool_112', 'concat_113', 'conv_114', 'bn_114', 'leaky_115', 'conv_115', 'bn_115', 'leaky_116', 'conv_116', 'bn_116', 'leaky_117', 'conv_117', 'bn_117', 'leaky_118', 'upsample_118', 'identity_119', 'conv_120', 'bn_120', 'leaky_121', 'concat_121', 'conv_122', 'bn_122', 'leaky_123', 'conv_123', 'bn_123', 'leaky_124', 'conv_124', 'bn_124', 'leaky_125', 'conv_125', 'bn_125', 'leaky_126', 'conv_126', 'bn_126', 'leaky_127', 'conv_127', 'bn_127', 'leaky_128', 'upsample_128', 'identity_129', 'conv_130', 'bn_130', 'leaky_131', 'concat_131', 'conv_132', 'bn_132', 'leaky_133', 'conv_133', 'bn_133', 'leaky_134', 'conv_134', 'bn_134', 'leaky_135', 'conv_135', 'bn_135', 'leaky_136', 'conv_136', 'bn_136', 'leaky_137', 'conv_137', 'bn_137', 'leaky_138', 'conv_138', 'permute_139', 'yolo_139', 'identity_140', 'conv_141', 'bn_141', 'leaky_142', 'concat_142', 'conv_143', 'bn_143', 'leaky_144', 'conv_144', 'bn_144', 'leaky_145', 'conv_145', 'bn_145', 'leaky_146', 'conv_146', 'bn_146', 'leaky_147', 'conv_147', 'bn_147', 'leaky_148', 'conv_148', 'bn_148', 'leaky_149', 'conv_149', 'permute_150', 'yolo_150', 'identity_151', 'conv_152', 'bn_152', 'leaky_153', 'concat_153', 'conv_154', 'bn_154', 'leaky_155', 'conv_155', 'bn_155', 'leaky_156', 'conv_156', 'bn_156', 'leaky_157', 'conv_157', 'bn_157', 'leaky_158', 'conv_158', 'bn_158', 'leaky_159', 'conv_159', 'bn_159', 'leaky_160', 'conv_160', 'permute_161', 'yolo_161')
```

### 6. Output Layers
```python
print(network.getUnconnectedOutLayers())
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
print(layers_names_output)
```
Identify the names of the output layers required by the YOLO algorithm.
Output:
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
Reads the input image and displays its shape. Then, it displays the image. 
Output shape:
```css
(1280, 1024, 3)
```

==================================image
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
========================================================bolb image
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

====================image ouput 
