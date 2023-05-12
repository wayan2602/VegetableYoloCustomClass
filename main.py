import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

# load the YOLOv5 model from the .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/My Project/Project_1/best.pt')

# define the classes
classes = ["bawang merah", "bawang putih", "bombay", "brokoli", "buah bit", "buncis", "daun bawang", "jagung", "kembang kol", "kentang", "pakcoy", "paprika", "terong", "tomat", "ubi", "wortel"]

# load an image
image_path = 'D:/My Project/Project_1/nama_sayuran.jpeg'
img = cv2.imread(image_path)

# make a prediction on the image
results = model(img, size=640)

# display the results
results.print()
results.show()

# extract the bounding boxes, labels, and confidences
boxes = results.xyxy[0].numpy()
labels = results.pred[0].numpy()[:, 5]
confidences = results.pred[0].numpy()[:, 4]

# filter out low-confidence detections
threshold = 0.9
mask = confidences > threshold
boxes = boxes[mask]
labels = labels[mask]

for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = box[:4].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, classes[int(label)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# convert the color space from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show the annotated image
plt.imshow(img)
plt.show()