import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyttsx3
import time

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voices", voices[0].id)


def speak(text):
    engine.say(text)
    engine.runAndWait()


# load the YOLOv5 model from the .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='model/best.pt')

# define the classes
classes = ["bawang merah", "bawang putih", "bombay", "brokoli", "buah bit", "buncis", "daun bawang",
           "jagung", "kembang kol", "kentang", "pakcoy", "paprika", "terong", "tomat", "ubi", "wortel"]

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame, size=320)
    results.print()
    boxes = results.xyxy[0].numpy()
    labels = results.pred[0].numpy()[:, 5]
    confidences = results.pred[0].numpy()[:, 4]

    threshold = 0.9
    mask = confidences > threshold
    boxes = boxes[mask]
    labels = labels[mask]

    for box, label in enumerate(boxes):
        x1, y1, x2, y2 = box[:4].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, classes[int(label)], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    detection_str = str(results)
    object_str = detection_str[detection_str.find(
        ":")+2: detection_str.find("\n")]

    object_str = object_str.split(" ")[1:]

    cv2.imshow('Video', frame)

    print(" ".join(object_str))
    speak(" ".join(object_str))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
