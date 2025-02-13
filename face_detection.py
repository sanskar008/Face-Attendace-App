import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN

image_path = "classroom.jpg"  
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detector = MTCNN()

faces = detector.detect_faces(image_rgb)

for face in faces:
    x, y, width, height = face["box"]
    cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 3)

plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
