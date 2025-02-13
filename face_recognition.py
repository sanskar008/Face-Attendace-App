import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Load MTCNN detector & FaceNet model
detector = MTCNN()
facenet = FaceNet()

# Function to extract face embeddings from an image
def get_face_embedding(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None, None

    x, y, width, height = faces[0]["box"]
    face = image_rgb[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    
    return facenet.embeddings(face)[0], os.path.basename(image_path).split(".")[0]  # Return embedding & name

# Step 1: Load known faces
known_faces = {}

# Folder containing individual student images
known_faces_folder = "known_faces"  # Change to your folder path

for filename in os.listdir(known_faces_folder):
    image_path = os.path.join(known_faces_folder, filename)
    embedding, name = get_face_embedding(image_path)
    
    if embedding is not None:
        known_faces[name] = embedding

# Step 2: Detect & Recognize faces in classroom image
classroom_image_path = "classroom.jpg"  # Change to your image path
classroom_image = cv2.imread(classroom_image_path)
classroom_rgb = cv2.cvtColor(classroom_image, cv2.COLOR_BGR2RGB)

faces = detector.detect_faces(classroom_rgb)

# Step 3: Compare & Recognize
for face in faces:
    x, y, width, height = face["box"]
    face_embedding, _ = get_face_embedding(classroom_image_path)

    if face_embedding is None:
        continue

    min_dist = float("inf")
    identity = "Unknown"

    for name, known_embedding in known_faces.items():
        dist = np.linalg.norm(face_embedding - known_embedding)
        if dist < 0.8:  # Threshold for recognition
            min_dist = dist
            identity = name

    # Draw rectangle & label
    cv2.rectangle(classroom_rgb, (x, y), (x + width, y + height), (0, 255, 0), 3)
    cv2.putText(classroom_rgb, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Step 4: Display Image with Recognition
plt.figure(figsize=(12, 8))
plt.imshow(classroom_rgb)
plt.axis("off")
plt.show()
