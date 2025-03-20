import cv2
import face_recognition
import numpy as np
import os

known_face_encodings = []
known_face_names = []

known_faces_folder = "known_faces" 

for filename in os.listdir(known_faces_folder):
    image_path = os.path.join(known_faces_folder, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if len(encoding) > 0:
        known_face_encodings.append(encoding[0])
        known_face_names.append(filename.split(".")[0])


classroom_image_path = "classroom.jpg"
classroom_image = face_recognition.load_image_file(classroom_image_path)
face_locations = face_recognition.face_locations(classroom_image)  
face_encodings = face_recognition.face_encodings(classroom_image, face_locations)  


classroom_image_cv = cv2.imread(classroom_image_path)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
    name = "Unknown"

    if True in matches:
        matched_index = matches.index(True)
        name = known_face_names[matched_index]

    
    cv2.rectangle(classroom_image_cv, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(classroom_image_cv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)



scale_percent = 25  
width = int(classroom_image_cv.shape[1] * scale_percent / 100)
height = int(classroom_image_cv.shape[0] * scale_percent / 100)
resized_image = cv2.resize(classroom_image_cv, (width, height))


cv2.imshow("Classroom Face Recognition", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()