# src/realtime_recognition.py

import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import joblib
from insightface.app import FaceAnalysis

# Paths
MODEL_FILE = "../models/svm_model.pkl"

# Load classifier
svm_clf = joblib.load(MODEL_FILE)

# Initialize ArcFace and MTCNN
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(112, 112))  # -1 for CPU, 0 for GPU
detector = MTCNN()
IMAGE_SIZE = (112, 112)

# Threshold for unknown faces (distance-based)
THRESHOLD = 0.7

def align_face(image, keypoints):
    left_eye = tuple(map(float, keypoints["left_eye"]))
    right_eye = tuple(map(float, keypoints["right_eye"]))
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated

# Start webcam
cap = cv2.VideoCapture(0)

print("🎥 Starting real-time recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    detections = detector.detect_faces(frame)

    for det in detections:
        x, y, w, h = det["box"]
        keypoints = det["keypoints"]

        # Align face
        aligned = align_face(frame, keypoints)
        face_crop = aligned[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, IMAGE_SIZE)

        # Convert to embedding
        face_img = np.array(Image.fromarray(face_resized).convert('RGB'))
        faces = app.get(face_img)
        if len(faces) == 0:
            continue
        embedding = faces[0].embedding.reshape(1, -1)

        # Predict
        probs = svm_clf.predict_proba(embedding)
        pred_idx = np.argmax(probs)
        pred_label = svm_clf.classes_[pred_idx]
        confidence = probs[0][pred_idx]

        if confidence < THRESHOLD:
            pred_label = "Unknown"

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{pred_label} ({confidence:.2f})", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
