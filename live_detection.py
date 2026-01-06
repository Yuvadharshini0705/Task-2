import cv2
import os
import numpy as np
from datetime import datetime

# ---------- PATHS ----------
KNOWN_DIR = "known_faces"
CAPTURE_DIR = "captured"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(CAPTURE_DIR, exist_ok=True)

# ---------- LOAD HAAR CASCADE ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- CREATE RECOGNIZER ----------
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
names = {}

label_id = 0

# ---------- LOAD KNOWN FACES ----------
for file in os.listdir(KNOWN_DIR):
    img_path = os.path.join(KNOWN_DIR, file)
    name = os.path.splitext(file)[0]

    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        faces.append(face)
        labels.append(label_id)
        names[label_id] = name
        label_id += 1

if len(faces) == 0:
    print("No face found in known_faces folder")
    exit()

recognizer.train(faces, np.array(labels))
print("Known faces trained successfully")

# ---------- ATTENDANCE FUNCTION ----------
def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    now = datetime.now()
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")

# ---------- START WEBCAM ----------
cap = cv2.VideoCapture(0)

recognized_once = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in detected_faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 65:
            name = names[label]
            text = f"✔ {name}"
            color = (0, 255, 0)

            if not recognized_once:
                # 📸 CAPTURE PHOTO
                cv2.imwrite(os.path.join(CAPTURE_DIR, f"{name}.jpg"), frame)

                # 📝 ATTENDANCE
                mark_attendance(name)

                recognized_once = True

        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
