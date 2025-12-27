import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import os

DB_FILE = "encodings.pkl"
ATT_FILE = "attendance.csv"
TOLERANCE = 0.5

# Load trained faces
with open(DB_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

if len(known_encodings) == 0:
    print("NO TRAINED FACES FOUND")
    exit()

marked_today = set()

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")

    if name in marked_today:
        return

    file_exists = os.path.exists(ATT_FILE)
    now = datetime.now()

    with open(ATT_FILE, "a") as f:
        if not file_exists:
            f.write("Name,Date,Time\n")
        f.write(f"{name},{today},{now:%H:%M:%S}\n")

    marked_today.add(name)
    print("ATTENDANCE MARKED:", name)


cap = cv2.VideoCapture(0)

print("LOOK AT CAMERA — PRESS Q TO QUIT")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for enc, (top, right, bottom, left) in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, enc)
        best = np.argmin(distances)

        name = "Unknown"
        confidence = 0

        if distances[best] < TOLERANCE:
            name = known_names[best]
            confidence = int((1 - distances[best]) * 100)
            mark_attendance(name)

        
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame,
            f"{name} {confidence}%",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Attendance (UI Mode)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
