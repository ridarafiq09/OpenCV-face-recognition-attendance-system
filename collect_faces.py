import cv2
import os
import time
import sys

# -------------------------------------------------
# GET STUDENT NAME
# -------------------------------------------------
# If name is passed from Flask → use it
# Else fallback to terminal input (manual run)
if len(sys.argv) >= 2 and sys.argv[1].strip():
    name = sys.argv[1].strip()
else:
    name = input("Enter person name: ").strip()

if not name:
    print("No name provided. Exiting.")
    exit()

# -------------------------------------------------
# CREATE SAVE DIRECTORY
# -------------------------------------------------
save_dir = os.path.join("known_faces", name)
os.makedirs(save_dir, exist_ok=True)

# -------------------------------------------------
# CAMERA SETUP (Windows-friendly)
# -------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# -------------------------------------------------
# FACE DETECTOR
# -------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------------------------------
# CAPTURE SETTINGS
# -------------------------------------------------
count = 0
max_images = 40
last_save_time = 0
save_interval = 0.25  # seconds between saves

print(f"Capturing for: {name}  |  Press Q to quit early")

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Only save when ONE face is detected
    if len(faces) == 1:
        (x, y, w, h) = faces[0]

        # Add padding around face
        pad = int(0.2 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (250, 250))
        
        blur = cv2.Laplacian(face_crop, cv2.CV_64F).var()
        if blur < 100:
         continue


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        now = time.time()
        if now - last_save_time >= save_interval and count < max_images:
            img_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_crop)
            count += 1
            last_save_time = now

    # Display text
    cv2.putText(
        frame,
        f"{name}  Saved: {count}/{max_images}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Faces", frame)

    # Exit conditions
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if count >= max_images:
        break

# -------------------------------------------------
# CLEANUP
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print("Done. Saved to:", save_dir)
