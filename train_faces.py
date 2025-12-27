import os
import pickle
import face_recognition

KNOWN_DIR = "known_faces"
OUT_FILE = "encodings.pkl"

encodings = []
names = []

person_encodings = []


for person in os.listdir(KNOWN_DIR):
    person_path = os.path.join(KNOWN_DIR, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        image = face_recognition.load_image_file(img_path)
        locations = face_recognition.face_locations(image, model="hog")

        if len(locations) != 1:
            continue

        encoding = face_recognition.face_encodings(image, locations)[0]
        person_encodings.append(encoding)

        if person_encodings:
         mean_encoding = sum(person_encodings) / len(person_encodings)
         encodings.append(mean_encoding)
         names.append(person)


with open(OUT_FILE, "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print(f"Training done. Total samples: {len(encodings)}")
print("Saved:", OUT_FILE)
