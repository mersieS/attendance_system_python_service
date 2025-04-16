import face_recognition
import numpy as np
import cv2
from api_client import send_attendance
import time

recognized_students = set()
last_message = ""
last_message_time = 0

total_checks = 0
correct_matches = 0

def recognize_faces(frame, known_faces, known_names):
    global last_message, last_message_time, total_checks, correct_matches

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        top, right, bottom, left = [v * 4 for v in face_location]
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        total_checks += 1
        if name != "Unknown":
            correct_matches += 1

        accuracy = correct_matches / total_checks if total_checks > 0 else 0
        print(f"Doğruluk oranı: {accuracy:.2%} ({correct_matches}/{total_checks})")

        if name != "Unknown" and name not in recognized_students:
            recognized_students.add(name)
            send_attendance(name)
            last_message = f"{name} recognized, attendance saved."
            last_message_time = time.time()
        elif name == "Unknown":
            last_message = "Face is not recognized."
            last_message_time = time.time()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if time.time() - last_message_time < 3:
        cv2.putText(frame, last_message, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame