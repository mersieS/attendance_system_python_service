import cv2
from recognizer import recognize_faces, recognized_students
from utils import load_known_faces
from api_client import send_absence

def start_camera():
    print("Cam is starting")
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Cam is not open")
        return

    print("Cam is open")

    known_faces, known_names = load_known_faces()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = recognize_faces(frame, known_faces, known_names)

        cv2.imshow("Yoklama Kamerasi", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

    absent_students = set(known_names) - recognized_students
    print(f"\n📋 {len(absent_students)} student is not come:")
    for name in absent_students:
        print(" -", name.replace('_', ' ').title())
        send_absence(name)