import face_recognition
import os

def load_known_faces():
    known_faces = []
    known_names = []
    directory = "known_faces"

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)

            if encoding:
                known_faces.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"{filename} içinde yüz bulunamadı!")

    print("🎓 Tüm yüz verileri yüklendi.")
    return known_faces, known_names
