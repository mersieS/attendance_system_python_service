import requests
import os

API_URL = "http://localhost:3000/api/students"
SAVE_DIR = "known_faces"

def fetch_and_save_students():
    os.makedirs(SAVE_DIR, exist_ok=True)

    response = requests.get(API_URL)
    if response.status_code != 200:
        print("❌ Öğrenci listesi alınamadı.")
        return

    students = response.json()

    for student in students:
        name = student["name"].strip().replace(" ", "_")
        filename = f"{name}.jpg"
        path = os.path.join(SAVE_DIR, filename)

        if os.path.exists(path):
            print(f"✅ {filename} zaten var, atlanıyor.")
            continue

        photo_url = student.get("photo_url")
        if not photo_url:
            print(f"⚠️ {name} öğrencisinin fotoğrafı yok.")
            continue

        try:
            img_data = requests.get(photo_url).content
            with open(path, "wb") as f:
                f.write(img_data)
            print(f"📥 {filename} downloaded.")
        except Exception as e:
            print(f"For {name} photo not downloaded: {e}")

if __name__ == "__main__":
    fetch_and_save_students()
