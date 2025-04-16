import requests
from datetime import datetime

API_URL = "http://localhost:3000/api/attendances"

def send_attendance(student_name):
    formatted_name = student_name.replace("_", " ").title()
    print(f"✅ {formatted_name} is detected!!!")

    try:
        response = requests.post(API_URL, json={
            "student_name": formatted_name,
            "attended_at": datetime.now().isoformat(),
            "status": True
        })
        print("🛰️ API response:", response.status_code)
    except Exception as e:
        print("API error:", e)

def send_absence(student_name):
    formatted_name = student_name.replace("_", " ").title()
    print(f"{formatted_name} being marked as absent")

    try:
        response = requests.post(API_URL, json={
            "student_name": formatted_name,
            "attended_at": datetime.now().isoformat(),
            "status": False
        })
        print("🛰️ API response:", response.status_code)
    except Exception as e:
        print("API error:", e)
