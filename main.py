from camera import start_camera
from fetch_and_save_students import fetch_and_save_students
import time

if __name__ == "__main__":
  fetch_and_save_students()
  time.sleep(1)
  start_camera()