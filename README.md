# Face Recognition Attendance System – Python (Client)

This is the Python-based face recognition client for the attendance system. It captures live video from the webcam, detects and recognizes student faces using a CNN model implemented in PyTorch, and sends attendance data to the Rails API.

---

## 🧠 Features

- Real-time webcam face detection and recognition
- Custom PyTorch CNN model (`cnn_face_pytorch.py`)
- Automatic attendance marking via POST request
- Dataset-based face encoding
- Unknown face detection & warning
- Seamless integration with a Rails API backend

---

## 🛠️ Technologies Used

- Python 3.10+
- OpenCV
- PyTorch
- NumPy
- `requests`
- `pickle`
- `torchvision`

---

## 📦 Setup Instructions

### 1. Clone the Project

```bash
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
face-recognition/
├── main.py                 # Entry point: webcam + recognition + API
├── cnn_face_pytorch.py     # PyTorch CNN model architecture
├── config.py               # API endpoint config
├── dataset/                # Student image folders
├── encodings/              # Saved face embeddings (pickle)
├── utils.py                # Image processing utilities
└── requirements.txt
```

---

## 📸 Dataset Format

Organize images by student folder name (used as label):

```
dataset/
├── salih_buker/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── elif_ayse/
│   ├── ...
```

---

## 🧠 CNN Model – `cnn_face_pytorch.py`

This file defines a lightweight CNN architecture using PyTorch, suitable for low-latency real-time face recognition.

### Model Highlights:

```python
class FaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(FaceRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Training & Saving

Model is trained on grayscale 100x100 cropped face images extracted from the `dataset/` directory. After training, both the model and face label mappings are saved using `torch.save()` and `pickle`.

---

## ▶️ Run the App

```bash
python main.py
```

The app will:
- Launch the webcam
- Detect and recognize faces using the CNN model
- Match face to student
- If matched, send attendance to the Rails API

---

## 🔗 Rails API Integration

Python client sends attendance results directly to your Rails API.

### Sample API Call (inside `main.py`):

```python
import requests

data = {
    "student_number": "20232015",
    "recognized": True
}

requests.post("http://localhost:3000/api/attendances/mark", json=data)
```

You can set this URL dynamically via `config.py`:

```python
API_URL = "http://localhost:3000/api/attendances/mark"
```

---

## 📈 Accuracy & Limitations

| Model           | Accuracy  | Note                    |
|----------------|-----------|-------------------------|
| PyTorch CNN     | ~92%      | Based on test images    |
| LBPH (fallback) | ~78%      | Less robust in practice |

- Accuracy depends on image quality, lighting, and dataset variety
- Avoid training with sunglasses, masks, extreme lighting

---

## 📄 License

MIT License – see [LICENSE](../LICENSE) for full text.

---

## 👨‍💻 Developed by

**Salih İmran Büker**  
Junior Python & Ruby Backend Developer  
📍 Istanbul, Turkey  
📧 salihimranbuker44@gmail.com

**Ali Serhat Aslan**
Software Engineer Student
📍 Istanbul, Turkey 
📧 serhathe0@gmail.com

**Melih Can**
Software Engineer Student
📍 Istanbul, Turkey 
📧 melihcann3@outlook.com