import torch
from torchvision import transforms
from PIL import Image
import os
from cnn_face_pytorch import FaceCNN

IMG_SIZE = 128
MODEL_PATH = "face_cnn_model.pth"
CLASS_NAMES = sorted(os.listdir("face_dataset"))

model = FaceCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def predict_face(frame):
    try:
        image = Image.fromarray(frame).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_name = CLASS_NAMES[predicted_idx]

        return predicted_name
    except Exception as e:
        print("Tahmin hatası:", e)
        return None
