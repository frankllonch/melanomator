import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

# ===========================
# 2️⃣ DEFINE THE SAME MODEL
# ===========================
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop4(x)
        logits = self.fc2(x)
        return logits

# === 2. Load Model ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CustomCNN()
model.to(device)
model.load_state_dict(torch.load("/Users/frankllonch/Desktop/quattroporte/aprendizado de máquina/melanomator/models/melanoma_model3.pth", map_location=device))
model.eval()

# === 3. Define Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===========================
# 4️⃣ STREAMLIT UI
# ===========================
st.title("🩺 Melanoma Classifier")
st.write("Upload a skin lesion image to predict if it's **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Threshold slider ---
st.sidebar.header("⚙️ Classification Settings")
threshold = st.sidebar.slider(
    "Decision threshold for 'Malignant' (lower = more recall, higher = more precision)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_t = transform(image).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        malignant_prob = probs[0][1].item()  # Probability of class 1 (Malignant)

    # Apply threshold decision rule
    if malignant_prob >= threshold:
        prediction = "Malignant"
        st.error(f"⚠️ Prediction: {prediction}")
    else:
        prediction = "Benign"
        st.success(f"✅ Prediction: {prediction}")

    # Display probabilities and chosen threshold
    st.write(f"**Malignant probability:** {malignant_prob:.2%}")
    st.write(f"**Decision threshold:** {threshold:.2f}")