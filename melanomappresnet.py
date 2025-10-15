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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the same model as training
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load the weights
model.load_state_dict(torch.load("melanoma_model.pth", map_location=device))
model.to(device)
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