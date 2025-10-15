import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch import nn
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===========================
# 1️⃣ DEVICE
# ===========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===========================
# 2️⃣ CUSTOM CNN MODEL
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

# ===========================
# 3️⃣ LOAD MODEL FUNCTION
# ===========================
def load_model(model_choice):
    if model_choice == "ResNet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load("/Users/frankllonch/Desktop/quattroporte/aprendizado de máquina/melanomator/models/melanoma_model.pth", map_location=device))
    else:
        model = CustomCNN()
        model.load_state_dict(torch.load("/Users/frankllonch/Desktop/quattroporte/aprendizado de máquina/melanomator/models/melanoma_model3.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

# ===========================
# 4️⃣ TRANSFORM
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===========================
# 5️⃣ GRAD-CAM
# ===========================
def grad_cam(model, image_tensor, class_idx=None):
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    if isinstance(model, CustomCNN):
        target_layer = model.conv3
    else:
        target_layer = model.layer4

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
    loss = output[0, class_idx]
    loss.backward()

    pooled_grads = torch.mean(gradients, dim=[0,2,3])
    features = features[0]
    for i in range(features.shape[0]):
        features[i,:,:] *= pooled_grads[i]

    heatmap = np.mean(features.detach().cpu().numpy(), axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    handle_f.remove()
    handle_b.remove()
    return heatmap

# ===========================
# 6️⃣ STREAMLIT UI
# ===========================
st.set_page_config(page_title="🩺 Melanoma Classifier", layout="wide", page_icon="🩺")
st.title("🩺 Melanoma Classifier")
st.write("Upload skin lesion images to predict **Benign** or **Malignant**.")

# --- Sidebar options ---
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose model:", ["ResNet18", "CustomCNN"])
threshold = st.sidebar.slider("Decision threshold for 'Malignant'", 0.0, 1.0, 0.5, 0.01)
brightness = st.sidebar.slider("Brightness", 0.5, 1.5, 1.0)
contrast = st.sidebar.slider("Contrast", 0.5, 1.5, 1.0)
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)

# --- Apply theme ---
def get_theme_css(dark_mode=True):
    if dark_mode:
        return """
        <style>
        [data-testid="stAppViewContainer"]{background-color:#0e1117;color:#e0e0e0;}
        [data-testid="stSidebar"]{background-color:#0b0d13;color:#e0e0e0;}
        header{background-color:#0e1117 !important;}
        #MainMenu, footer{visibility:hidden;}
        .stButton>button{background-color:#1e1e1e;color:#e0e0e0;border:1px solid #333;}
        .css-1lsmgbg.e1tzin5v3,.css-10trblm.e1tzin5v3,.css-17x69mk.e16nr0p31{color:#e0e0e0 !important;}
        </style>
        """
    else:
        return """
        <style>
        [data-testid="stAppViewContainer"]{background-color:#ffffff;color:#000000;}
        [data-testid="stSidebar"]{background-color:#f0f2f6;color:#000000;}
        header{background-color:#ffffff !important;}
        #MainMenu, footer{visibility:hidden;}
        .stButton>button{background-color:#f0f2f6;color:#000000;border:1px solid #ccc;}
        </style>
        """

st.markdown(get_theme_css(dark_mode), unsafe_allow_html=True)

# --- Load model ---
model = load_model(model_choice)

# --- File uploader ---
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        # Apply brightness/contrast
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        # Preprocess
        img_t = transform(image).unsqueeze(0).to(device)
        img_t.requires_grad_()

        # Prediction
        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)
            malignant_prob = probs[0][1].item()

        prediction = "Malignant" if malignant_prob >= threshold else "Benign"
        color = "🔴" if prediction=="Malignant" else "🟢"
        st.subheader(f"{prediction} {color}")
        st.write(f"Malignant probability: {malignant_prob:.2%} | Threshold: {threshold:.2f}")

        # Confidence bar
        st.bar_chart({"Benign": float(probs[0][0]), "Malignant": float(probs[0][1])})

        # Grad-CAM
        heatmap = grad_cam(model, img_t, class_idx=None)
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap_img = np.uint8(255*heatmap)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        img_np = np.array(image.resize((224,224)))
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_img, 0.4, 0)
        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_container_width=True)