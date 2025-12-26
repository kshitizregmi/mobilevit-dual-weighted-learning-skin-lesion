import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .download-btn {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = (weights * self.activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx

def overlay_cam(img, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Better overlay with proper blending"""
    h, w = img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    
    # Smooth the cam
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=2, sigmaY=2)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Better blending: multiply mode for better visibility
    heatmap_float = heatmap.astype(float) / 255
    img_float = img.astype(float) / 255
    
    # Blend with the original image visible underneath
    overlay = img_float * (1 - alpha * cam[..., None]) + heatmap_float * alpha * cam[..., None]
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file = Path(model_path)
    
    if not model_file.exists():
        return None, None, None, None
    
    ckpt = torch.load(model_file, map_location=device)
    model_name = ckpt.get('model_name', 'mobilevitv2_100')
    classes = ckpt['classes']
    
    model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    
    # Better target layer selection for MobileViT
    target_layer = None
    if hasattr(model, 'stages'):
        # Get the last stage
        target_layer = model.stages[-1]
    elif hasattr(model, 'blocks'):
        target_layer = model.blocks[-1]
    else:
        # Fallback: find last conv
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
    
    return model, classes, device, target_layer


def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize(int(256 * 1.14)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


st.markdown('<h1 class="big-title">Skin Lesion Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">MobileViT-V2 dermoscopic analysis with explainable AI predictions</p>', unsafe_allow_html=True)

# Define available models
AVAILABLE_MODELS = {
    "Model 1 (best_skin_model_wf1.pth)": "models/best_skin_model_wf1.pth",
    "Model 2 (2best_skin_model_wf1.pth)": "models/2best_skin_model_wf1.pth"
}

# Sidebar controls
with st.sidebar:
    st.header("Model Selection")
    selected_model_name = st.selectbox(
        "Choose Model",
        list(AVAILABLE_MODELS.keys()),
        help="Select which trained model to use for predictions"
    )
    selected_model_path = AVAILABLE_MODELS[selected_model_name]
    
    st.markdown("---")
    st.header("Visualization Settings")
    overlay_alpha = st.slider("Overlay Intensity", 0.0, 1.0, 0.35, 0.05)
    colormap_opt = st.selectbox("Colormap", ["JET", "TURBO", "VIRIDIS", "PLASMA", "HOT"])
    show_heatmap = st.checkbox("Show pure heatmap", value=False)
    
    colormap_dict = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "HOT": cv2.COLORMAP_HOT
    }

# Load selected model
model, classes, device, target_layer = load_model(selected_model_path)

if model is None:
    st.error(f"Model file '{selected_model_path}' not found. Please ensure the model file exists.")
    st.info("Available model files should be in the same directory as this script.")
    st.stop()

# Display model info
st.info(f"Using: **{selected_model_name}** | Classes: {len(classes)} | Device: {device}")

# Upload
uploaded = st.file_uploader("Upload dermoscopy image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    
    # Predict
    input_tensor = preprocess(image).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, pred_idx = probs.max(0)
    
    predicted_class = classes[pred_idx.item()]
    confidence = conf.item()
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### Grad-CAM Attention Map")
        if target_layer:
            try:
                gradcam = GradCAM(model, target_layer)
                cam, _ = gradcam.generate_cam(input_tensor, pred_idx.item())
                img_np = np.array(image.resize((256, 256)))
                
                if show_heatmap:
                    # Pure heatmap
                    h, w = img_np.shape[:2]
                    cam_resized = cv2.resize(cam, (w, h))
                    cam_resized = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=2, sigmaY=2)
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap_dict[colormap_opt])
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    st.image(heatmap, use_container_width=True)
                else:
                    # Overlay
                    overlay = overlay_cam(img_np, cam, alpha=overlay_alpha, colormap=colormap_dict[colormap_opt])
                    st.image(overlay, use_container_width=True)
                
                st.caption("Red areas = High attention | Blue areas = Low attention")
                
                # Download button
                from io import BytesIO
                overlay_final = overlay_cam(img_np, cam, alpha=overlay_alpha, colormap=colormap_dict[colormap_opt])
                overlay_pil = Image.fromarray(overlay_final)
                buf = BytesIO()
                overlay_pil.save(buf, format='PNG')
                st.download_button(
                    label="Download Attention Map",
                    data=buf.getvalue(),
                    file_name=f"attention_map_{predicted_class}.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {str(e)}")
        else:
            st.info("Grad-CAM not available for this model")
    
  
    st.markdown("---")
    
    
    col3, col4, col5 = st.columns([2, 2, 1])
    
    with col3:
        st.metric("Predicted Class", predicted_class)
    
    with col4:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col5:
        if confidence > 0.7:
            st.success("High")
        elif confidence > 0.5:
            st.warning("Medium")
        else:
            st.error("Low")
    

    st.markdown("### Top 5 Predictions")
    top_k = min(5, len(classes))
    top_probs, top_idx = probs.topk(top_k)
    
    for i, (idx, prob) in enumerate(zip(top_idx, top_probs)):
        cls = classes[idx.item()]
        p = prob.item()
        if i == 0:
            st.progress(p, text=f"**{cls}** — {p:.1%}")
        else:
            st.progress(p, text=f"{cls} — {p:.1%}")

else:
    st.info("Upload an image to begin analysis")
    
   
    with st.expander("How to use"):
        st.markdown("""
        1. **Select a model** from the sidebar dropdown
        2. Click **Browse files** above to upload a dermoscopy image
        3. View the **prediction** and **confidence score**
        4. Check the **attention map** to see what the model focused on
        5. Review **top 5 predictions** for additional insights
        6. Adjust visualization settings in the sidebar
        7. Switch between models to compare predictions
        
        **Note:** This is a research tool. Always consult medical professionals for diagnosis.
        """)

st.markdown("---")
st.caption(f"Active Model: {selected_model_name} • {len(classes)} classes • Device: {device} • Framework: PyTorch + Timm")