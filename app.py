import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, ImageFilter
import time

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_sign_model():
    try:
        return load_model("sign_language_model.h5")
    except:
        st.error("Model file not found. Please ensure 'sign_language_model.h5' is in the correct directory.")
        return None

model = load_sign_model()
if model:
    input_shape = model.input_shape[1:3]  # (224, 224)

# -------------------------
# Custom CSS for styling
# -------------------------
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .subheader { font-size: 1.5rem; color: #2c3e50; margin-bottom: 1rem; }
    .prediction-box { background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin: 10px 0; }
    .camera-container { display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .stButton button { width: 100%; border-radius: 5px; border: 1px solid #1f77b4; background-color: #1f77b4; color: white; padding: 10px; margin: 5px 0; }
    .stButton button:hover { background-color: #1668a8; color: white; }
    .prob-bar { background-color: #1f77b4; border-radius: 5px; padding: 2px; margin: 2px 0; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.title("📚 Project Info")
    st.markdown("""
    **🤟 Sign Language Digit Recognition**

    Dataset by **Ankara Ayrancı Anadolu High School**.

    - Digits: **0–9**
    - Image Size: **100x100 → trained with MobileNet (224x224)**

    👩‍💻 *Built with Keras + Streamlit*
    """)

    st.markdown("---")
    st.write("### ⚙️ How to use:")
    st.write("1. Upload an image OR use webcam")
    st.write("2. Wait for prediction")
    st.write("3. View probabilities")

    st.markdown("---")
    st.write("### 🖐️ Hand Pose Tips")
    st.write("- Ensure good lighting")
    st.write("- Keep hand centered")
    st.write("- Avoid complex backgrounds")
    st.write("- Show clear finger positions")

# -------------------------
# App Layout
# -------------------------
st.markdown('<h1 class="main-header">🤟 Sign Language Digit Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload a hand sign or capture with your camera to recognize the digit!</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera Capture"])

# -------------------------
# Function: Preprocess + Predict
# -------------------------
def predict_digit(pil_img):
    if model is None:
        return None, None
    img = pil_img.resize(input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array, verbose=0)[0]
    predicted_class = np.argmax(preds)
    return predicted_class, preds

# -------------------------
# Function: Enhance image
# -------------------------
def enhance_image(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    gray_img = img.convert('L')
    img = gray_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = img.convert('RGB')
    return img

# -------------------------
# Upload Image Tab
# -------------------------
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload a hand sign image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            enhanced_img = enhance_image(img)

            col1a, col1b = st.columns(2)
            with col1a:
                st.image(img, caption="Original Image", use_container_width=True)
            with col1b:
                st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

            with st.spinner("Analyzing image..."):
                time.sleep(0.5)
                label, probs = predict_digit(enhanced_img)

            if label is not None:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.success(f"✅ Predicted Digit: **{label}**")
                st.markdown('</div>', unsafe_allow_html=True)

                st.write("### 🔎 Prediction Confidence")
                for i, p in enumerate(probs):
                    col_prob1, col_prob2, col_prob3 = st.columns([1, 4, 1])
                    with col_prob1: st.write(f"Digit {i}:")
                    with col_prob2: st.progress(float(p), text=f"{p:.2%}")
                    with col_prob3: st.write(f"{p:.2%}")
                    if i == label: st.markdown("**✓**")

    with col2:
        st.write("### 💡 Tips for better results")
        st.info("""
        - Use clear hand signs against a plain background
        - Ensure good lighting on your hand
        - Center your hand in the image
        - Avoid shadows on your hand
        - Make sure all fingers are visible
        """)

        if uploaded_file and label is not None:
            st.write("### 📊 Confidence Chart")
            chart_data = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability": probs})
            st.bar_chart(chart_data.set_index("Digit"))

# -------------------------
# Camera Capture Tab
# -------------------------
with tab2:
    col1, col2 = st.columns([2, 1])

    label = None
    probs = np.zeros(10)

    with col1:
        st.markdown('<div class="camera-container">', unsafe_allow_html=True)

        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📷 Open Camera" if not st.session_state.camera_active else "🔄 Camera Active"):
                st.session_state.camera_active = True
                st.rerun()
        with col_btn2:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_active = False
                st.rerun()

        if st.session_state.camera_active:
            camera_img = st.camera_input("Take a picture of your hand sign", key="camera")
            if camera_img:
                img = Image.open(camera_img).convert("RGB")
                enhanced_img = enhance_image(img)
                label, probs = predict_digit(enhanced_img)

                col1a, col1b = st.columns(2)
                with col1a: st.image(img, caption="Captured Image", use_container_width=True)
                with col1b: st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

                if label is not None:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.success(f"✅ Predicted Digit: **{label}**")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.write("### 🔎 Prediction Confidence")
                    for i, p in enumerate(probs):
                        col_prob1, col_prob2, col_prob3 = st.columns([1, 4, 1])
                        with col_prob1: st.write(f"Digit {i}:")
                        with col_prob2: st.progress(float(p), text=f"{p:.2%}")
                        with col_prob3: st.write(f"{p:.2%}")
                        if i == label: st.markdown("**✓**")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.write("### 📸 Camera Tips")
        st.info("""
        - Hold your hand steady
        - Position hand in the center of view
        - Ensure even lighting
        - Use a plain background
        - Keep fingers clearly separated
        """)

        if label is not None:
            st.write("### 📊 Confidence Chart")
            chart_data = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability": probs})
            st.bar_chart(chart_data.set_index("Digit"))

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Sign Language Recognition App | Built by Mirkamol Rakhimov</div>",
    unsafe_allow_html=True
)
