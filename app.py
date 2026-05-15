import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT,
    password TEXT
)
""")

conn.commit()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PlantCare AI", layout="wide")

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# ---------------- CUSTOM WEBSITE CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #e8f5e9, #ffffff);
    font-family: "Segoe UI", sans-serif;
}

.navbar {
    background: #2e7d32;
    padding: 18px;
    border-radius: 0px 0px 20px 20px;
    color: white;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.result {
    border-left: 8px solid #43a047;
    background: #f1f8e9;
    padding: 25px;
    border-radius: 15px;
    font-size: 18px;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 17px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #1b5e20;
}

</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown(
    "<div class='navbar'>PlantCare AI - Plant Disease Detection Portal</div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR MENU ----------------
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Disease Detection", "About"]
)

# ---------------- LOAD MODEL (FIXED) ----------------
@st.cache_resource
def load_trained_model():
    return load_model("plant_disease_model.h5", compile=False)

model = load_trained_model()

# ---------------- CLASS LABELS ----------------
class_labels = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight"
]

# ---------------- DISEASE INFO ----------------
info = {

    "Tomato_Healthy": (
        "Healthy Leaf",
        "No treatment required",
        "Continue regular monitoring and balanced nutrition",
        ""
    ),

    "Potato_Healthy": (
        "Healthy Leaf",
        "No treatment required",
        "Maintain proper irrigation and soil health",
        ""
    ),

    "Tomato_Early_Blight": (
        "Early Blight",
        "Chlorothalonil Fungicide",
        "Remove infected leaves and avoid overhead watering",
        "https://www.amazon.in/s?k=chlorothalonil+fungicide"
    ),

    "Tomato_Late_Blight": (
        "Late Blight",
        "Copper Fungicide",
        "Ensure airflow and proper drainage",
        "https://www.amazon.in/s?k=copper+fungicide"
    ),

    "Tomato_Bacterial_Spot": (
        "Bacterial Spot",
        "Copper-based Bactericide",
        "Avoid leaf wetting and use certified seeds",
        "https://www.amazon.in/s?k=copper+bactericide"
    ),

    "Potato_Early_Blight": (
        "Early Blight",
        "Mancozeb Fungicide",
        "Crop rotation and remove infected leaves",
        "https://www.amazon.in/s?k=mancozeb+fungicide"
    ),

    "Potato_Late_Blight": (
        "Late Blight",
        "Metalaxyl Fungicide",
        "Avoid excessive moisture and ensure ventilation",
        "https://www.amazon.in/s?k=metalaxyl+fungicide"
    )
}

# =====================================================
# HOME PAGE
# =====================================================
if menu == "Home":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if not st.session_state.logged_in:

        st.header("User Login")

        name = st.text_input("Enter Your Name")
        password = st.text_input("Enter Password", type="password")

    # ---------------- LOGIN ----------------
        if st.button("Login"):

            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (name, password)
            )

            user = cursor.fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.user_name = name
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # ---------------- SIGNUP ----------------
        st.markdown("---")
        st.subheader("New User? Sign Up")

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Sign Up"):
            if new_user and new_pass:

                cursor.execute(
                    "INSERT INTO users VALUES (?, ?)",
                    (new_user, new_pass)
                )
                conn.commit()

                st.success("Account created! Please login.")

            else:
                st.error("Please fill all fields")
    else:

        st.success(f"Welcome, {st.session_state.user_name}!")

        st.header("PlantCare AI System")

        st.write("""
        Upload a leaf image to detect diseases and receive cure and prevention suggestions.
        """)

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# BLOCK OTHER PAGES UNTIL LOGIN
# =====================================================
elif not st.session_state.logged_in:
    st.warning("Please login from the Home page to access other features.")

# =====================================================
# DISEASE DETECTION PAGE
# =====================================================
elif menu == "Disease Detection":

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Plant Diagnosis Input")

        plant_name = st.text_input("Enter Plant Name (Tomato or Potato)")

        uploaded_file = st.file_uploader(
            "Upload Leaf Image",
            type=["jpg", "jpeg", "png"]
        )

        predict_btn = st.button("Detect Disease")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Diagnosis Report")

        if uploaded_file and predict_btn:

            # ✅ FIX 1: Convert to RGB
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

            # ✅ FIX 2: Correct preprocessing
            img_resized = img.resize((300, 300))
            img_array = np.array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            pred = model.predict(img_array)[0]

            # Plant filtering
            plant_name_clean = plant_name.strip().lower()

            allowed_indexes = []
            for i, label in enumerate(class_labels):
                if plant_name_clean in label.lower():
                    allowed_indexes.append(i)

            if len(allowed_indexes) == 0:
                st.error("Plant name not recognized. Please enter Tomato or Potato.")

            else:
                filtered_pred = pred[allowed_indexes]
                best_local_index = np.argmax(filtered_pred)
                best_index = allowed_indexes[best_local_index]

                predicted_label = class_labels[best_index]
                confidence = round(filtered_pred[best_local_index] * 100, 2)

                disease, cure, prevention, link = info[predicted_label]

                st.markdown(f"""
                <div class="result">
                <h3>Diagnosis Result</h3>

                <b>Plant Name:</b> {plant_name}<br><br>

                <b>Disease Detected:</b> {disease}<br><br>

                <b>Confidence Score:</b>
                <span style="color:green; font-size:20px; font-weight:bold;">
                {confidence}%
                </span><br><br>

                <b>Recommended Cure:</b> {cure}<br><br>

                <b>Prevention Measures:</b> {prevention}<br><br>
                </div>
                """, unsafe_allow_html=True)

                if link:
                    st.markdown(
                        f"<p style='font-size:20px; font-weight:700;'>"
                        f"Buy Recommended Pesticide: "
                        f"<a href='{link}' target='_blank'>Click Here</a>"
                        f"</p>",
                        unsafe_allow_html=True
                    )

        else:
            st.info("Enter plant name, upload an image, and click Detect Disease.")

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# ABOUT PAGE
# =====================================================
elif menu == "About":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.header("About This Project")

    st.write("""
    PlantCare AI is a deep learning based plant disease detection system.

    It allows users to upload a tomato or potato leaf image and instantly predicts 
    whether the plant is healthy or affected by a disease.

    The system also provides recommended cure and prevention measures to help reduce 
    crop loss through early diagnosis.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Educational Project: AI-based Plant Disease Detection using CNN and TensorFlow")
