import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Counter App")

# Load face model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------- IMAGE UPLOAD --------
st.header("Upload Photo")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    # FIX 1: Use RGB to GRAY instead of BGR
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # FIX 2: Tuning parameters for better sensitivity
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30) # Optional: ignores very small noise
    )

    # When drawing the rectangle, remember OpenCV will draw on the 
    # numpy array. Streamlit's st.image() handles the display.
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(img, caption=f"Faces detected: {len(faces)}", use_container_width=True)


# -------- CAMERA --------
st.header("Camera")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    st.image(img, caption=f"Faces detected: {len(faces)}", use_column_width=True)
