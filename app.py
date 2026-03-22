import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

st.title("Face Counter App")
st.write("Using MediaPipe AI for better accuracy.")

def process_image(image):
    img = np.array(image)
    
    # MediaPipe needs RGB (which PIL already provides)
    results = face_detection.process(img)
    
    count = 0
    if results.detections:
        count = len(results.detections)
        for detection in results.detections:
            # Get bounding box coordinates
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                         int(bbox.width * iw), int(bbox.height * ih)
            
            # Draw rectangle (Green)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
    return img, count

# -------- IMAGE UPLOAD --------
st.header("Upload Photo")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    processed_img, face_count = process_image(image)
    st.image(processed_img, caption=f"Faces detected: {face_count}", use_container_width=True)

# -------- CAMERA --------
st.header("Camera")
camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image)
    processed_img, face_count = process_image(image)
    st.image(processed_img, caption=f"Faces detected: {face_count}", use_container_width=True)
