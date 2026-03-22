import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_camera_input_live import camera_input_live

st.set_page_config(page_title="Face Detection", page_icon="😀")

st.title("😀 Face Detection App")
st.write("Detect faces from camera or photo")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def detect_faces(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

    return img, len(faces)


# --------------------
# Upload Image
# --------------------

st.subheader("Upload Photo")

file = st.file_uploader(
    "Upload image",
    type=["jpg","png","jpeg"]
)

if file is not None:

    image = Image.open(file)
    img = np.array(image)

    img, count = detect_faces(img)

    st.image(img, channels="BGR")

    st.success(f"Faces detected: {count}")


# --------------------
# Camera
# --------------------

st.subheader("Camera")

photo = camera_input_live()

if photo is not None:

    img = Image.open(photo)
    img = np.array(img)

    img, count = detect_faces(img)

    st.image(img, channels="BGR")

    st.success(f"Faces detected: {count}")
