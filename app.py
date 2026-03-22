import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection", page_icon="😀")

st.title("😀 Face Detection App")
st.write("Detect faces using camera or upload image")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def detect(img):

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


# -------------------
# Upload
# -------------------

st.subheader("Upload Image")

file = st.file_uploader(
    "Choose image",
    type=["jpg","png","jpeg"]
)

if file:

    image = Image.open(file)
    img = np.array(image)

    img, count = detect(img)

    st.image(img, channels="BGR")

    st.success(f"Faces detected: {count}")


# -------------------
# Camera
# -------------------

st.subheader("Camera")

photo = st.camera_input("Take photo")

if photo:

    image = Image.open(photo)
    img = np.array(image)

    img, count = detect(img)

    st.image(img, channels="BGR")

    st.success(f"Faces detected: {count}")
