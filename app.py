import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.title("🚦 AI Traffic Violation Detection using YOLOv8")

st.write("Upload a traffic video to detect vehicles.")

# Load model
model = YOLO("yolov8n.pt")

# Upload video
uploaded_file = st.file_uploader("Upload Traffic Video", type=["mp4","avi"])

if uploaded_file is not None:

    st.success("Video uploaded successfully!")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()

        stframe.image(frame, channels="BGR")

    cap.release()