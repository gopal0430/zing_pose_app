import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile

st.title("Zing Coach – Mobile Pose Snapshot")

st.write("📱 Works on mobile too! Use your camera to capture a pose snapshot:")

img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Load image with PIL and convert to OpenCV format
    img = Image.open(img_file_buffer)
    img_np = np.array(img)

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    # Process pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Pose Landmarks", use_column_width=True)
    else:
        st.warning("⚠️ No pose detected. Try retaking the photo.")
