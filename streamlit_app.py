import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import platform

st.title("Zing Coach – Pose Detection")

device = st.radio("Select device type:", ["Desktop", "Mobile"])

FRAME_WINDOW = st.image([])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 🔁 Mobile camera input
if device == "Mobile":
    img_file_buffer = st.camera_input("📸 Take a photo to detect pose")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        img_np = np.array(img)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Pose detection
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Pose Landmarks", use_column_width=True)
        else:
            st.warning("⚠️ No pose detected. Try retaking the photo.")

# 🖥️ Desktop webcam input
else:
    run = st.checkbox("🎥 Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Webcam not found or is in use.")
        else:
            st.success("✅ Webcam accessed. Pose detection running...")

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Could not capture frame.")
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                if results.pose_landmarks:
                    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
