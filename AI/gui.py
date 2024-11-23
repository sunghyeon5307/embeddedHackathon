import tkinter as tk
from tkinter import Label
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Pose Analysis")
root.geometry("640x480")

video_label = Label(root)
video_label.pack()

status_label = tk.Label(root, text="정상입니당", font=("Arial", 14))
status_label.pack()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    status = "정상입니당"
    color = (0, 255, 0)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        neck = landmarks[mp_pose.PoseLandmark.NOSE]

        ear_x, ear_y = int(ear.x * frame.shape[1]), int(ear.y * frame.shape[0])
        neck_x, neck_y = int(neck.x * frame.shape[1]), int(neck.y * frame.shape[0])

        delta_x = ear_x - neck_x
        delta_y = neck_y - ear_y
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        if angle < 50:
            status = "거북목입니다유ㅠ"
            color = (0, 0, 255)

        status_label.config(text=f"{status} ({int(angle)} deg)")

        cv2.circle(frame, (ear_x, ear_y), 5, (255, 0, 0), -1)
        cv2.circle(frame, (neck_x, neck_y), 5, (0, 255, 255), -1)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)

    video_label.config(image=img_tk)
    video_label.image = img_tk

    video_label.after(10, update_frame)

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()