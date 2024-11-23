import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

image_path = "/Users/bagseonghyeon/Desktop/project/신이박최/images (1).jpeg"
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(image_rgb)

output_image = image.copy()

if results.pose_landmarks:
    mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    landmarks = results.pose_landmarks.landmark

    ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    neck = landmarks[mp_pose.PoseLandmark.NOSE]

    ear_x, ear_y = int(ear.x * image.shape[1]), int(ear.y * image.shape[0])
    neck_x, neck_y = int(neck.x * image.shape[1]), int(neck.y * image.shape[0])

    delta_x = ear_x - neck_x
    delta_y = neck_y - ear_y
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    if angle < 50:
        status = "거북목입니다유ㅠ"
        color = (0, 0, 255)
    else:
        status = "정상입니당"
        color = (0, 255, 0)

    print(f"{status} ({int(angle)} deg)")

    cv2.circle(output_image, (ear_x, ear_y), 5, (255, 0, 0), -1)
    cv2.circle(output_image, (neck_x, neck_y), 5, (0, 255, 255), -1)

    cv2.imwrite("output_image.jpg", output_image)

cv2.imshow("Pose Analysis", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()