import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model("../models/sign_model.h5")

# Label map (change according to your dataset folders)
labels = sorted(os.listdir("../data/landmarks"))
label_map = {i: label for i, label in enumerate(labels)}

# MediaPipe Hands (FAST MODE)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Phone camera (IP Webcam)
cap = cv2.VideoCapture(1)

# 🔥 FPS BOOST SETTINGS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
prediction_text = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    frame_count += 1

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Predict only every 3 frames (FPS boost)
            if frame_count % 3 == 0:
                keypoints = []
                for lm in handLms.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                keypoints = np.array(keypoints).reshape(1, -1)
                pred = model.predict(keypoints, verbose=0)
                class_id = np.argmax(pred)
                prediction_text = label_map[class_id]

    # Show prediction
    cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
    cv2.putText(frame, prediction_text, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Sign Language Call", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
