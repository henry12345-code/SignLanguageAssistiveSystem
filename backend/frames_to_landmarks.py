import cv2
import mediapipe as mp
import numpy as np
import os

RAW_DIR = "../data/raw_frames"
LANDMARK_DIR = "../data/landmarks"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

os.makedirs(LANDMARK_DIR, exist_ok=True)

for label in os.listdir(RAW_DIR):
    label_path = os.path.join(RAW_DIR, label)
    save_path = os.path.join(LANDMARK_DIR, label)
    os.makedirs(save_path, exist_ok=True)

    count = 0

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        # Load image
        img = cv2.imread(img_path)

        # Skip bad images
        if img is None:
            continue

        # Resize to reduce memory usage
        img = cv2.resize(img, (640, 480))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                keypoints = []

                for lm in hand.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                file_path = os.path.join(save_path, f"{count}.npy")
                np.save(file_path, keypoints)

                count += 1

print("Landmark extraction complete.")