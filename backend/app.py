from flask import Flask, render_template, Response
import tensorflow as tf
import numpy as np
import cv2
import os
import mediapipe as mp

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# Load trained model
model_path = os.path.join("..", "models", "sign_model.h5")
model = tf.keras.models.load_model(model_path)

# Load labels from training folders
labels = sorted(os.listdir(os.path.join("..", "data", "landmarks")))

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Enable two hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = ""

        if results.multi_hand_landmarks:

            hand_count = len(results.multi_hand_landmarks)

            # Show number of detected hands on screen
            cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            # Draw all detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            # Use first hand for prediction (since model input is 63)
            first_hand = results.multi_hand_landmarks[0]

            landmarks = []
            for lm in first_hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                input_data = np.array(landmarks).reshape(1, 63)
                prediction = model.predict(input_data, verbose=0)
                predicted_class = labels[np.argmax(prediction)]
                prediction_text = predicted_class

        # Subtitle-style text at bottom center
        if prediction_text != "":
            h, w, _ = frame.shape

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(
                prediction_text, font, font_scale, thickness
            )

            x = int((w - text_width) / 2)
            y = h - 40

            cv2.rectangle(
                frame,
                (x - 20, y - text_height - 20),
                (x + text_width + 20, y + 10),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                prediction_text,
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)