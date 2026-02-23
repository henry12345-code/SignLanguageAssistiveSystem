import tensorflow as tf
import os

model_path = os.path.join("..", "models", "sign_model.h5")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

model.summary()