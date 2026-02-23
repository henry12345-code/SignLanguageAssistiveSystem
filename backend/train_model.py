import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_DIR = "../data/landmarks"

X = []
y = []
labels = sorted(os.listdir(DATA_DIR))

label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    for file in os.listdir(os.path.join(DATA_DIR, label)):
        data = np.load(os.path.join(DATA_DIR, label, file))
        X.append(data)
        y.append(label_map[label])

X = np.array(X)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dense(64, activation="relu"),
    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
model.save("../models/sign_model.h5")

print("Model trained and saved.")
