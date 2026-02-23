import os

DATA_DIR = "../data/landmarks"

labels = sorted(os.listdir(DATA_DIR))

print("Model Classes:")
for i, label in enumerate(labels):
    print(f"{i} -> {label}")