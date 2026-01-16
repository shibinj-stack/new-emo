import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from model import create_model # Ensure model.py exists

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "emotion_lstm.h5")

X, y = [], []
for label in ["happy", "sad", "calm", "stressed"]:
    df = pd.read_csv(os.path.join(DATA_DIR, f"{label}.csv"), header=None)
    for row in df.values:
        # Scale data: 0 to 1 range (where 1.0 = 1000ms)
        X.append(np.array(row, dtype=np.float32) / 1000.0)
        y.append(label)

X = np.array(X).reshape(-1, 50, 1)
encoder = LabelEncoder()
y = to_categorical(encoder.fit_transform(y))

model = create_model()
print("🧠 Training AI on behavioral consistency logic...")
model.fit(X, y, epochs=40, batch_size=16, verbose=1)
model.save(MODEL_PATH)
print(f"🎉 Success! High-fidelity model saved to {MODEL_PATH}")