import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

SAMPLES = 250  # More samples = Better AI accuracy
SEQ_LEN = 50

def generate_row(emotion):
    if emotion == "happy":
        # Fast & very consistent (Low standard deviation)
        return np.random.normal(40, 5, SEQ_LEN)
    elif emotion == "calm":
        # Medium & steady
        return np.random.normal(85, 10, SEQ_LEN)
    elif emotion == "sad":
        # Slow & heavy
        return np.random.normal(450, 50, SEQ_LEN)
    elif emotion == "stressed":
        # Inconsistent: Rapid bursts (20ms) mixed with sudden stops (400ms)
        row = []
        for _ in range(SEQ_LEN):
            # 70% chance of a fast burst, 30% chance of a stress pause
            val = np.random.randint(20, 80) if np.random.random() > 0.3 else np.random.randint(200, 550)
            row.append(val)
        return np.array(row)

for emo in ["happy", "sad", "calm", "stressed"]:
    path = os.path.join(DATASET_DIR, f"{emo}.csv")
    with open(path, "w") as f:
        for _ in range(SAMPLES):
            row = generate_row(emo)
            # Clip to ensure no negative numbers
            row = np.clip(row, 10, 800)
            f.write(",".join(map(str, row.astype(int))) + "\n")
    print(f"✅ Created {emo}.csv with behavioral logic.")