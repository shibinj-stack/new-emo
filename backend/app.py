import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_lstm.h5")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

model = None
emotions = ["Happy", "Sad", "Calm", "Stressed"]

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    json_data = request.json
    keystroke_data = json_data.get("data", [])
    user_text = json_data.get("text", "")

    # --- STAGE 1: VADER NLP with 90-100% Scaling Equation ---
    if user_text.strip():
        vs = analyzer.polarity_scores(user_text)
        compound = vs['compound'] 

        # Activate if sentiment intensity is at least 0.4
        if abs(compound) >= 0.4:
            # SCALING EQUATION: Map compound (0.4 to 1.0) to Confidence (0.90 to 1.00)
            # Formula: 0.90 + ((|compound| - 0.4) / (1.0 - 0.4)) * 0.10
            scaled_score = 0.90 + ((abs(compound) - 0.4) / 0.6) * 0.10
            
            emotion_label = "Happy (NLP)" if compound > 0 else "Sad/Stressed (NLP)"
            return jsonify({
                "emotion": emotion_label, 
                "confidence": float(scaled_score)
            })

    # --- STAGE 2: AI KEYSTROKE ANALYSIS (Mathematical Probability) ---
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"emotion": "AI Model not found", "confidence": 0})
        model = load_model(MODEL_PATH)

    if not isinstance(keystroke_data, list) or len(keystroke_data) < 10:
        return jsonify({"emotion": "Neutral / Need more typing", "confidence": 0})

    # Prepare data for LSTM model
    data = np.array(keystroke_data, dtype=np.float32)
    data = np.pad(data, (0, max(0, 50 - len(data))))[:50].reshape(1, 50, 1)

    # SOFTMAX EQUATION: Confidence is the maximum value in the output probability vector
    prediction = model.predict(data, verbose=0)
    confidence = float(np.max(prediction)) 
    emotion = emotions[int(np.argmax(prediction))]

    return jsonify({"emotion": emotion, "confidence": confidence})

if __name__ == "__main__":
    # Get the PORT from Render's environment variable, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # Bind to 0.0.0.0 so the service is accessible externally
    app.run(host="0.0.0.0", port=port, debug=False)