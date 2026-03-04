from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# -----------------------------
# Load model and scaler
# -----------------------------
model = load_model("models/iris_nn_model.h5")

# IMPORTANT:
# Use the SAME scaler used during training
# (Recommended: save scaler using pickle during training)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Class labels
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]])

    # Scale input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return jsonify({
        "prediction": predicted_class
    })


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
