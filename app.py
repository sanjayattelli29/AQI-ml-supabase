import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask
app = Flask(__name__)

# Load ML models from the local directory
models = {
    "Naïve Bayes": joblib.load("ml_models/Naïve Bayes_model.pkl"),
    "KNN": joblib.load("ml_models/KNN_model.pkl"),
    "SVM": joblib.load("ml_models/SVM_model.pkl"),
    "Random Forest": joblib.load("ml_models/Random_Forest_model.pkl")
}

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Input values from Lovable.dev
        model_name = data.get("model")  # Select Model
        features = np.array(data.get("features")).reshape(1, -1)

        if model_name not in models:
            return jsonify({"error": "Invalid model name or model not found"}), 400
        
        model = models[model_name]
        prediction = model.predict(features).tolist()
        
        return jsonify({"model": model_name, "prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
