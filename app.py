import joblib
import base64
import numpy as np
import json
from flask import Flask, request, jsonify
from supabase import create_client

# Initialize Flask
app = Flask(__name__)

# Supabase Configuration
SUPABASE_URL = "https://ilhvxwmuekvwrekwdgyb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlsaHZ4d211ZWt2d3Jla3dkZ3liIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI1ODEyMDYsImV4cCI6MjA1ODE1NzIwNn0.w7cH3o9MwpWZ97EjuVh8D-HqM1ESq26fvb-2SWQtmoI"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load ML models from Supabase
def load_model(model_name):
    response = supabase.table("ml_models").select("model_data").eq("model_name", model_name).execute()
    
    if response.data:
        model_data = response.data[0]['model_data']
        decoded_model = base64.b64decode(model_data)
        model_filename = f"{model_name}.pkl"
        
        with open(model_filename, "wb") as f:
            f.write(decoded_model)

        return joblib.load(model_filename)
    return None

# Load all models
models = {
    "Naïve Bayes": load_model("Naïve Bayes"),
    "KNN": load_model("KNN"),
    "SVM": load_model("SVM"),
    "Random Forest": load_model("Random Forest")
}

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Input values from Lovable.dev
        model_name = data.get("model")  # Select Model
        features = np.array(data.get("features")).reshape(1, -1)

        if model_name not in models or models[model_name] is None:
            return jsonify({"error": "Invalid model name or model not found"}), 400
        
        model = models[model_name]
        prediction = model.predict(features).tolist()
        
        return jsonify({"model": model_name, "prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
