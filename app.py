from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        input_features = [float(request.form[f"f{i}"]) for i in range(30)]
        input_array = np.array(input_features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]
        class_names = ["Malignant", "Benign"]
        result = class_names[prediction]
        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except:
        return render_template("index.html", prediction_text="Please enter all fields correctly.")

if __name__ == "__main__":
    app.run(debug=True)
