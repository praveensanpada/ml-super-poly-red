# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('polynomial_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Polynomial Regression Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    feature = data['feature']
    
    features = np.array([[feature]])
    prediction = model.predict(features)
    
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
