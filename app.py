from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    try:
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']

        # Prepare input data (same features as in the notebook)
        sex_male = 1 if sex == 'male' else 0
        sex_female = 1 if sex == 'female' else 0
        embarked_C = 1 if embarked == 'C' else 0
        embarked_Q = 1 if embarked == 'Q' else 0
        embarked_S = 1 if embarked == 'S' else 0

        # Create feature array
        features = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_S]])

        # Make prediction
        prediction = model.predict(features)[0]
        prediction_text = 'Survived' if prediction == 1 else 'Did not survive'

        return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
