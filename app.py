from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = 1 if request.form['sex'] == 'male' else 0
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    
    embarked = request.form['embarked']
    embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
    embarked_encoded = embarked_dict.get(embarked, 2)  # Default to 'S'

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(features)[0]
    
    result = "Survived" if prediction == 1 else "Did not survive"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
