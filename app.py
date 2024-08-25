from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve features from the form
        features = request.form['feature']
        features = features.split(',')
        np_features = np.asarray(features, dtype=np.float32)

        # Ensure features are in the correct shape
        if len(np_features) != model.n_features_in_:
            raise ValueError("Incorrect number of features provided")

        # Make prediction
        pred = model.predict(np_features.reshape(1, -1))
        message = 'Cancerous' if pred[0] == 1 else 'Not Cancerous'

    except ValueError as ve:
        message = f"Error: {ve}"
    except Exception as e:
        message = f"Error: {str(e)}"

    # Render the result on the same page
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
