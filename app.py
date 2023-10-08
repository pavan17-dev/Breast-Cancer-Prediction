from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model_filename.pkl')

# Function to calculate and get the mean values for each column


def get_mean_values():
    mean_values = {
        'radius_mean': 15.0,
        'perimeter_mean': 80.0,
        'area_mean': 550.0,
        'symmetry_mean': 0.2,
        'compactness_mean': 0.1,
        'concave_points_mean': 0.05
    }
    return mean_values

# Home page


@app.route('/')
def home():
    return render_template('index.html', mean_values=get_mean_values())

# Prediction page


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the request
    form_data = request.form.to_dict()

    # Convert form data to DataFrame
    sample_data = pd.DataFrame(form_data, index=[0])

    # Use the loaded model to make predictions on the new data
    predictions = model.predict(sample_data)

    # Convert predictions to human-readable labels (0 for benign, 1 for malignant)
    diagnosis = 'Benign' if predictions[0] == 0 else 'Malignant'

    # Render the result template with the prediction
    return render_template('result.html', diagnosis=diagnosis)


if __name__ == '__main__':
    app.run(debug=True)
