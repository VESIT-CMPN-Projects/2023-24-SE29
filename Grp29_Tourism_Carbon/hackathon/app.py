# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__, template_folder='E:/hackathon/hackathon')

# Load the pre-trained model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load some training data for fitting the imputer and scaler
# Assuming you have access to your training data or use a similar dataset
train_data = pd.read_csv('E:/hackathon/hackathon/training.csv')

# Define the pipeline for preprocessing
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Fit the imputer and scaler on the training data
train_features_transformed = my_pipeline['imputer'].fit_transform(train_data[['DIST', 'STOP', 'FE', 'LEN']])
my_pipeline['std_scaler'].fit(train_features_transformed)

# Load the HTML template
@app.route('/')
def index():
    return render_template('calc.html', prediction_text="")

# Handle form submission and display results
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract user inputs from the form
            distance = float(request.form['distance'])
            num_of_stops = float(request.form['stops'])
            fuel_efficiency = float(request.form['fuelEfficiency'])
            length_of_stay = float(request.form['lengthOfStay'])

            # Perform prediction using the pre-trained model
            input_data = pd.DataFrame({
                'DIST': [distance],
                'STOP': [num_of_stops],
                'FE': [fuel_efficiency],
                'LEN': [length_of_stay]
            })

            # Use the pipeline to preprocess the input data
            input_data_transformed = my_pipeline.transform(input_data)

            # Make the prediction
            prediction = model.predict(input_data_transformed)

            # Render the result in the HTML template
            return render_template('calc.html', prediction_text=f"Predicted Carbon Footprint: {prediction[0]:.2f} kg CO2")
        except Exception as e:
            # Handle errors, if any
            return render_template('calc.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
