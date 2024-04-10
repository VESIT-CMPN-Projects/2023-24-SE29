import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
tourism = pd.read_csv("dataset4.csv")

# Separate features and labels
tourism_labels = tourism["CARBF"].copy()
tourism_features = tourism.drop("CARBF", axis=1)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    tourism_features, tourism_labels, test_size=0.2, random_state=42
)

# Create a pipeline for preprocessing
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Fit and transform the training data using the pipeline
train_features_transformed = my_pipeline.fit_transform(train_features)

# Create and train the Random Forest model
model = RandomForestRegressor()
model.fit(train_features_transformed, train_labels)

# Save the model as a pickle file
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
