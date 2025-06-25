import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from math import sqrt

# Start MLflow experiment
mlflow.set_experiment("Wind Speed Prediction")

with mlflow.start_run():
    # Load dataset
    df = pd.read_csv("data/windt.csv")

    # Basic preprocessing: drop NA values
    df = df.dropna()

    # Define features and target column (update target column name if needed)
    X = df.drop("TurbineName", axis=1)  # ← Replace with actual target column if different
    y = df["TurbineName"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simple Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Log the model with input example
    example_input = X_test.iloc[:1]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="linear_regression_model",
        input_example=example_input
    )

    # Print evaluation results
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
