import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(path: str = "/sample_sales_data.csv"):
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(data: pd.DataFrame, target: str = "estimated_stock_pct"):
    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm
def train_algorithm_with_cross_validation(X: pd.DataFrame, y: pd.Series):
    # k value
    K = 10
    split = 0.75

    # Create a list that will store the accuracies of each fold
    accuracies = []

    for fold in range(K):
        # Instantiate algorithm
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

        # Scale X data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_test, y_pred)
        accuracies.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    print(f"Average MAE: {np.mean(accuracies):.2f}")

    # Plot feature importances
    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 20))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


