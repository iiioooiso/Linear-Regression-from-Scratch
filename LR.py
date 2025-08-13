import pandas as pd



import numpy as np
from sklearn.model_selection import train_test_split

# Custom linear regression with gradient descent
class LinearRegressionGD:
    """
    A simple implementation of Linear Regression using Gradient Descent.
    """
    def __init__(self, n_features, lr=0.01):
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr

    def forward(self, X):
        # Compute predictions
        return np.dot(X, self.w) + self.b

    def backward(self, X, y_hat, y):
        # Compute gradients for weights and bias
        n = X.shape[0]
        dw = -1/n * np.dot(X.T, (y - y_hat))
        db = -1/n * np.sum(y - y_hat)
        self.dw = dw
        self.db = db

    def optimize(self):
        # Update weights using gradient descent
        self.w -= self.lr * self.dw
        self.b -= self.lr * self.db

def regression_metrics(y_true, y_pred):
    """
    Compute MSE, MAE, and R^2 for predictions.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) /
              np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, mae, r2

def main():
    # Load dataset
    data = pd.read_excel('DB.xlsm')
    
    # Handle missing values and extract features
    data['Precip_Type'] = data['Precip_Type'].fillna(data['Precip_Type'].mode()[0])
    data['Formatted_Date'] = pd.to_datetime(data['Formatted_Date'], utc=True)
    data['Month'] = data['Formatted_Date'].dt.month

    # Prepare feature matrix X and target vector y
    X_numeric = data[['Humidity', 'Visibility (km)', 'Month']]
    X_categorical = pd.get_dummies(data['Precip_Type'], drop_first=True)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = data['Temperature (C)']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )

    # Convert to float64 for numerical stability
    X_train = X_train.astype(np.float64)
    X_test  = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test  = y_test.astype(np.float64)

    # Initialize and train the model
    n_features = X_train.shape[1]
    model = LinearRegressionGD(n_features=n_features, lr=0.01)

    num_epochs = 1000
    for epoch in range(num_epochs):
        # Training step
        y_hat_train = model.forward(X_train)
        model.backward(X_train, y_hat_train, y_train)
        model.optimize()

    # Evaluate the model
    train_preds = model.forward(X_train)
    test_preds = model.forward(X_test)
    train_mse, train_mae, train_r2 = regression_metrics(y_train, train_preds)
    test_mse, test_mae, test_r2 = regression_metrics(y_test, test_preds)

    # Output metrics
    print(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R^2: {train_r2:.4f}")
    print(f"Test  MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, R^2: {test_r2:.4f}")

if __name__ == "__main__":
    main()
