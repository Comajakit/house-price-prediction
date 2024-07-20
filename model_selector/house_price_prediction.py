# house_price_prediction.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import xgboost as xgb

# Load the California Housing dataset
california = fetch_california_housing()
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['PRICE'] = california.target

# Check the dataset
print(california_df.head())

# Preprocess the data
X = california_df.drop(columns=['PRICE'])
y = california_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models and their parameters
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor()
}

params = {
    'LinearRegression': {},
    'DecisionTree': {
        'max_depth': [10, 20],
        'min_samples_split': [10, 20]
    },
    'RandomForest': {
        'n_estimators': [10, 20],
        'max_depth': [None, 2, 20],
        'min_samples_split': [10, 20]
    },
    'GradientBoosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    },
    'XGBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5]
    }
}

# Store the best model and its performance
best_model = None
best_model_name = ""
best_score = float('-inf')
best_params = {}

# Iterate through each model and perform GridSearchCV
for model_name in models:
    model = models[model_name]
    param = params[model_name]
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{model_name} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')
    print(f'Best Parameters: {grid_search.best_params_}')
    
    # Check if this model is the best
    if r2 > best_score:
        best_score = r2
        best_model = grid_search.best_estimator_
        best_model_name = model_name
        best_params = grid_search.best_params_

# Print the best model and its parameters
print(f'Best Model: {best_model}')
print(f'Best Model Name: {best_model_name}')
print(f'Best Parameters: {best_params}')
print(f'Best R-squared: {best_score:.2f}')

# Plot the best model's predictions
y_pred = best_model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='k')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Actual vs Predicted Prices ({best_model_name})')
plt.show()
