import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score

# Load the model
best_model = joblib.load('model_components/best_model.pkl')

# Load and preprocess the dataset
california = fetch_california_housing()
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['PRICE'] = california.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(california_df.drop(columns=['PRICE']))

# Use the model for predictions
predictions = best_model.predict(X_scaled)

# Calculate evaluation scores
mse = mean_squared_error(california_df['PRICE'], predictions)
r2 = r2_score(california_df['PRICE'], predictions)

# Create a DataFrame with actual and predicted values
results_df = california_df.copy()
results_df['Predicted Price'] = predictions

# Display the first ten records with all features, actual prices, and predicted prices
print("First 10 records with actual and predicted prices:")
print(results_df.head(10))

# Print the best model evaluation scores
print("\nBest Model Evaluation Scores:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
