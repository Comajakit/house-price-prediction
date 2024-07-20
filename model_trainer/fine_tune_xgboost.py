import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import xgboost as xgb
import joblib

# Load the California Housing dataset
california = fetch_california_housing()
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['PRICE'] = california.target

# Check the dataset
print(california_df.head())

# Preprocess the data
X = california_df.drop(columns=['PRICE'])
y = california_df['PRICE']

# Feature Selection using RFE
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=7,
                         subsample=0.8, colsample_bytree=0.6, gamma=0.05,
                         reg_alpha=1, reg_lambda=0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'XGBoost - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')

# Plot the model's predictions
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='k')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title(f'Actual vs Predicted Prices (XGBoost)')
plt.show()

# Save the model and selected features
joblib.dump(model, '../model_components/best_model.pkl')
