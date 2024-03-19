import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('total_crimes.csv')  # Replace 'your_data.csv' with your actual file path

# Convert categorical variables (State, District) into numerical values using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['STATE/UT', 'DISTRICT'], drop_first=True)

# Extract relevant columns
X = df_encoded.drop(['YEAR', 'MURDER'], axis=1)  # Features: Exclude Year and Murders
y = df_encoded['MURDER']  # Target: Number of murders

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\nFeature Importance:')
print(feature_importance)

# Optionally, you can visualize the actual vs. predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Number of Murders')
plt.ylabel('Predicted Number of Murders')
plt.title('Actual vs. Predicted Number of Murders')
plt.show()