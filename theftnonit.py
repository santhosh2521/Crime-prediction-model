import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Embedding, Input, Concatenate, Flatten,Dropout
from keras.utils import to_categorical

adilabad_year = 2023

# Load data from CSV file
df = pd.read_csv("total_crimes.csv")  # Replace 'your_data.csv' with your actual file path

# Filter data for the district of Adilabad
adilabad_data = df[df['DISTRICT'] == 'ADILABAD']

# Feature engineering
adilabad_data['target'] = (adilabad_data['THEFT'] > 50).astype(int)

# Extract relevant features
features = ['YEAR']  # Adjust this based on your data
X = adilabad_data[features].values
y = adilabad_data['target'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
accuracy = model.evaluate(X_test_reshaped, y_test)[1]
print(f"Accuracy on the test set: {accuracy}")

# Make predictions for Adilabad
adilabad_data_for_prediction = np.array([[adilabad_year]])  # Adjust based on your data
adilabad_data_for_prediction_scaled = scaler.transform(adilabad_data_for_prediction)
adilabad_data_for_prediction_reshaped = adilabad_data_for_prediction_scaled.reshape(
    (adilabad_data_for_prediction_scaled.shape[0], 1, adilabad_data_for_prediction_scaled.shape[1]))

probability = model.predict(adilabad_data_for_prediction_reshaped)[0][0]
print(f"Probability of theft in Adilabad: {probability}")
print("The percentage is: ",int(probability*100))