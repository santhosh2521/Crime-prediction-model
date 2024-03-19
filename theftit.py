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

# Load data from CSV file
df = pd.read_csv("total_crimes1.csv")  # Replace 'your_data.csv' with your actual file path

# Get unique districts in the dataset
unique_districts = df['DISTRICT'].unique()

# Initialize a dictionary to store results
district_results = {}

for district in unique_districts:
    # Filter data for the current district
    district_data = df[df['DISTRICT'] == district]

    # Feature engineering
    district_data['target'] = (district_data['THEFT'] > 60).astype(int)

    # Extract relevant features
    features = ['YEAR']  # Adjust this based on your data
    X = district_data[features].values
    y = district_data['target'].values

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
    model.fit(X_train_reshaped, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    accuracy = model.evaluate(X_test_reshaped, y_test)[1]
    print(f"Accuracy on the test set for {district}: {accuracy}")

    # Make predictions for the current district
    district_data_for_prediction = np.array(district_data['YEAR']).reshape(-1, 1)  # Adjust based on your data
    district_data_for_prediction_scaled = scaler.transform(district_data_for_prediction)
    district_data_for_prediction_reshaped = district_data_for_prediction_scaled.reshape(
        (district_data_for_prediction_scaled.shape[0], 1, district_data_for_prediction_scaled.shape[1]))

    probability = model.predict(district_data_for_prediction_reshaped)[0][0]
    print(f"Probability of theft in {district}: {probability}")
    print("The percentage is: ", int(probability * 100))

    # Store results in the dictionary
    district_results[district] = probability

# Display the results
for district, probability in district_results.items():
    print(f"Probability of theft in {district}: {probability}")
    print("The percentage is: ", int(probability * 100)) 

model.save(f"theft_model.h5")