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
display(unique_districts)
print(len(unique_districts))