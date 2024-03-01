import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['heart_rate'] = data['Heart'].apply(lambda x: np.array(x.split(','), dtype=np.float32))
    data['Breath'] = data['Breath'].apply(lambda x: np.array(x.split(','), dtype=np.float32))
    return data

def preprocess_data(data):
    X = data[['heart_rate', 'Breath']].values
    age_class_mapping = {'0-3': 0, '3-7': 1, '7-10': 2, '10-13': 3}
    y = data['Age_Class'].map(age_class_mapping)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
