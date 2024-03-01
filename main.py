import numpy as np
from data_processing import load_data, preprocess_data, split_data
from model import build_model, compile_model, train_model, scale_data
from sklearn.metrics import mean_squared_error

# Load data
data = load_data('AgeClassified.csv')

# Preprocess data
X, y = preprocess_data(data)

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Scale data
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Build model
input_shape = (X_train_scaled.shape[1],)
model = build_model(input_shape)

# Compile model
model = compile_model(model)

# Train model
history = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
