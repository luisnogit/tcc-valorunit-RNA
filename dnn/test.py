import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)  # Import metrics

# --- Configuration for GPU (from your original code) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# The following lines are for advanced GPU usage and can be simplified
# or removed if running on CPU or standard GPU setup.
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction=0.5
# session = tf.compat.v1.InteractiveSession(config=config)
# --------------------------------------------------------

# --- Data Loading and Preparation ---
# Assuming 'test.csv' exists and the last column is the continuous target 'y'
try:
    test_dataset = pd.read_csv("dnn/test.csv")
    x_test = test_dataset.iloc[:, :-1].values
    # Get the actual continuous target values for regression
    y_test_actual = test_dataset.iloc[:, -1].values

except FileNotFoundError:
    print(
        "Error: 'test.csv' not found. Please ensure the file is in the correct directory."
    )
    exit()

# --- Model Loading ---
try:
    # Recreate the exact same model, including its weights and the optimizer
    # Ensure the original model was trained for REGRESSION (e.g., final layer activation=None, loss='mse')
    dnn_model = tf.keras.models.load_model("dnn_model.h5")
    dnn_model.summary()
except FileNotFoundError:
    print(
        "Error: 'dnn_model.h5' not found. Please ensure the file is in the correct directory."
    )
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Prediction ---
P = dnn_model.predict(x_test)

# For regression, the output P is the predicted continuous value.
# We assume the output layer has 1 neuron, so we might need to flatten it.
# P.shape should be (n_samples, 1). If so, reshape to (n_samples,)
if P.shape[-1] == 1:
    y_pred = P.flatten()
else:
    # If the model had more than 1 output neuron, it's NOT a standard univariate regression
    # Check your model architecture. Assuming we take the first output as the prediction.
    y_pred = P[:, 0]
    print(
        "Warning: Model output shape is not (samples, 1). Using only the first output column."
    )


# --- 1. Calculate and Print Regression Metrics ---
## 📊 Regression Metrics
print("\n" + "=" * 50)
print("📊 Regression Model Evaluation Metrics")
print("=" * 50)
y_test_actual = (1* y_test_actual) - 0
y_test_actual = (y_test_actual*(29887.820513)) + 10.010010

y_pred= (1* y_pred) - 0
y_pred= (y_pred*(29887.820513)) + 10.010010

area = x_test[:, 0]

area= (1* area) - 0
area = (area*(2520.000000)) + 101.000000
## Mean Squared Error (MSE)
mse = mean_squared_error(y_test_actual, y_pred)
# Root Mean Squared Error (RMSE)
rmse = mse
# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_actual, y_pred)
# R-squared (Coefficient of Determination)
r2 = r2_score(y_test_actual, y_pred)

print(f"Mean Squared Error (MSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print("=" * 50)

# --- 2. Plot: Actual vs. Predicted Values ---
## 📈 Actual vs. Predicted Plot

plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.7)
# Plot the ideal line (y=x) where actual = predicted
min_val = min(y_test_actual.min(), y_pred.min())
max_val = max(y_test_actual.max(), y_pred.max())
plt.plot(
    [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction (y=x)"
)

plt.title("Actual vs. Predicted Values for DNN Regression")
plt.xlabel("Actual Values ($Y_{Actual}$)")
plt.ylabel("Predicted Values ($\hat{Y}$)")
plt.legend()
plt.grid(True)
plt.show()


# --- 3. Plot: Residuals Plot ---
## 📉 Residuals Plot

# Calculate residuals
residuals = y_test_actual - y_pred

plt.figure(figsize=(10, 6))
# Plot residuals against the predicted values
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color="r", linestyle="--", label="Zero Residuals")

plt.title("Residuals Plot")
plt.xlabel("Predicted Values ($\hat{Y}$)")
plt.ylabel("Residuals ($Y_{Actual} - \hat{Y}$)")
plt.legend()
plt.grid(True)
plt.show()


plt.scatter(y_pred, area, alpha=0.7)
plt.show()

print("\nDNN Regression Test Complete.")
