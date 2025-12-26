import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# Get the path of the CSV file
script_dir = os.path.dirname(__file__) 
file_path = os.path.join(script_dir, 'data.csv')

# Fetch the CSV file using Pandas
data = pd.read_csv(file_path)

# Load the data onto our numpy arrays
x_data = data[["study_hours", "sleep_hours"]].values
y_data = data["passed"].values

# Normalize the data using z-score normalization
# The axis = 0 tells it to find the mean for each column
myu = np.mean(x_data, axis = 0)
std = np.std(x_data, axis = 0)
x = (x_data - myu) / std

# For testing whether the model performs well on unseen data, we divide the data into training and testing data
x_train = x[:80]
x_test = x[80:]

y_train = y_data[:80]
y_test = y_data[80:]

# Since we have two features of the input, we use a weight vector of size 2
# We also set an initial value for the bias variable
w = np.array([0.0, 0.0])
b = 0.0

# Let's define the number of iterations to 10000
iterations = 10000

count = 0

# Let's also find the number of data points and the number of features
m, n = x_train.shape

# Let's also define the cost array to find whether the cost is decreasing with time or not
cost = np.zeros(iterations)

# Let's also have a fixed learning rate of 0.01
alpha = 0.1

# Let's define the sigmoid function
def sigmoid(z):

    return 1 / ( 1 + np.exp(-z))


while(count < iterations):

    # Compute predictions (z = wx + b)
    z = np.dot(x_train, w) + b
    f_wb = sigmoid(z)
    
    # Compute Cost (J)
    # Add a tiny value (1e-15) to prevent log(0) errors
    cost[count] = (-1 / m) * np.sum(y_train * np.log(f_wb + 1e-15) + (1 - y_train) * np.log(1 - f_wb + 1e-15))

    # Compute Gradients
    err = f_wb - y_train
    dj_dw = (1 / m) * np.dot(x_train.T, err)
    dj_db = (1 / m) * np.sum(err)

    # Update Parameters
    w = w - alpha * dj_dw
    b = b - alpha * dj_db

    count += 1

plt.figure(figsize=(10, 6))
    
# Plot the actual test data points
# Green for actual pass (1), Red for actual fail (0)
passed = y_test == 1
failed = y_test == 0

plt.scatter(x_test[passed, 0], x_test[passed, 1], c='green', marker='o', label='Passed (Actual)')
plt.scatter(x_test[failed, 0], x_test[failed, 1], c='red', marker='x', label='Failed (Actual)')

# Calculate the Decision Boundary line
# The boundary is where w1*x1 + w2*x2 + b = 0
# So: x2 = -(w1*x1 + b) / w2
x1_coords = np.array([np.min(x_test[:, 0]), np.max(x_test[:, 0])])
x2_coords = -(w[0] * x1_coords + b) / w[1]

plt.plot(x1_coords, x2_coords, color='blue', linestyle='--', label='Decision Boundary')

plt.title("Model Performance on Test Data")
plt.xlabel("Normalized Study Hours")
plt.ylabel("Normalized Sleep Hours")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()








