import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# Get the path of the CSV file
script_dir = os.path.dirname(__file__) 
file_path = os.path.join(script_dir, 'data.csv')

# Fetch the CSV file using Pandas
data = pd.read_csv(file_path)

# Extract the various columns into NumPy arrays using Pandas 
# We can also scale each of the values quite simply as if x and y were a matrix and we were scaling them
x = data["size"].values / 1000.0
y = data["price"].values / 100000.0

# We define the parameters w and b and set them to some initial value, say 10
w = 10
b = 10

# We also define a fixed learning rate of 0.01
alpha = 0.01

# For now, let's define a fixed number of times for which the algorithm runs
iterations = 10000

# We define the cost array to see the progression of the cost with number of iterations
cost = np.zeros(iterations)

count = 0

# This is where the training takes place
while(count < iterations):

    # Since x is a NumPy array it has a shape attribute which returns a tuple like (Rows, Columns)
    m = x.shape[0]

    # Define the predicted array as y_cap
    y_cap = w * x + b

    cost[count] = (1 / (2 * m)) * np.sum(np.square(y_cap - y))

    # Perform the gradient descent
    w_next = w - (alpha / m) * np.sum((y_cap - y) * x)
    b_next = b - (alpha / m) * np.sum((y_cap - y))

    # Update the parameters
    w = w_next
    b = b_next

    count = count + 1


# When the algorithm is complete, we find 2 points to draw our predicted line
x_pred = np.array([min(x), max(x)])
y_pred = w * x_pred + b

# We define 2 plots for the actual data plot and the cost plot with number of iterations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the actual data points (the red X's)
ax1.scatter(x, y, marker = 'x', color = 'red', label =' Actual Data')

# Plot the regression line (the blue line)
ax1.plot(x_pred, y_pred, color = 'blue', linewidth = 2, label = 'Linear Regression Fit')

ax1.set_title("House Prices vs. Living Area")
ax1.set_xlabel("Size (sqft)")
ax1.set_ylabel("Price (USD)")
ax1.grid(True, linestyle = '--', alpha = 0.6)
ax1.legend()

ax2.plot(cost, color='green')
ax2.set_title("Cost vs. Iterations")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost (J)")
ax2.set_yscale('log')

plt.tight_layout() # Prevents labels from overlapping
plt.show()
