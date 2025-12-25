import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# Get the path of the CSV file
script_dir = os.path.dirname(__file__) 
file_path = os.path.join(script_dir, 'data.csv')

# Load the CSV file using Pandas
data = pd.read_csv(file_path)

# Initialise the NumPy arrays from the data
# The x array is an array of vectors with three different values and thus a shape of (100, 3)
x_data = data[["schooling", "health_expenditure", "gdp_per_capita"]].values

# Before proceeding further, normalize the data using z-score normalization
# The axis = 0 tells that the maximum is calculated for each column and NOT the overall maximum value
mu = np.mean(x_data, axis=0)
sigma = np.std(x_data, axis=0)
x = (x_data - mu) / sigma

# The y array is just a regular array with shape (100, 1)
# Normalize it by some max value like 80 years
y = data["life_expectancy"].values / 80

# This time, the model looks like f(x) = w1x1 + w2x2 + w3x3 + b
# In vector form, that is f(x) = W dot X + b
# Let's define some initial values for the W vector like
# TIP: Always define the initial parameters as floating point numbers to prevent rounding off errors
w = np.array([10.0, 10.0, 10.0])

# Let's define a fixed learning rate
alpha = 0.01

# Let's also initialize b as some random number
b = 10.0

# Let's define the number of iterations to something like 10000
iterations = 10000

count = 0

# Initialize a cost array for tracking the progression of the cost with number of iterations
cost = np.zeros(iterations)

# Initialize m and n for the calculations
m, n = x.shape

# Since we need a vector for the parameters as well, we define it here
w_next = np.zeros(n)

while(count < iterations):

    cost[count] = (1 / (2 * m)) * np.sum(np.square(np.dot(x, w) + b - y))

    for i in range(0, n):

        # We need to account for all the rows so that the sum method can work for us
        # x[:, i] gives us all the rows for the ith column, which is exactly what we need

        w_next[i] = w[i] - (alpha / m) * np.sum((np.dot(x, w) + b - y) * x[:, i])

    b_next = b - (alpha / m) * np.sum(np.dot(x, w) + b - y)

    for i in range(0, 3):

        w[i] = w_next[i]
    
    b = b_next

    count += 1


print(w, b)

plt.plot(cost)
plt.title("Learning Curve")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.yscale("log")
plt.show()

