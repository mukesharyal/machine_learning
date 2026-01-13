import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# We need to define the activation function, but we will also need the derivatives of them, se we will 
# have to define a class for the activation function itself
class ActivationFunction:

    def __init__(self, signature, derivative_signature):
        self.signature = signature
        self.derivative_signature = derivative_signature



#Let's also define the loss function class for the loss functions used
class LossFunction:

    def __init__(self, signature, derivative_signature):
        self.signature = signature
        self.derivative_signature = derivative_signature



# Let's define the activation functions
sigmoid = ActivationFunction(
    signature = lambda x: 1 / (1 + np.exp(-x)),
    derivative_signature=lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
)

relu = ActivationFunction(
    signature = lambda x: np.maximum(0, x),
    derivative_signature = lambda x: (x > 0).astype(float)
)



# Let's also define the loss functions
# We add 1e-15 (epsilon) to prevent log(0) which would result in NaN errors
bce_loss = LossFunction(
    signature=lambda target, output: -np.mean(
        target * np.log(output + 1e-15) + (1 - target) * np.log(1 - output + 1e-15)
    ),
    derivative_signature=lambda target, output: (
        (output - target) / (output * (1 - output) + 1e-15)
    )
)


# Let's define a layer of neurons
class Layer:

    # In the new scheme to allow for vectorization, we will have the layer as the smallest unit of the network
    def __init__(self, num_neurons, num_inputs, activation_function):

        # Now, we will have an array with num_neurons number of neurons, but instead of neurons we will directly have
        # the weights and biases in a matrix
        # Instead of using a for loop, we can use numpy to initialise all the random values at once
        # This function will generate a matrix of dimensions [num_neurons] X [num_inputs]
        # We are using the randn for the values to be taken from a normal distribution with mean 0 and sigma 1
        # Also, we are normalizing the values so that they are not very large
        self.W = np.random.randn(num_neurons, num_inputs) * np.sqrt(1.0 / num_inputs)
        self.B = np.zeros((num_neurons, 1))
        self.activation_function = activation_function


    def __repr__(self):
        return f"Layer W = {self.W.shape}, Layer B = {self.B.shape}\n"
    

    def output(self, input):

        # Let's store the input to the layer as we might need it
        self.inputA = input

        # Let's also store the thereby formed z
        self.Z = np.dot(self.W, input) + self.B

        # Let's also store the output value produced by the layer as well
        self.A = self.activation_function.signature(self.Z)

        return self.A


    



# Let's define the neural network
# [neurons in input layer, number of hidden layers and neurons in each hidden layer, activation function of the hidden layer, neurons in output layer, activation function of the output layer]
# A neural network would be made like this:
# NeuralNetwork(10, [4 4], relu, 4, sigmoid)
# This means the network has 10 input neurons, then 2 hidden layers with 4 neurons in each layer, and 4 output neurons
class NeuralNetwork:

    def __init__(
            self, 
            input_neurons, 
            hidden_layers, 
            hidden_activation_function, 
            output_neurons, 
            output_activation_function,
            loss_function
            ):

        # Our neural network is essentially an array of the various layers
        self.network = []

        # We just assign the loss function to the network as well
        self.loss_function = loss_function

        # Let's not have the input layer as one layer because it doesn't need weights and biases
        # Then, the layers used are the hidden layers

        # For the number of inputs in each layer, we need the number of neurons in the previous layer
        # For the first hidden layer, the number of neurons in the previous layer is the number of input neurons
        num_inputs = input_neurons

        for i in range(len(hidden_layers)):

            self.network.append(Layer(hidden_layers[i], num_inputs, hidden_activation_function))

            num_inputs = hidden_layers[i]
        
        self.network.append(Layer(output_neurons, num_inputs, output_activation_function))

    
    def __repr__(self):
        return f"Network = [{self.network}]"
    

    def forward_pass(self, input, target):

        for i in range(len(self.network)):

            output = self.network[i].output(input)

            input = output

        # Here we have the output for the layer after the whole forward pass
        loss = self.loss_function.signature(target, output)

        # Let's also store the value of the final del, so we can just loop backwards in the backpropagation step
        final_layer = self.network[len(self.network) - 1]

        self.delta = (
            self.loss_function.derivative_signature(target, final_layer.A) *
            final_layer.activation_function.derivative_signature(final_layer.Z)
        )

    
    def backpropagate(self, learning_rate):

        for i in reversed(range(len(self.network))):

            # In forward pass, we have the delta for the final layer, so we just change it till we reach the input
            current_delta = self.delta

            layer = self.network[i]

            # For every layer (coming from the back), we find the gradients for that layer as
            gradW = np.dot(current_delta, layer.inputA.T)
            gradB = current_delta

            # While we haven't reached the input layer, we need to change the current delta
            if i > 0:

                previous_layer = self.network[i-1]

                self.delta = np.dot(layer.W.T, current_delta) * previous_layer.activation_function.derivative_signature(previous_layer.Z)

            layer.W -= learning_rate * gradW
            layer.B -= learning_rate * gradB

            print(f"W is {layer.W}, B is {layer.B}")




# Let's also import the data for testing
import os

# Get the path of the CSV file
script_dir = os.path.dirname(__file__) 
file_path = os.path.join(script_dir, 'simpler_data.csv')

# Fetch the CSV file using Pandas
data = pd.read_csv(file_path)

# Get the first row of the data
first_row_input = data.iloc[0, :2].to_numpy()
first_row_target = data.iloc[0]['y']

column_matrix = first_row_input[:, None]

nn = NeuralNetwork(2, [4, 3], relu, 1, sigmoid, bce_loss)

nn.forward_pass(column_matrix, first_row_target)

nn.backpropagate(0.001)


        


