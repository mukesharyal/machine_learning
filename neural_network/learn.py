import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Let's define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Let's define the neuron
class Neuron:

    # For a neuron, we need the number of inputs it receives and the bias
    def __init__(self, num_inputs, activation_function):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.activation_function = activation_function

    def __repr__(self):
        clean_weights = np.round(self.weights, 4)
        return f"Neuron(weight={clean_weights}, bias={self.bias:.4f})"

    def output(self, input):
        return self.activation_function(np.dot(self.weights, input) + self.bias)
    


# Let's define a layer of neurons
class Layer:

    # For a layer, we need the number of neurons and the number of inputs from the previous layer and the activation function
    def __init__(self, num_neurons, num_inputs, activation_function):
        self.neurons = [Neuron(num_inputs, activation_function) for _ in range(num_neurons)]

    def __repr__(self):
        return f"Layer = {self.neurons}\n"



# Let's define the neural network
# [neurons in input layer, number of hidden layers and neurons in each hidden layer, activation function of the hidden layer, neurons in output layer, activation function of the output layer]
# A neural network would be made like this:
# NeuralNetwork(10, [4 4], relu, 4, sigmoid)
# This means the network has 10 input neurons, then 2 hidden layers with 4 neurons in each layer, and 4 output neurons
class NeuralNetwork:

    def __init__(self, input_neurons, hidden_layers, hidden_activation_function, output_neurons, output_activation_function):

        # Our neural network is essentially an array of the various layers
        self.network = []

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
    

    def forward_pass(self, input):

        # Pass the input through each layer of the network, calculate the output and feed that to the next layer
        for layer in self.network:
            new_input = [neuron.output(input) for neuron in layer.neurons]
            input = new_input

        return input





nn = NeuralNetwork(4, [4, 4], relu, 4, sigmoid)
output = nn.forward_pass([2.0, 3.0, -4.0, 5.0])

print(output)


        


