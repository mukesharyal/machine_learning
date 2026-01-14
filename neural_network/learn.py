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
# The Sigmoid function
sigmoid = ActivationFunction(
    signature = lambda x: 1 / (1 + np.exp(-x)),
    derivative_signature=lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
)

# The ReLU function
relu = ActivationFunction(
    signature = lambda x: np.maximum(0, x),
    derivative_signature = lambda x: (x > 0).astype(float)
)

# The SoftMax function
softmax = ActivationFunction(
    signature=lambda x: (
        np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)
    ),
    
    # We won't actually use the derivative for the loss as we will pair it with the CE Loss function for an 
    # elegant expression for the loss
    derivative_signature=lambda x: (
        (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)) * (1 - (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=0)))
    )
)





# Let's also define the loss functions
# Binary Cross Entropy Loss function
bce_loss = LossFunction(
    signature=lambda target, output: -np.mean(
        target * np.log(output + 1e-15) + (1 - target) * np.log(1 - output + 1e-15)
    ),

    # We can also pair this with the sigmoid function to have another elegant loss function
    derivative_signature=lambda target, output: (
        (output - target) / (output * (1 - output) + 1e-15)
    )
)


# Categorical Cross-Entropy Loss Function
cce_loss = LossFunction(
    signature=lambda target, output: -np.sum(
        target * np.log(output + 1e-15)
    ),
    derivative_signature=lambda target, output: (
        -(target / (output + 1e-15))
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


    # Let's define the function that actually feeds the input forward to the network to produce the output
    def feed_forward(self, input):

        for i in range(len(self.network)):

            output = self.network[i].output(input)

            input = output

        return output


    def _forward_pass(self, input, target):

        output = self.feed_forward(input)

        self.loss = self.loss_function.signature(target, output)

        # CHECK: If using Softmax + CCE, the derivative is just (Output - Target)
        # This is numerically stable and avoids the 'increasing loss' issue.
        if self.network[-1].activation_function == softmax and self.loss_function == cce_loss:
            self.delta = output - target
        else:
            # Standard chain rule for other combinations (like Sigmoid/BCE)
            final_layer = self.network[-1]
            self.delta = (
                self.loss_function.derivative_signature(target, final_layer.A) *
                final_layer.activation_function.derivative_signature(final_layer.Z)
            )

        return output

    
    def _backpropagate(self, learning_rate):

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

                # Propagate the error to the next layer
                self.delta = np.dot(layer.W.T, current_delta) * previous_layer.activation_function.derivative_signature(previous_layer.Z)
            
            layer.W -= learning_rate * gradW
            layer.B -= learning_rate * gradB


    # The forward_pass and the backward_pass are the internal functions for training
    # Now, we actually define the train function exposed via the API
    def train(self, input, target, learning_rate):

        # First, perform forward pass
        self._forward_pass(input, target)

        # Now, perform backward pass with the learning rate
        self._backpropagate(learning_rate)

        # Now, we return the loss produced by the forward pass
        return self.loss


import os


# Import the MnistDataLoader to actually load the MNIST data to train our model
from mnist_helper import MnistDataloader
# 1. Get the directory where your script is located
script_dir = os.path.dirname(__file__)

# 2. Define the path to the 'mnist' folder relative to this script
input_path = os.path.join(script_dir, 'mnist')
training_images_filepath = f'{input_path}/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = f'{input_path}/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = f'{input_path}/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = f'{input_path}/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_list, y_train_list), (x_test_list, y_test_list) = dataloader.load_data()

# Helper function to One-Hot Encode
def one_hot(labels, classes=10):
    labels = np.array(labels)
    one_hot_matrix = np.zeros((classes, labels.size))
    one_hot_matrix[labels, np.arange(labels.size)] = 1
    return one_hot_matrix

# Process Inputs: Convert to NumPy, Flatten to 784, Transpose to (784, N), and Normalize
train_inputs = np.array(x_train_list).reshape(len(x_train_list), -1).T / 255.0
test_inputs = np.array(x_test_list).reshape(len(x_test_list), -1).T / 255.0

# Process Outputs: One-Hot Encode and ensure shape is (10, N)
train_outputs = one_hot(y_train_list)
test_outputs = one_hot(y_test_list)




# Let's define our neural network
nn = NeuralNetwork(784, [128, 64, 16], relu, 10, softmax, cce_loss)

# Let's create an empty array that holds the loss as the training progresses
loss_history = []

for epoch in range(100):
    
    # Let's store the loss of each epoch
    epoch_loss = 0.0

    for i in range(train_inputs.shape[1]):
        # Get the i-th sample as a column vector (2, 1)
        x = train_inputs[:, i:i+1]
        # Get the i-th label as a column vector (1, 1)
        y = train_outputs[:, i:i+1]

        # Now, the train function returns the loss of the netork for the iteration
        loss = nn.train(x, y, 0.01)

        # Add the loss to the epoch loss
        epoch_loss += loss

    # After each epoch, calculate the average loss of the epoch
    average_loss = epoch_loss / train_inputs.shape[1]
    loss_history.append(average_loss)

    # Also test the accuracy of the network, say after each 20 epochs
    if(epoch % 20 == 0):
        # Forward pass the whole subset
        predictions = nn.feed_forward(train_inputs) 
        
        # Get index of highest probability for each column
        y_pred_labels = np.argmax(predictions, axis=0)
        y_true_labels = np.argmax(train_outputs, axis=0)
        
        accuracy = np.mean(y_pred_labels == y_true_labels) * 100
        print(f"Epoch {epoch} | Loss: {average_loss:.4f} | Accuracy: {accuracy:.2f}%")

# Plot the results after training is finished
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

    






        


