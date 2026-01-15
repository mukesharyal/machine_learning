import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import our new nano library mytorch for the creation and training of the neural network
import mytorch as mt

# Import the MnistDataLoader
from mnist_helper import MnistDataloader

# Load the MNIST dataset
script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, 'mnist')
training_images_filepath = f'{input_path}/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = f'{input_path}/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = f'{input_path}/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = f'{input_path}/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_list, y_train_list), (x_test_list, y_test_list) = dataloader.load_data()

# Helper function to for one-hot encoding
def one_hot(labels, classes=10):
    labels = np.array(labels)
    one_hot_matrix = np.zeros((classes, labels.size))
    one_hot_matrix[labels, np.arange(labels.size)] = 1
    return one_hot_matrix

# Process the inputs
train_inputs_full = np.array(x_train_list).reshape(len(x_train_list), -1).T / 255.0
test_inputs_full = np.array(x_test_list).reshape(len(x_test_list), -1).T / 255.0

# Process the outputs
train_outputs_full = one_hot(y_train_list)
test_outputs_full = one_hot(y_test_list)

# Only pick a small sample (because my laptop wasn't happy when I tried to train on the whole 60K examples)
train_inputs = train_inputs_full[:, :10000]
train_outputs = train_outputs_full[:, :10000]
test_inputs = test_inputs_full[:, :1000]
test_outputs = test_outputs_full[:, :1000]


# Define the layers individually as per our new library structure
l1 = mt.Layer(128, 784, mt.activation.relu)
l2 = mt.Layer(64, 128, mt.activation.relu)
l3 = mt.Layer(10, 64, mt.activation.softmax)

# Create the network by passing the list of layers
nn = mt.NeuralNetwork([l1, l2, l3], mt.loss.cce_loss)

# Some hyperparameters
initial_lr = 0.01
epochs = 100
loss_history = []
num_samples = train_inputs.shape[1]


for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Implement learning rate decay
    current_lr = initial_lr * (0.5 ** (epoch // 25))
    
    # Shuffle the data for each epoch
    indices = np.random.permutation(num_samples)
    shuffled_inputs = train_inputs[:, indices]
    shuffled_outputs = train_outputs[:, indices]

    for i in range(num_samples):
        x = shuffled_inputs[:, i:i+1]
        y = shuffled_outputs[:, i:i+1]

        # The train function in the library returns the loss for the particular training
        loss = nn.train(x, y, current_lr)
        epoch_loss += loss

    average_loss = epoch_loss / num_samples
    loss_history.append(average_loss)

    # Test accuracy every 10 epochs
    if(epoch % 10 == 0):
        predictions = nn.feed_forward(test_inputs) 
        y_pred_labels = np.argmax(predictions, axis=0)
        y_true_labels = np.argmax(test_outputs, axis=0)
        
        accuracy = np.mean(y_pred_labels == y_true_labels) * 100
        print(f"Epoch {epoch:02d} | LR: {current_lr:.4f} | Loss: {average_loss:.4f} | Test Acc: {accuracy:.2f}%")



# Plot the final results
final_preds = nn.feed_forward(test_inputs)
final_acc = np.mean(np.argmax(final_preds, axis=0) == np.argmax(test_outputs, axis=0)) * 100
print(f"\nTraining Complete! Final Test Accuracy: {final_acc:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss with Learning Rate Decay')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()