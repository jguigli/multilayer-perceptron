import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, vertical_data
from utils import load, plot_learning_curves

nnfs.init()



#############################################################################
#############################################################################
#############################################################################


def fit(self):
    X_train, y_train = spiral_data(samples=100,classes=3)
    # X_train, y_train = vertical_data(samples=100, classes=3)
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    # plt.show()

    losses = []
    accuracies = []
    learning_rates = []
    
    layer1 = self.Layer_Dense(2, 64)
    activation1 = self.Activation_ReLU()
    layer2 = self.Layer_Dense(64, 3)
    loss_activation = self.Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = self.Optimizer_Adam(learning_rate=0.001, decay=1e-4)
    
    for epoch in range(10001):

        old_weights_layer1 = layer1.weights.copy()
        old_weights_layer2 = layer2.weights.copy()

        #Feed forward
        layer1.forward(X_train)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        loss = loss_activation.forward(layer2.output, y_train)

        predictions = np.argmax(loss_activation.output, axis=1)

        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(predictions == y_train)
        
        if not epoch % 100:
            #Rajouter la sauvegarde des metrics
            print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')


        # Backward pass
        loss_activation.backward(loss_activation.output, y_train)
        layer2.backward(loss_activation.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.post_update_params()

        #Early stopping
        if np.array_equal(old_weights_layer1, layer1.weights) and np.array_equal(old_weights_layer2, layer2.weights):
            print("Break early stopping")
            break

        losses.append(loss)
        accuracies.append(accuracy)
        learning_rates.append(optimizer.current_learning_rate)
    
    #Ajouter PLOT LOSS / Accuracy / learning rate
    plot_learning_curves(losses, 'Loss')
    plot_learning_curves(accuracies, 'Accuracy')
    plot_learning_curves(learning_rates, 'Learning rate')


def predict(self):
    lol = 1

def main():
    lol = 1

if __name__ == "__main__":
    main()

