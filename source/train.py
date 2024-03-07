import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load

class Layer_Dense:
    def __init__(self, nb_inputs, nb_neurons):
        self.weigths = 0.1 * np.random.rand(nb_inputs, nb_neurons)
        self.bias = np.zeros((1, nb_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weigths) + self.bias

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Sigmoid():
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

class Activation_Softmax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs), axis=1, keepdims=True)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 1:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def train():
    try:
        X_train = load("../data/X_train.csv")
        y_train = load("../data/y_train.csv")

        X = [[1, 2, 3 ,2.5],
             [2.0, 5.0, -1.0, 2.0],
             [-1.5, 2.7, 3.3, -0.8]]
        
        layer1 = Layer_Dense(2, 3)
        activation1 = Activation_ReLU()

        layer2 = Layer_Dense(3, 3)
        activation2 = Activation_Softmax()

        layer1.forward(X_train)
        activation1.forward(layer1.output)

        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        loss_function = Loss_CategoricalCrossEntropy()
        loss = loss_function.calculate(activation2.output, y_train)

        lowest_loss = 9999999
        best_dense1_weights = layer1.weights.copy()
        best_dense1_bias = layer1.bias.copy()
        best_dense2_weights = layer2.weights.copy()
        best_dense2_bias = layer2.bias.copy()

        for i in range(10000):
            
            layer1.weights += 0.05 * np.random.randn(2, 3)
            layer1.bias += 0.05 * np.random.randn(1, 3)
            layer2.weights += 0.05 * np.random.randn(2, 3)
            layer2.bias += 0.05 * np.random.randn(1, 3)

            layer1.forward(X_train)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)

            loss = loss_function.calculate(activation2.output, y_train)
            predictions = np.argmax(activation2.output, axis=1)
            accuracy = np.mean(predictions == y_train)

            if loss < lowest_loss:
                print(f"New set of weights found, iteration:{i}, loss:{loss}, accuracy:{accuracy}")
                best_dense1_weights = layer1.weights.copy()
                best_dense2_bias = layer1.bias.copy()
                best_dense1_weights = layer2.weights.copy()
                best_dense2_bias = layer2.bias.copy()
                lowest_loss = loss
            else:
                layer1.weigths = best_dense1_weights.copy()
                layer1.bias = best_dense1_bias.copy()
                layer2.weigths = best_dense2_weights.copy()
                layer2.bias = best_dense2_bias.copy()

    

    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    train()

if __name__ == "__main__":
    main()