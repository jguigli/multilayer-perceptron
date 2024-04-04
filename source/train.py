import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, vertical_data
from utils import load, plot_learning_curves

nnfs.init()

class MultilayerPerceptron:
    def __init__(self): ######################################################3
        self.lol = 1
    class Layer_Dense:
        def __init__(self, nb_inputs, nb_neurons):
            self.weights = 0.1 * np.random.rand(nb_inputs, nb_neurons)
            self.biases = np.zeros((1, nb_neurons))
        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.dot(inputs, self.weights) + self.biases
        def backward(self, dvalues):
            self.dweights = np.dot(self.inputs.T, dvalues)
            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
            self.dinputs = np.dot(dvalues, self.weights.T)

    class Activation_ReLU():
        def forward(self, inputs):
            self.inputs = inputs
            self.output = np.maximum(0, inputs)
        def backward(self, dvalues):
            self.dinputs = dvalues.copy()
            self.dinputs[self.inputs <= 0] = 0

    class Activation_Sigmoid():
        def forward(self, inputs):
            self.output = 1 / (1 + np.exp(-inputs))

    class Activation_Softmax():
        def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities
        def backward(self, dvalues): # REVOIR EXPLICATION
            self.dinputs = np.empty_like(dvalues)

            for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
                single_output = single_output.reshape(-1 ,1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
            negative_log = -np.log(correct_confidences)
            return negative_log
        def backward(self, dvalues, y_true): # REVOIR EXPLICATION
            samples = len(dvalues)
            labels = len(dvalues[0])

            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]
            
            self.dinputs = -y_true / dvalues
            self.dinputs = self.dinputs / samples
        
    class Loss_BinaryCrossEntropy(Loss):
        def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

            sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
            sample_losses = np.mean(sample_losses, axis=-1)
            return sample_losses
        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            outputs = len(dvalues[0])
            clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
            self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
            self.dinputs = self.dinputs / samples

    class Activation_Softmax_Loss_CategoricalCrossentropy():
        """Increase chain rule derivative of Softmax and Loss function"""
        def __init__(self):
            self.activation = MultilayerPerceptron.Activation_Softmax()
            self.loss = MultilayerPerceptron.Loss_CategoricalCrossEntropy()
        def forward(self, inputs, y_true):
            self.activation.forward(inputs)
            self.output = self.activation.output
            return self.loss.calculate(self.output, y_true)
        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            
            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples

    class Optimizer_SGD:
        def __init__(self, learning_rate=1., decay=0., momentum=0.):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.momentum = momentum
        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        def update_params(self,layer):
            if self.momentum:
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)

                weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
                layer.weight_momentums = weight_updates
                layer.bias_momentums = bias_updates
            else:
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases
            layer.weights += weight_updates
            layer.biases += bias_updates
        def post_update_params(self):
            self.iterations += 1
        
    class Optimizer_AdaGrad:
        def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.epsilon = epsilon
        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        def update_params(self,layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2

            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        def post_update_params(self):
            self.iterations += 1

    class Optimizer_RMSprop:
        def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.epsilon = epsilon
            self.rho = rho
        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        def update_params(self,layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
            layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

            layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        def post_update_params(self):
            self.iterations += 1
        
    class Optimizer_Adam:
        def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.epsilon = epsilon
            self.beta_1 = beta_1
            self.beta_2 = beta_2
        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        def update_params(self,layer):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

            weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
            
            layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

            weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
            bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

            layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        def post_update_params(self):
            self.iterations += 1

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
    model = MultilayerPerceptron()
    model.fit()

if __name__ == "__main__":
    main()