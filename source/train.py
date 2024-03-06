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


def train():
    try:
        X_train = load("../data/X_train.csv")
        y_train = load("../data/y_train.csv")

        X = [[1, 2, 3 ,2.5],
             [2.0, 5.0, -1.0, 2.0],
             [-1.5, 2.7, 3.3, -0.8]]
        
        layer1 = Layer_Dense(2,5)
        activation1 = Activation_ReLU()
        layer1.forward(X_train)
        activation1.forward(layer1.output)
    

    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    train()

if __name__ == "__main__":
    main()