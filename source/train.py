from components.model import Multilayer_Perceptron
from components.activation import Activation_Softmax, Activation_ReLU
from components.layer import Layer_Dense, Layer_Input
from components.optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from components.loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy
from components.accuracy import Accuracy_Categorical
from utils import load, standard_scaler
import pandas as pd

# import numpy as np
# import nnfs
# from nnfs.datasets import spiral_data

# nnfs.init()


# X, y = spiral_data(samples=100, classes=3)

def train():

    X_train = load("../data_sets/X_train.csv")
    y_train = load("../data_sets/y_train.csv")
    X_validation = load("../data_sets/X_validation.csv")
    y_validation = load("../data_sets/y_validation.csv")

    X_train_scale = standard_scaler(X_train.values)
    X_validation_scale = standard_scaler(X_validation.values)

    y_train_one_hot = pd.get_dummies(y_train, dtype=int).values
    y_validation_one_hot = pd.get_dummies(y_validation, dtype=int).values

    model = Multilayer_Perceptron()

    model.add(Layer_Dense(30, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 2))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossEntropy(),
        optimizer=Optimizer_Adam(learning_rate=0.001),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()

    model.fit(X_train_scale, y_train_one_hot, epochs=10000, print_every=100, validation_data=(X_validation_scale, y_validation_one_hot))


def main():
    train();

if __name__ == "__main__":
    main()
