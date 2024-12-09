from components.model import Multilayer_Perceptron
from components.activation import Activation_Softmax, Activation_ReLU
from components.layer import Layer_Dense, Layer_Input
from components.optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from components.loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy
from components.accuracy import Accuracy_Categorical
from utils import load, scaler


def train():

    X_train = load("../data_sets/dataset_train.csv")
    y_train = load("../data_sets/dataset_train.csv")
    X_validation = load("../data_sets/dataset_train.csv")
    y_validation = load("../data_sets/dataset_train.csv")

    X_train_scale = scaler(X_train)
    X_validation_scale = scaler(X_validation)

    model = Multilayer_Perceptron()

    model.add(Layer_Dense(1, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(1, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(1, 64))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossEntropy(),
        optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-4),
        accuracy=Accuracy_Categorical()
    )

    model.finalize()

    model.fit(X_train_scale, y_train, epochs=10000, print_every=100, validation_data=(X_validation_scale, y_validation))


def main():
    train();

if __name__ == "__main__":
    main()
