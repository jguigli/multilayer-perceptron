import pandas as pd

from components.model import Multilayer_Perceptron
from components.activation import Activation_Softmax, Activation_ReLU, Activation_Sigmoid
from components.layer import Layer_Dense, Layer_Dropout
from components.optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from components.loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy
from components.accuracy import Accuracy_Categorical

from utils import load, standard_scaler


def predict():
    try:
        X = load("../data_sets/X_validation.csv")
        y = load("../data_sets/y_validation.csv")

        X_scale = standard_scaler(X.values)
        y_one_hot = pd.get_dummies(y, dtype=int).values
        # y_sparse = y.replace({'B': 1, 'M':0}).values.reshape(-1)

        model = Multilayer_Perceptron()

        model.add(Layer_Dense(X_scale.shape[1], 30))
        model.add(Activation_ReLU())
        # model.add(Layer_Dropout(0.1))
        model.add(Layer_Dense(30, 30))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(30, 2))
        model.add(Activation_Softmax())

        model.set(
            loss=Loss_CategoricalCrossEntropy(),
            optimizer=Optimizer_Adam(),
            accuracy=Accuracy_Categorical()
        )

        model.finalize()

        model.load_parameters('../saved_parameters/mlp.params')

        loss, accuracy = model.evaluate(X_scale, y_one_hot)
        print(f"predict, " +
            f'  accuracy: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (')

    except Exception as e:
        print(f"Error : {str(e)}")
        return


def main():
    predict();

if __name__ == "__main__":
    main()