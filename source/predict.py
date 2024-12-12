import pandas as pd

from components.model import Multilayer_Perceptron
from components.activation import Activation_Softmax, Activation_ReLU, Activation_Sigmoid
from components.layer import Layer_Dense, Layer_Dropout
from components.optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from components.loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy
from components.accuracy import Accuracy_Categorical


def predict():
    try:
        print(f"Loading dataset ...")
        X = pd.read_csv("../data_sets/X_validation.csv")
        y = pd.read_csv("../data_sets/y_validation.csv")

        X_scale = Multilayer_Perceptron.standard_scaler(X.values)
        y_one_hot = pd.get_dummies(y, dtype=int).values

        print(f"Loading model ...")
        model = Multilayer_Perceptron.load_model('../saved_model/mlp.model')

        loss, accuracy = model.evaluate(X_scale, y_one_hot)

        print(f"\nPrediction on data set to predict : " +
            f'  accuracy: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} ')

    except Exception as e:
        print(f"Error : {str(e)}")
        return


def main():
    predict();

if __name__ == "__main__":
    main()