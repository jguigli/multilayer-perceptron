import pandas as pd

from components.model import Multilayer_Perceptron
from components.activation import Activation_Softmax, Activation_ReLU
from components.layer import Layer_Dense, Layer_Dropout
from components.optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from components.loss import Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy
from components.accuracy import Accuracy_Categorical



def train():
    try:
        print(f"Loading training and validation datasets ...")
        X_train = pd.read_csv("../data_sets/X_train.csv")
        y_train = pd.read_csv("../data_sets/y_train.csv")
        X_validation = pd.read_csv("../data_sets/X_validation.csv")
        y_validation = pd.read_csv("../data_sets/y_validation.csv")

        #Scaling
        X_train_scale = Multilayer_Perceptron.standard_scaler(X_train.values)
        X_validation_scale = Multilayer_Perceptron.standard_scaler(X_validation.values)

        #One hot encoded
        y_train_one_hot = pd.get_dummies(y_train, dtype=int).values
        y_validation_one_hot = pd.get_dummies(y_validation, dtype=int).values


        model = Multilayer_Perceptron()

        model.add(Layer_Dense(X_train_scale.shape[1], 36))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(36, 36))
        model.add(Activation_ReLU())
        model.add(Layer_Dense(36, 2))
        model.add(Activation_Softmax())

        model.set(
            loss=Loss_BinaryCrossEntropy(),
            optimizer=Optimizer_Adam(),
            accuracy=Accuracy_Categorical()
        )

        model.finalize()

        model.fit(X_train_scale,
                y_train_one_hot,
                epochs=200,
                print_every=10,
                batch_size=5,
                validation_data=(X_validation_scale, y_validation_one_hot),
                early_stopping=True,
                print_step=False,
                plot_curves=False,
                )
        
        print(f"Saving parameters in /saved_parameters")
        model.save_parameters('../saved_parameters/mlp.params')
        print(f"Saving topology model in /saved_model")
        model.save_model('../saved_model/mlp.model')

    except Exception as e:
        print(f"Error : {str(e)}")
        return



def main():
    train();


if __name__ == "__main__":
    main()
