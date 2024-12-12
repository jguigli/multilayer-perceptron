import pickle
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .activation import Activation_Softmax, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from .layer import Layer_Dense, Layer_Input
from .optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from .loss import Loss, Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy


class Multilayer_Perceptron:

    def __init__(self):
        self.layers = []
        self.best_loss = float('inf');
        self.patience_counter = 0

        self.epoch = []

        self.train_accuracies = []
        self.train_losses = []
        self.train_data_losses = []
        self.train_reg_losses = []
        self.train_learning_rates = []
        
        self.validation_accuracies = []
        self.validation_losses = []
    

    def add(self, layer):
        self.layers.append(layer)
    

    def set(self, *, loss, optimizer, accuracy):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy


    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)


    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)


    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))


    def save_model(self, path):
        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs','dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        
    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    

    def manage_metrics(self, validation_data):
        train_metrics = {
                'epoch': self.epoch,
                'accuracy': self.train_accuracies,
                'loss': self.train_losses,
                'data_loss': self.train_data_losses,
                'reg_loss': self.train_reg_losses,
                'learning_rate': self.train_learning_rates
        }

        df = pd.DataFrame(train_metrics)
        df.to_csv(f"../saved_metrics/train_metrics.csv", index=False)

        if validation_data is not None:
            validation_metrics = {
                    'epoch': self.epoch,
                    'accuracy': self.validation_accuracies,
                    'loss': self.validation_losses,
            }

            df = pd.DataFrame(validation_metrics)
            df.to_csv(f"../saved_metrics/validation_metrics.csv", index=False)

        print(f"\nSaving training and validation metrics in /saved_metrics")

    @staticmethod
    def standard_scaler(data):
        mean = np.mean(data)
        scale = np.std(data - mean)
        return (data - mean) / scale


    def early_stopping(self, val_loss, patience=5, delta=0.001):
        if val_loss < self.best_loss - delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                return True
        return False
    

    def plot_learning_curves(self, name, curves_train, curves_validation=None):
        if curves_train is not None:
            if curves_validation is None:
                plt.plot(range(len(curves_train)), curves_train)
            else:
                plt.plot(range(len(curves_train)), curves_train, curves_validation)
            plt.xlabel('Epochs')
            plt.ylabel(name)
            plt.show()
            plt.show()
    

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output


    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    

    def finalize(self):
        self.input_layer = Layer_Input()

        layer_count = len(self.layers)

        self.softmax_classifier_output = None

        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    def evaluate(self, X_val, y_val,*, batch_size=None):
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size

            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):

            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
            
        
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        
        return validation_loss, validation_accuracy
    

    def fit(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None, early_stopping=False, print_step=False, plot_curves=False):
        
        self.accuracy.init(y)

        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1

        for epoch in range(1, epochs + 1):
            if epoch % print_every == 0 :
                print(f'=============== epoch: {epoch} ===============')

            self.loss.new_pass()
            self.accuracy.new_pass()

            # Batch training
            for step in range(train_steps):

                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]


                output = self.forward(batch_X, training=True)
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                    self.optimizer.post_update_params()

                if epoch % print_every == 0 and print_step :
                    print(f'step    :    {step}, ' +
                        f'accuracy: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f' lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            #Save metrics
            self.epoch.append(epoch)
            self.train_losses.append(epoch_loss)
            self.train_data_losses.append(epoch_data_loss)
            self.train_reg_losses.append(epoch_regularization_loss)
            self.train_accuracies.append(epoch_accuracy)
            self.train_learning_rates.append(self.optimizer.current_learning_rate)

            # Print metrics
            if epoch % print_every == 0 :
                print(f'\ntraining    :' +
                    f'  accuracy: {epoch_accuracy:.3f}, ' +
                    f'loss: {epoch_loss:.3f} (' +
                    f'data_loss: {epoch_data_loss:.3f}, ' +
                    f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                    f'learning_rate: {self.optimizer.current_learning_rate}')
            
            #Validation
            if validation_data is not None:
                validation_loss, validation_accuracy = self.evaluate(*validation_data, batch_size=batch_size)

                if epoch % print_every == 0 :
                    print(f'validation  :' +
                        f'  accuracy: {validation_accuracy:.3f}, ' +
                        f'loss: {validation_loss:.3f}')
                    print()
                
                self.validation_losses.append(validation_loss)
                self.validation_accuracies.append(validation_accuracy)
            
            #Early stopping
            if early_stopping:
                if self.early_stopping(epoch_loss):
                    print(f"Early stopping at epoch {epoch}")
                    print(f'training    :' +
                        f'  accuracy: {epoch_accuracy:.3f}, ' +
                        f'loss: {epoch_loss:.3f} ')
                    
                    if validation_data is not None:
                        validation_loss = self.loss.calculate_accumulated()
                        validation_accuracy = self.accuracy.calculate_accumulated()
                        print(f'validation  :' +
                            f'  accuracy: {validation_accuracy:.3f}, ' +
                            f'loss: {validation_loss:.3f}')
                    break
        
        #Export metrics
        self.manage_metrics(validation_data)

        if plot_curves :
            self.plot_learning_curves('Loss', self.train_losses, self.validation_losses)
            self.plot_learning_curves('Accuracy', self.train_accuracies, self.validation_accuracies)
            self.plot_learning_curves('Learning rate', self.train_learning_rates)


    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size

        if prediction_steps * batch_size < len(X):
            prediction_steps += 1

        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)