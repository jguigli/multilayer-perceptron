from .activation import Activation_Softmax, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from .layer import Layer_Dense, Layer_Input
from .optimizer import Optimizer_AdaGrad, Optimizer_Adam, Optimizer_RMSprop, Optimizer_SGD
from .loss import Loss, Loss_BinaryCrossEntropy, Loss_CategoricalCrossEntropy

from utils import plot_learning_curves


class Multilayer_Perceptron:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

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

        self.trainable_layers = []

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()


    # def overfitting_detected(self, epoch):

    #     # If the model is overfitting, stop the training
    #     # (early stopping)

    #     if not self.check_overfitting:
    #         return False

    #     if epoch % self.early_stopping == 0:

    #         current_loss = self.validation_metrics["loss"][-1]

    #         # If the loss is not decreasing, stop the training
    #         if self.last_loss is not None and current_loss >= self.last_loss:
    #             return True

    #         self.last_loss = current_loss

    #     return False
    
    def fit(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        losses = []
        accuracies = []
        learning_rates = []
        
        self.accuracy.init(y)

        for epoch in range(1, epochs+1):

            # old_weights_layer1 = layer1.weights.copy()
            # old_weights_layer2 = layer2.weights.copy()

            # if self.overfitting_detected(epoch):
            #     print("\nOverfitting detected, training stopped\n")
            #     break

            #Rajouter validation

            output = self.forward(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

            #Early stopping
            # if np.array_equal(old_weights_layer1, layer1.weights) and np.array_equal(old_weights_layer2, layer2.weights):
            #     print("Break early stopping")
            #     break

            losses.append(loss)
            accuracies.append(accuracy)
            learning_rates.append(self.optimizer.current_learning_rate)

            if not epoch % print_every:
                #Rajouter la sauvegarde des metrics
                print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f} (' +
                f'data_loss: {data_loss:.3f}, ' +
                f'reg_loss: {regularization_loss:.3f}), ' +
                f' lr: {self.optimizer.current_learning_rate}')
            
            if validation_data is not None:
                X_validation, y_validation = validation_data

                output = self.forward(X_validation, training=False)

                loss = self.loss.calculate(output, y_validation)

                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_validation)
                
                if not epoch % print_every:
                    print(f'validation, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}')

         #Ajouter PLOT LOSS / Accuracy / learning rate
        plot_learning_curves(losses, 'Loss')
        plot_learning_curves(accuracies, 'Accuracy')
        plot_learning_curves(learning_rates, 'Learning rate')