# Multilayer Perceptron (MLP) Model

This project implements a Multilayer Perceptron (MLP) for classification tasks using Python. It includes components for data processing, model training, evaluation, and prediction.

## Features

- **Data Processing**: Splits and exports datasets for training and validation.
- **Model Architecture**: Configurable layers with ReLU and Softmax activations.
- **Training**: Supports early stopping, learning rate decay, and regularization.
- **Evaluation**: Calculates accuracy and loss on validation datasets.
- **Prediction**: Loads a trained model to make predictions on new data.
- **Metrics Management**: Saves training and validation metrics for analysis.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/jguigli/multilayer-perceptron.git
   cd multilayer-perceptron
   ```

2. **Install Dependencies**:
   - Ensure Python 3 is installed.
   - Run the setup command:
     ```bash
     make install
     ```

## Usage

- **Process Data**:
  ```bash
  make process
  ```

- **Train Model**:
  ```bash
  make train
  ```

- **Predict**:
  ```bash
  make predict
  ```

- **Clean Up**:
  - Remove generated files and directories:
    ```bash
    make clean
    ```

- **Full Clean**:
  - Remove all generated files, including virtual environment:
    ```bash
    make fclean
    ```

## Directory Structure

- `source/`: Contains the main source code for components and scripts.
- `data_sets/`: Directory for storing datasets.
- `saved_parameters/`: Stores model parameters.
- `saved_model/`: Stores the serialized model.
- `saved_metrics/`: Stores training and validation metrics.

## Notes

- Ensure that the `requirements.txt` file is up-to-date with all necessary Python packages.
- The project uses a virtual environment for dependency management. Activate it before running scripts manually.

## Important Notions

### Data Encoding
- **One-hot encoded**: [0, 1] (shows the position of the right target)
- **Sparse encoded**: 0 or 1 (binary)

### Neural Network Components
#### Feedforward
- **Layer**: Basic building block of neural networks
- **Activation Functions**:
  - Sigmoid
  - ReLU
  - Softmax
- **Loss**: Measure of prediction error

#### Backpropagation
- Chain Rule
- Derivative
- Partial derivative
- Gradient

#### Gradient Descent
- **Optimizer** (decreases loss):
  - Hyperparameters
  - SGD
  - AdaGrad
  - RMSProp
  - Adam
- Local minimum
- Global minimum
- Learning rate
- **Learning rate decay**:
  - Decay rate (decays the learning rate per batch or epoch)
- Momentum
- Gradient explosion

### Error Metrics
- Standard Error

## Useful Links

- [SUBJECT](https://cdn.intra.42.fr/pdf/pdf/112647/en.subject.pdf)
- [Wiki MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- [Wiki MLP FR](https://fr.wikipedia.org/wiki/Perceptron_multicouche)
- [Scikit learn MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Medium MLP explained](https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141)
- [Shiksa MLP explained](https://www.shiksha.com/online-courses/articles/understanding-multilayer-perceptron-mlp-neural-networks/)
- [SENTDEX YT MLP in python](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [Neural Network case study](https://cs231n.github.io/neural-networks-case-study/)
