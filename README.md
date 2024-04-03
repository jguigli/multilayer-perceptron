# multilayer-perceptron

# LINKS

[SUBJECT](https://cdn.intra.42.fr/pdf/pdf/112647/en.subject.pdf)  
[Wiki MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron)  
[Wiki MLP FR](https://fr.wikipedia.org/wiki/Perceptron_multicouche)  
[Scikit learn MLP](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)  
[Medium MLP explained](https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141)  
[Shiksa MLP explained](https://www.shiksha.com/online-courses/articles/understanding-multilayer-perceptron-mlp-neural-networks/)  
[SENTDEX YT MLP in python](https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)  
[Neural Network case study](https://cs231n.github.io/neural-networks-case-study/)  
[]()  
[]()  
[]()  


# GUIDELINE

Data processing :
- Recuperer le fichier .csv
- Mettre les bons nom de colonnes trouve ici (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)
- Enlever les features inutiles (si necessaire)
- Creer un set de train (80%)
- Creer un set de test (20%)

Training phase :
- Recuperer le set de train csv
- Utilisation de la backprogation et de la descente de gradient pour obtenir les parametres
- Sauvegarder les parametres a la fin de la phase de training
- Afficher a chaque periode, les metriques d'entrainements et de validations
- Afficher le cout et la precision (learning curves) a la fin du training

Predict phase :
- Recuperer le set de test csv
- Recuperer les parametres
- Predire les valeurs
- Evaluer les valeurs obtenues a partir de la fonction d'erreur binary cross-entropy (https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_error_function_and_logistic_regression)



# Important Notions

Feedforward
	Layer
	Activation function
		Sigmoid
		ReLu
		Softmax
	Loss
Backpropagation
	Chain Rule
	Derivative
	Partial derivative
	Gradient
Gradient descent
	Optimizer -> decrease loss
		Hyperparameters
		SGD
		AdaGrad
		RMSProp
		Adam
	Local minimum
	Global minimum
	Learning rate
	Learning rate decay
		Decay rate -> decays the learning rate per batch or epoch
	Momentum
	Gradient explosion
Standard Error


# Bonus

- A more complex optimization function (for example : nesterov momentum, RMSprop, Adam, ...).
- A display of multiple learning curves on the same graph (really useful to compare different models).
- An historic of the metric obtained during training.
- The implementation of early stopping.
- Evaluate the learning phase with multiple metrics.


# Comprehension

## Introduction

### Dense layer : Weight and bias

Dense layers, the most common layers, consist of interconnected neurons. In a dense layer, each
neuron of a given layer is connected to every neuron of the next layer, which means that its output
value becomes an input for the next neurons. Each connection between neurons has a weight
associated with it, which is a trainable factor of how much of this input to use, and this weight
gets multiplied by the input value. Once all of the ​inputs·weights​ flow into our neuron, they are
summed, and a bias, another trainable parameter, is added. The purpose of the bias is to offset the
output positively or negatively, which can further help us map more real-world types of dynamic
data.

Biases and weights are both tunable parameters, and both will impact the neurons’ outputs, but
they do so in different ways. Since weights are multiplied, they will only change the magnitude or
even completely flip the sign from positive to negative, or vice versa. ​Output = weight·input+bias
is not unlike the equation for a line ​y = mx+b​. We can visualize this with:

### Step function

For a step function, if the neuron’s output value, which is calculated by ​sum(inputs · weights)
+ bias​, is greater than 0, the neuron fires (so it would output a 1). Otherwise, it does not fire
and would pass along a 0. The formula for a single neuron might look something like:
output ​= ​sum​(inputs ​* ​weights) ​+ ​bias
We then usually apply an activation function to this output, noted by ​activation()​:
output ​= ​activation(output)
While you can use a step function for your activation function, we tend to use something slightly
more advanced. Neural networks of today tend to use more informative activation functions
(rather than a step function), such as the ​Rectified Linear​ (ReLU) activation function, which we
will cover in-depth in Chapter 4. Each neuron’s output could be a part of the ending output layer,
as well as the input to another layer of neurons. While the full function of a neural network can
get very large, let’s start with a simple example with 2 hidden layers of 4 neurons each.

### Input / Output

Along with these 2 hidden layers, there are also two more layers here — the input and output
layers. The input layer represents your actual input data, for example, pixel values from an image
or data from a temperature sensor. While this data can be “raw” in the exact form it was collected,
you will typically ​preprocess​ your data through functions like ​normalization​ and ​scaling​, and
your input needs to be in numeric form. Concepts like scaling and normalization will be covered
later in this book. However, it is common to preprocess data while retaining its features and
having the values in similar ranges between 0 and 1 or -1 and 1. To achieve this, you will use
either or both scaling and normalization functions. The output layer is whatever the neural
network returns. With classification, where we aim to predict the class of the input, the output
layer often has as many neurons as the training dataset has classes, but can also have a single
output neuron for binary (two classes) classification. We’ll discuss this type of model later and,
for now, focus on a classifier that uses a separate output neuron per each class. For example, if
our goal is to classify a collection of pictures as a “dog” or “cat,” then there are two classes in
total. This means our output layer will consist of two neurons; one neuron associated with “dog”
and the other with “cat.” You could also have just a single output neuron that is “dog” or “not
dog.”

### Purpose of the model

The concept of a long function with millions of variables that could be used to solve
a problem isn’t all too difficult. With that many variables related to neurons, arranged as
interconnected layers, we can imagine there exist some combinations of values for these variables
that will yield desired outputs. Finding that combination of parameter (weight and bias) values is
the challenging part.
The end goal for neural networks is to adjust their weights and biases (the parameters), so when
applied to a yet-unseen example in the input, they produce the desired output. When supervised
machine learning algorithms are trained, we show the algorithm examples of inputs and their
associated desired outputs. One major issue with this concept is `​overfitting`​ — when the
algorithm only learns to fit the training data but doesn’t actually “understand” anything about
underlying input-output dependencies. The network basically just “memorizes” the training data.
Thus, we tend to use “in-sample” data to train a model and then use “out-of-sample” data to
validate an algorithm (or a neural network model in our case). Certain percentages are set aside
for both datasets to partition the data. For example, if there is a dataset of 100,000 samples of data
and labels, you will immediately take 10,000 and set them aside to be your “out-of-sample” or
“validation” data. You will then train your model with the other 90,000 in-sample or “training”
data and finally validate your model with the 10,000 out-of-sample data that the model hasn’t yet
seen. The goal is for the model to not only accurately predict on the training data, but also to be
similarly accurate while predicting on the withheld out-of-sample validation data.
This is called `​generalization`​, which means learning to fit the data instead of memorizing it. The
idea is that we “train” (slowly adjusting weights and biases) a neural network on many examples
of data. We then take out-of-sample data that the neural network has never been presented with
and hope it can accurately predict on these data too.
You should now have a general understanding of what neural networks are, or at least what the
objective is, and how we plan to meet this objective. To train these neural networks, we calculate
how “wrong” they are using algorithms to calculate the error (called ​loss​), and attempt to slowly
adjust their parameters (weights and biases) so that, over many iterations, the network gradually
becomes less wrong. The goal of all neural networks is to generalize, meaning the network can
see many examples of never-before-seen data, and accurately output the values we hope to
achieve. Neural networks can be used for more than just classification. They can perform
regression (predict a scalar, singular, value), clustering (assign unstructured data into groups), and
many other tasks. Classification is just a common task for neural networks.

## Single Neuron

## A Layer of Neurons

## Tensors, Matrix and Vectors

A matrix is pretty simple. It’s a rectangular array. It has columns and rows. It is two dimensional.
So a matrix can be an array (a 2D array).

“What is a tensor, to a computer scientist, in
the context of deep learning?” We believe that we can solve the debate in one line:
A tensor object is an object that can be represented as an array.
What this means is, as programmers, we can (and will) treat tensors as arrays in the context of
deep learning, and that’s really all the thought we have to put into it. Are all tensors ​just​ arrays?
No, but they are represented as arrays in our code, so, to us, they’re only arrays, and this is why
there’s so much argument and confusion.
Now, what is an array? In this book, we define an array as an ordered homologous container for
numbers, and mostly use this term when working with the NumPy package since that’s what the
main data structure is called within it. A linear array, also called a 1-dimensional array, is the
simplest example of an array, and in plain Python, this would be a list. Arrays can also consist
of multi-dimensional data, and one of the best-known examples is what we call a matrix in
mathematics, which we’ll represent as a 2-dimensional array. Each element of the array can be
accessed using a tuple of indices as a key, which means that we can retrieve any array element.
We need to learn one more notion ​—​ a vector. Put simply, a vector in math is what we call a list
in Python or a 1-dimensional array in NumPy. Of course, lists and NumPy arrays do not have
the same properties as a vector, but, just as we can write a matrix as a list of lists in Python, we
can also write a vector as a list or an array! Additionally, we’ll look at the vector algebraically
(mathematically) as a set of numbers in brackets. This is in contrast to the physics perspective,
where the vector’s representation is usually seen as an arrow, characterized by a magnitude and
a direction.

## A layer of Neurons with Numpy

This syntax involving the dot product of weights and inputs followed by the vector addition of
bias is the most commonly used way to represent this calculation of ​inputs·weights+bias​. To
explain the order of parameters we are passing into ​np.dot(),​ we should think of it as whatever
comes first will decide the output shape. In our case, we are passing a list of neuron weights first
and then the inputs, as our goal is to get a list of neuron outputs. As we mentioned, a dot product
of a matrix and a vector results in a list of dot products. The ​np.dot()​ method treats the matrix as
a list of vectors and performs a dot product of each of those vectors with the other vector. In this
example, we used that property to pass a matrix, which was a list of neuron weight vectors and a
vector of inputs and get a list of dot products ​—​ neuron outputs.

## Matrix Product

To perform a matrix product, the size of the second dimension of the left matrix must match the
size of the first dimension of the right matrix. For example, if the left matrix has a shape of ​(5, 4)
then the right matrix must match this 4 within the first shape value ​(4, 7)​. The shape of the
resulting array is always the first dimension of the left array and the second dimension of the right
array, ​(5, 7).​ In the above example, the left matrix has a shape of ​(5, 4),​ and the upper-right matrix
has a shape of ​(4, 5)​. The second dimension of the left array and the first dimension of the second
array are both ​4,​ they match, and the resulting array has a shape of ​(5, 5).​

## Transposition for the Matrix Product

The dot product and matrix product are both implemented in a single method: ​np.dot().​

## A Layer of Neurons & Batch of Data w/ NumPy

We need to perform dot products on all of the vectors that consist of both input and weight matrices.
We just need to perform transposition on its second argument, which is the weights matrix in our case, to turn the row
vectors it currently consists of into column vectors.

We mentioned that the second argument for ​np.dot()​ is going to be our transposed weights, so
first will be inputs, but previously weights were the first parameter. We changed that here.
Before, we were modeling neuron output using a single sample of data, a vector, but now we are
a step forward when we model layer behavior on a batch of data. We could retain the current
parameter order, but, as we’ll soon learn, it’s more useful to have a result consisting of a list of
layer outputs per each sample than a list of neurons and their outputs sample-wise. We want the
resulting array to be sample-related and not neuron-related as we’ll pass those samples further
through the network, and the next layer will expect a batch of inputs.

## Y Target

One-hot encoded : [0, 1] (show the position of the right target)
Sparse encoded : 0 or 1 (binary)



End to page 135