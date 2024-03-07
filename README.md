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
Backpropagation
Gradient descent
Standard Error
Softmax function


# Bonus

- A more complex optimization function (for example : nesterov momentum, RMSprop, Adam, ...).
- A display of multiple learning curves on the same graph (really useful to compare different models).
- An historic of the metric obtained during training.
- The implementation of early stopping.
- Evaluate the learning phase with multiple metrics.