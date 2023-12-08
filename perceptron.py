# Implementing a perceptron

import numpy as np

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0).
    n_iter : int
        Passes over the training set.
    random_state : int
        Seed for random weight initialization

    Attributes
    ------------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    Methods
    ------------
    fit(X, y):
        Fit training data.
    net_input(X):
        Calculate net input.
    predict(X):
        Return predicted class labels.
    """

    def __init__(self, eta=0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Fit training data.

        Parameters
        -------------
        X : array-like
            Training matrix of shape [observations, features].
        y : 1d-array
            Class labels of training data.
        
        Returns
        -------------
        self : Perceptron
            Trained perceptron with w_ and errors_.
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1] + 1)

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input.

        Parameters
        -------------
        X : array-like
            Matrix of shape [observations, features].

        Returns
        -------------
        : 1d-array
            Net inputs for each observation in X.

        """

        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return predicted class labels.

        Parameters
        -------------
        X : array-like
            Matrix of shape [observations, features].
        
        Returns
        -------------
        : 1d-array
            Class labels for each observation in X.
        """

        return np.where(self.net_input(X) >= 0, 1, 0)
    
import os
import pandas as pd

# Iris dataset
s = os.path.join('https://archive.ics.uci.edu', 'ml',
              'machine-learning-databases', 'iris', 'iris.data')

df = pd.read_csv(s, header = None, encoding = 'utf-8')
print(df.tail())

import matplotlib.pyplot as plt

# Selecting setosa and versicolor flowers
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 1, 0)

# Extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# Plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Fit perceptron to data
ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X, y)

# Plot errors per iteration
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = "o")
plt.xlabel = "Epochs"
plt.ylabel = "Number of updates"
plt.show()