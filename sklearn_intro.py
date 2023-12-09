from sklearn import datasets
import numpy as np

# Iris dataset
iris = datasets.load_iris()
# Petal length and petal width
X = iris.data[:, [2, 3]]
# Class labels
y = iris.target
print("Class labels:", np.unique(y))

# Split data, preserving proportions of class labels in test and training sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1, stratify=y)

# Standardize features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Fit multiclass perceptron
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, max_iter=50, random_state=1)
ppn.fit(X_train_std, y_train)
# Predict class labels of test data
y_pred = ppn.predict(X_test_std)
# Report metrics for goodness of fit
print("Misclassified examples: %d" % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))
