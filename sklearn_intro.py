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

# Fit logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=10.0, random_state=1,
                        solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_std, y_train)
# Predict class labels of test data
y_pred_lr = lr.predict(X_test_std)

import matplotlib.pyplot as plt
plt.scatter(X_test[y_pred_lr == 0, 0], X_test[y_pred_lr == 0, 1],
            color='red', marker='o', label="0")
plt.scatter(X_test[y_pred_lr == 1, 0], X_test[y_pred_lr == 1, 1],
            color='green', marker='o', label="1")
plt.scatter(X_test[y_pred_lr == 2, 0], X_test[y_pred_lr == 2, 1],
            color='blue', marker='o', label="2")
plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend(loc='upper left')
plt.title("Predicted class by petal length, width")
plt.show()