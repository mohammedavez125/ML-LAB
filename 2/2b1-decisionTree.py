# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import display
from sklearn.metrics import confusion_matrix

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Categorical.from_codes(iris.target, iris.target_names)

print(X.head())
print(y)

y = pd.get_dummies(y)
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

display(graphviz.Source(export_graphviz(dt)))

y_pred = dt.predict(X_test)

species = np.array(y_test).argmax(axis=1)
predictions = np.array(y_pred).argmax(axis=1)

print("Confusion Matrix:\n", confusion_matrix(species, predictions))
