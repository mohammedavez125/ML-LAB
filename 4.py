# 4. Demonstrate ensemble techniques like boosting, bagging, random forests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50,random_state=42)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)
print("Bagging accuracy:", accuracy_score(y_test, y_pred_bagging))

adaboost_clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50,random_state=42)
adaboost_clf.fit(X_train, y_train)
y_pred_adaboost = adaboost_clf.predict(X_test)
print("AdaBoost accuracy:", accuracy_score(y_test, y_pred_adaboost))

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_clf.fit(X_train, y_train)
y_pred_random_forest = random_forest_clf.predict(X_test)
print("Random Forest accuracy:", accuracy_score(y_test, y_pred_random_forest))