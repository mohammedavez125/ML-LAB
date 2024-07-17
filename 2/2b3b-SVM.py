import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

iris=datasets.load_iris()
X=iris.data[:,]
y=iris.target

print("Features :",iris['feature_names'])
iris_dataframe=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])

plt.plot(X,y)
grr=pd.plotting.scatter_matrix(iris_dataframe,c=iris['target'],figsize=(15,5),s=60,alpha=0.8)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n",cm)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))