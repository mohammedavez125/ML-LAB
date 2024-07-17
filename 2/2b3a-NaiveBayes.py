import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

iris=datasets.load_iris()
X=iris.data
y=iris.target

print("Features :",iris['feature_names'])

iris_dataframe=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])

# plt.plot(X,y)
grr=pd.plotting.scatter_matrix(iris_dataframe,c=iris['target'],figsize=(15,5),s=60,alpha=0.8)
# plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

NB=GaussianNB()
NB.fit(X_train,y_train)

Y_predict=NB.predict(X_test)

cm=confusion_matrix(y_test,Y_predict)

df_cm=pd.DataFrame(cm,columns=np.unique(y_test),index=np.unique(y_test))
df_cm.index.name=('Actual')
df_cm.columns.name=('predicted')

plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.show()