import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X, y = load_iris(return_X_y=True)

sse = []  
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(X)
    sse.append(km.inertia_)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
g = sns.lineplot(x=range(1, 11), y=sse, marker='o')
g.set(xlabel="Number of clusters (k)", ylabel="Sum Squared Error", title='Elbow Method')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(X)

print("Cluster Centers:\n", kmeans.cluster_centers_)

pred = kmeans.fit_predict(X)
print("Cluster Predictions:\n", pred)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=cm.Accent)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c='red')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=pred, cmap=cm.Accent)
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], marker='^', c='red')
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width")

plt.show()