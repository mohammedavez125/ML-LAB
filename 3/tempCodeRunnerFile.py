
plt.figure(figsize=(10, 6))
g = sns.lineplot(x=range(1, 11), y=sse, marker='o')
g.set(xlabel="Number of clusters (k)", ylabel="Sum Squared Error", title='Elbow Method')
plt.show()
