from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

texts = ["I love programming.", "Python is awesome!", "I hate bugs.", "Debugging is fun.", "I enjoy learning new things."]
labels = [1, 1, 0, 1, 1] 

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

clf = SVC()
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))
