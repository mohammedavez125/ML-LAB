import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import label_binarize
from sklearn.metrics import  confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binarize the output (required for some metrics like ROC)
y_bin = label_binarize(y, classes=[0, 1, 2])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(X, y, y_bin, test_size=0.3, random_state=42)

# Initialize classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Train classifiers
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)

# Function to calculate TPR and FPR
def calculate_tpr_fpr(conf_matrix):
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

# Evaluate classifiers
results = []
for name, clf in classifiers.items():
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    if y_proba is not None:
        roc_auc = roc_auc_score(y_bin_test, y_proba, multi_class="ovo", average="macro")
    else:
        roc_auc = "N/A"
    conf_matrix = confusion_matrix(y_test, y_pred)
    tpr, fpr = calculate_tpr_fpr(conf_matrix)
    results.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc,
        "TPR": tpr,
        "FPR": fpr
    })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)
