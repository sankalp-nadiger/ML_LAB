import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(X, y, test_size, dataset_name):
    print(f"\n--- {dataset_name}: Train-Test Split {int((1-test_size)*100)}-{int(test_size*100)} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
evaluate_model(X_iris, y_iris, 0.10, "Iris Dataset")
evaluate_model(X_iris, y_iris, 0.30, "Iris Dataset")

data = pd.read_csv("titanic.csv")
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data.dropna(subset=['Age'])
X_titanic = data[['Pclass', 'Sex', 'Age', 'Fare']]
y_titanic = data['Survived']
evaluate_model(X_titanic, y_titanic, 0.10, "Titanic Dataset")
evaluate_model(X_titanic, y_titanic, 0.30, "Titanic Dataset")

def evaluate_model(test_size):
    print(f"\n--- Iris Dataset: Train-Test Split {int((1-test_size)*100)}-{int(test_size*100)} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

evaluate_model(0.10)
evaluate_model(0.30)
