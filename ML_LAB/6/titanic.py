import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("titanic.csv")
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data.dropna(subset=['Age'])

X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

def evaluate_model(test_size):
    print(f"\n--- Titanic Dataset: Train-Test Split {int((1-test_size)*100)}-{int(test_size*100)} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_model(0.10)
evaluate_model(0.30)
