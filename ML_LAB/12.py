import numpy as np

def unit_step(x):
    return np.where(x > 0, 1, 0)

class SLP:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        self.weights = np.zeros((X.shape[1], 1))
        for _ in range(self.epochs):
            for ip, label in zip(X, y):
                ip = ip.reshape(-1, 1)
                linear_op = np.dot(ip.T, self.weights) + self.bias
                prediction = unit_step(linear_op)
                err = label - prediction
                self.weights += self.lr * err * ip
                self.bias += self.lr * err

    def predict(self, X):
        X = X.reshape(-1, 1)
        linear_op = np.dot(X.T, self.weights) + self.bias
        prediction = unit_step(linear_op)
        return prediction.item()

X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([0, 0, 0, 1])

X_or = np.array([[0,0], [0,1], [1,0], [1,1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = SLP()
perceptron_and.fit(X_and, y_and)
print("AND Gate:")
print(perceptron_and.predict(np.array([0,0])))
print(perceptron_and.predict(np.array([0,1])))
print(perceptron_and.predict(np.array([1,0])))
print(perceptron_and.predict(np.array([1,1])))

perceptron_or = SLP()
perceptron_or.fit(X_or, y_or)
print("\nOR Gate:")
print(perceptron_or.predict(np.array([0,0])))
print(perceptron_or.predict(np.array([0,1])))
print(perceptron_or.predict(np.array([1,0])))
print(perceptron_or.predict(np.array([1,1])))
