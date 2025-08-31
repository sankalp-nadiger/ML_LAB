import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris

X = load_iris().data
y = load_iris().target
print("Shape of Data:", X.shape)

plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Original")
plt.show()

lda = LDA(n_components=2)
lda.fit(X,y)
X_projected = lda.transform(X)
print("Shape of transformed Data:", X_projected.shape)

ld1 = X_projected[:, 0]
ld2 = X_projected[:, 1]

plt.scatter(ld1, ld2, c=y, cmap="jet")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA")
plt.show()