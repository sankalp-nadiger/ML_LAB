import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris

X = load_iris().data
y = load_iris().target
print("Shape of Data:", X.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Original")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Shape of PCA transformed Data:", X_pca.shape)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="jet")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA")
plt.show()

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
print("Shape of LDA transformed Data:", X_lda.shape)

plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="jet")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA")
plt.show()
