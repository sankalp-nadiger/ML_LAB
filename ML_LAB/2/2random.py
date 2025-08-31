import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random data
X = np.random.rand(50) * 10   # 50 random numbers between 0–10
Y = np.random.rand(50) * 10
Z = np.random.rand(50) * 10

# Scatter plot
plt.scatter(X, Y)
plt.title("Scatter Plot")
plt.show()

# Box plot
sns.boxplot(data=np.column_stack([X, Y, Z]))
plt.title("Box Plot")
plt.show()

# Heatmap (correlation)
data = np.column_stack([X, Y, Z])
sns.heatmap(np.corrcoef(data, rowvar=False), annot=True)
plt.title("Heatmap")
plt.show()

# Contour plot
plt.tricontour(X, Y, Z, cmap='jet')
plt.title("Contour Plot")
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(X, Y, Z, cmap='jet')
ax.set_title("3D Surface Plot")
plt.show()
