import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
data = pd.read_csv("ToyotaCorolla.csv")
X = data['KM']
Y = data['Doors']
Z = data['Price']

# Scatter plot
plt.scatter(X, Y)
plt.title("Scatter Plot")
plt.show()

# Box plot
sns.boxplot(data=data)
plt.title("Box Plot")
plt.show()

# Heatmap (correlation)
sns.heatmap(data.corr(), annot=True)
plt.title("Heatmap")
plt.show()

# Contour plot
plt.tricontour(X, Y, Z, cmap='jet')
plt.title("Contour Plot")
plt.show()


#3d surface plot
ax = plt.axes(projection='3d')
ax.plot_trisurf(X, Y, Z, cmap='jet')
ax.set_title("3D Surface Plot")
plt.show()

