import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

glass_data=pd.read_csv("glass.csv")

Xg = glass_data.drop('Type',axis=1).values
yg = glass_data['Type'].values

distance_metrics= ['manhattan','euclidean']

for p,distance_metric in enumerate(distance_metrics):
            print(f"Distance Metric={distance_metric}: ")
            knn=KNN(n_neighbors=3,p=p+1,weights="distance")
            X_train,X_test, y_train,y_test=train_test_split(Xg,yg,test_size=0.3,random_state=42)
            knn.fit(X_train,y_train)
            accuracy=knn.score(X_test,y_test)
            print(f"accuracy on test set: {accuracy:.2f}")
            print("------------")




