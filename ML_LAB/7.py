import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN

glass_data=pd.read_csv("glass.csv")
fruit_data=pd.read_csv("fruits.csv")

datasets=['Glass','Fruit']
Xg = glass_data.drop('Type',axis=1).values
yg = glass_data['Type'].values
Xf = fruit_data[['mass','width','height','color_score']].values
yf = fruit_data['fruit_label'].values
k_values=[3,5,7]
distance_metrics= ['manhattan','euclidean']
test_splits = [0.1,0.3]
for dataset in datasets:
    print(f"Datasets:{dataset}")
    for k in k_values:
        for p,distance_metric in enumerate(distance_metrics):
            for ts in test_splits:
                print(f"K={k},Distance Metric={distance_metric}: ")
                print(f"Split = {ts}")
                knn=KNN(n_neighbors=k,p=p,weights="distance")
                if dataset == 'Glass':
                    X_train,X_test, y_train,y_test=train_test_split(Xg,yg,test_size=ts,random_state=42)
                else:
                    X_train,X_test, y_train,y_test=train_test_split(Xf,yf,test_size=ts,random_state=42)
                knn.fit(X_train,y_train)
                accuracy=knn.score(X_test,y_test)
                print(f"accuracy on test set: {accuracy:.2f}")
                print("------------")
            print("----------------------------")
    print("------------------------------------------------")