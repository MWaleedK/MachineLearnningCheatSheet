import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


dataset=pd.read_csv('Mall_Customers.csv')
matrix_of_features=dataset.iloc[:,[3,4]].values
'''
wcss=[]
for i in range(1,11,1):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
    (kmeans.fit(matrix_of_features))
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11,1),wcss)
plt.title("Elow Method")
plt.xlabel('Numer of Clusters')
plt.ylabel('WCSS')
plt.show()
'''
#becasue the above gave 5 as an optimal cluster count
kmeans=KMeans(n_clusters=5, init='k-means++',random_state=42)
dependent_variable_vector_kmeans=kmeans.fit_predict(matrix_of_features)

plt.scatter(matrix_of_features[dependent_variable_vector_kmeans==0,0],matrix_of_features[dependent_variable_vector_kmeans==0,1],s=100,c='red',label='Cluster1')
plt.scatter(matrix_of_features[dependent_variable_vector_kmeans==1,0],matrix_of_features[dependent_variable_vector_kmeans==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(matrix_of_features[dependent_variable_vector_kmeans==2,0],matrix_of_features[dependent_variable_vector_kmeans==2,1],s=100,c='green',label='Cluster3')
plt.scatter(matrix_of_features[dependent_variable_vector_kmeans==3,0],matrix_of_features[dependent_variable_vector_kmeans==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(matrix_of_features[dependent_variable_vector_kmeans==4,0],matrix_of_features[dependent_variable_vector_kmeans==4,1],s=100,c='magenta',label='Cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='yellow',label='centroids')
plt.title("Clusters of customer")
plt.xlabel('Annual Income')
plt.ylabel('Spendng Score (1-100)')
plt.legend()
plt.show()


