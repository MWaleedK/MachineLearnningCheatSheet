import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
import scipy.cluster.hierarchy as sch


dataset=pd.read_csv('Mall_Customers.csv')
matrix_of_features=dataset.iloc[:,[3,4]].values
'''dendrogram=sch.dendrogram(sch.linkage(matrix_of_features,method='ward'))
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
'''
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
dependent_variable_vector_hc=hc.fit_predict(matrix_of_features)

plt.scatter(matrix_of_features[dependent_variable_vector_hc==0,0],matrix_of_features[dependent_variable_vector_hc==0,1],s=100,c='red',label='Cluster1')
plt.scatter(matrix_of_features[dependent_variable_vector_hc==1,0],matrix_of_features[dependent_variable_vector_hc==1,1],s=100,c='blue',label='Cluster2')
plt.scatter(matrix_of_features[dependent_variable_vector_hc==2,0],matrix_of_features[dependent_variable_vector_hc==2,1],s=100,c='green',label='Cluster3')
plt.scatter(matrix_of_features[dependent_variable_vector_hc==3,0],matrix_of_features[dependent_variable_vector_hc==3,1],s=100,c='cyan',label='Cluster4')
plt.scatter(matrix_of_features[dependent_variable_vector_hc==4,0],matrix_of_features[dependent_variable_vector_hc==4,1],s=100,c='magenta',label='Cluster5')
 
plt.title("Clusters of customer")
plt.xlabel('Annual Income')
plt.ylabel('Spendng Score (1-100)')
plt.legend()
plt.show()
