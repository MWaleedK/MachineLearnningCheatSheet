import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA as KPCA

database=pd.read_csv('Wine.csv')
matrix_of_features=database.iloc[:,:-1].values
dependent_variable_vector=database.iloc[:,-1].values

matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.2,random_state=0)

sc=StandardScaler()
matrix_of_features_train=sc.fit_transform(matrix_of_features_train)
matrix_of_features_test=sc.transform(matrix_of_features_test)

kpca=KPCA(n_components=2,kernel='rbf')
matrix_of_features_train=kpca.fit_transform(matrix_of_features_train)
matrix_of_features_test=kpca.transform(matrix_of_features_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(matrix_of_features_train,dependent_variable_vector_train)

dependent_variable_vector_predict=classifier.predict(matrix_of_features_test)
cm=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(cm)
v=accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict)
print(v)
