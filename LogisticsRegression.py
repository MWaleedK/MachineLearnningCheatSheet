from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score


dataset=pd.read_csv('Social_Network_Ads.csv')
matrix_of_features=dataset.iloc[:,:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values


matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.25,random_state=0)

sc=StandardScaler()
matrix_of_features_train=sc.fit_transform(matrix_of_features_train)
matrix_of_features_test=sc.fit_transform(matrix_of_features_test)

lr=LogisticRegression(random_state=0)
lr.fit(matrix_of_features_train,dependent_variable_vector_train)

dependent_variable_vector_predict=lr.predict(matrix_of_features_test)
print(np.concatenate((dependent_variable_vector_predict.reshape(len(dependent_variable_vector_predict),1),dependent_variable_vector_test.reshape(len(dependent_variable_vector_test),1)),1))

acc=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict))