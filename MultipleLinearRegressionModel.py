import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv('50_Startups.csv')
matrix_of_features=dataset.iloc[:,:-1].values
dependednt_variable_vector=dataset.iloc[:,-1].values


#apply OneHot encoding on index 3
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
matrix_of_features=np.array(ct.fit_transform(matrix_of_features))


matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependednt_variable_vector,test_size=0.2,random_state=0)


#multiple linear regression
regressor=LinearRegression()
regressor.fit(matrix_of_features_train,dependent_variable_vector_train)

dependent_variable_vector_predicted=regressor.predict(matrix_of_features_test)
np.set_printoptions(precision=2)
print(np.concatenate((dependent_variable_vector_predicted.reshape(len(dependent_variable_vector_predicted),1),dependent_variable_vector_test.reshape(len(dependent_variable_vector_test),1)),1))