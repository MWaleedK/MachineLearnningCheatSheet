import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#import dataset
dataset=pd.read_csv('Data.csv')
#create a matrix of features and the dependent variable vector
matrix_of_feature=dataset.iloc[:,:-1].values
dependednt_variable_vector=dataset.iloc[:,-1].values

#handling missing data
# replace NAN values with mean 

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(matrix_of_feature[:,1:])#take in data on which work has to be done
matrix_of_feature[:,1:]=imputer.transform(matrix_of_feature[:,1:])#apply work on data, transform your matrix

#onehot encoding
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
#this method caters to fitting and transforming in one go
matrix_of_feature=np.array(ct.fit_transform(matrix_of_feature))

le=LabelEncoder()
dependednt_variable_vector=le.fit_transform(dependednt_variable_vector)

#print your variables
#print(matrix_of_feature)
#print(dependednt_variable_vector)

#deparating the training set and the test set
matrix_of_feature_train,matrix_of_feature_test,dependednt_variable_vector_train,dependednt_variable_vector_test=train_test_split(matrix_of_feature,dependednt_variable_vector,test_size=0.2,random_state=1)

#feature scaling
sc=StandardScaler()
matrix_of_feature_train[:,-2:]=sc.fit_transform(matrix_of_feature_train[:,-2:])
matrix_of_feature_test[:,-2:]=sc.transform(matrix_of_feature_test[:,-2:])#skip fit, only apply same multiplicative scalar to the test sample

print(matrix_of_feature_train)
print(matrix_of_feature_test)
#print(dependednt_variable_vector_train)
#print(dependednt_variable_vector_test)
