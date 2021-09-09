import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import neighbors
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv('Social_Network_Ads.csv')
matrix_of_features=dataset.iloc[:,:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values


matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.25,random_state=0)

sc=StandardScaler()
matrix_of_features_train=sc.fit_transform(matrix_of_features_train)
matrix_of_features_test=sc.fit_transform(matrix_of_features_test)


classifier=RandomForestClassifier(random_state=0,criterion='entropy',n_estimators=10)
classifier.fit(matrix_of_features_train,dependent_variable_vector_train)
dependent_variable_vector_predict=classifier.predict(matrix_of_features_test)
print(np.concatenate((dependent_variable_vector_predict.reshape(len(dependent_variable_vector_predict),1),dependent_variable_vector_test.reshape(len(dependent_variable_vector_test),1)),1))

acc=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict))