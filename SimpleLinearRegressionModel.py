from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv('Salary_Data.csv')
matrix_of_features=dataset.iloc[:,:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values

matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.2,random_state=0)

#continuous  
regressor=LinearRegression()
regressor.fit(matrix_of_features_train,dependent_variable_vector_train)#applying the simple linear regression model on the training set
dependent_variable_vector_predicted=regressor.predict(matrix_of_features_test)
plt.figure(1)
plt.scatter(matrix_of_features_train,dependent_variable_vector_train,color='red')
plt.plot(matrix_of_features_train,regressor.predict(matrix_of_features_train),color='blue')
plt.title('Salary Vs Experience Training Set')
plt.xlabel('Salary')
plt.ylabel('Experience')
plt.show()

plt.figure(2)
plt.scatter(matrix_of_features_test,dependent_variable_vector_test,color='red')
plt.plot(matrix_of_features_train,regressor.predict(matrix_of_features_train),color='blue')
plt.title('Salary Vs Experience Test Set')
plt.xlabel('Salary')
plt.ylabel('Experience')
plt.show()