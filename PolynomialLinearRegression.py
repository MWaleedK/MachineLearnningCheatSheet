import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


database=pd.read_csv('Position_Salaries.csv')
matrix_of_features=database.iloc[:,1:-1].values
dependent_variable_vector=database.iloc[:,-1].values

#linear regression Model
linear_regressor=LinearRegression()
linear_regressor.fit(matrix_of_features,dependent_variable_vector)
Linear_dependent_variable_vector_predicted=linear_regressor.predict(matrix_of_features)

#polynomial regression model
poly_regressor=PolynomialFeatures(degree=4)
matrix_of_features_poly=poly_regressor.fit_transform(matrix_of_features)
lin_reg_2=LinearRegression()
lin_reg_2.fit(matrix_of_features_poly,dependent_variable_vector)
Linear_dependent_variable_vector_predicted_2=lin_reg_2.predict(matrix_of_features_poly)

print(linear_regressor.predict([[6.5]]))
print(lin_reg_2.predict(poly_regressor.fit_transform([[6.5]])))


plt.scatter(matrix_of_features,dependent_variable_vector,color='red')
plt.plot(matrix_of_features,Linear_dependent_variable_vector_predicted,color='blue')
plt.title('Truth or bluff (LinearModel)')
plt.xlabel('PositionLabel')
plt.ylabel('Salary')
plt.show()

plt.scatter(matrix_of_features,dependent_variable_vector,color='red')
plt.plot(matrix_of_features,Linear_dependent_variable_vector_predicted_2,color='blue')
plt.title('Truth or bluff (PolyModel)')
plt.xlabel('PositionLabel')
plt.ylabel('Salary')
plt.show()
