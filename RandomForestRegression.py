from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

dataset=pd.read_csv('Position_Salaries.csv')
matrix_of_features=dataset.iloc[:,1:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values

regressor=RandomForestRegressor( n_estimators=10, random_state=0)
regressor.fit(matrix_of_features,dependent_variable_vector)
reg_predicted=regressor.predict([[6.5]])

print(reg_predicted)