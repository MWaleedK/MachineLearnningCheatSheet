from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


dataset=pd.read_csv('Position_Salaries.csv')
matrix_of_features=dataset.iloc[:,1:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values

dependent_variable_vector=np.array(dependent_variable_vector).reshape(len(dependent_variable_vector),1)
#feature scaling
sc_x=StandardScaler()
sc_y=StandardScaler()
#two scaling variables used
matrix_of_features=sc_x.fit_transform(matrix_of_features)
dependent_variable_vector=sc_y.fit_transform(dependent_variable_vector)

#training SVR model
from sklearn.svm import SVR

regressor=SVR(kernel='rbf')
regressor.fit(matrix_of_features,dependent_variable_vector)
#must transform with the scaling variables
predictedMat=sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print(predictedMat)

plt.scatter(sc_x.inverse_transform(matrix_of_features),sc_y.inverse_transform(dependent_variable_vector),color='red')
plt.plot(sc_x.inverse_transform(matrix_of_features),sc_y.inverse_transform(regressor.predict(matrix_of_features)),color='blue')
plt.title('Truth or bluff (SVR)')
plt.xlabel('PositionLabel')
plt.ylabel('Salary')
plt.show()




