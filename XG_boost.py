from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split





dataset=pd.read_csv('Data_2.csv')
matrix_of_features=dataset.iloc[:,:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values


matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.2,random_state=0)

from xgboost import XGBClassifier

classifier=XGBClassifier()
classifier.fit(matrix_of_features_train,dependent_variable_vector_train)
dependent_variable_vector_predict=classifier.predict(matrix_of_features_test)

acc=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(acc)
print(accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict))
