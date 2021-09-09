import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset=pd.read_csv('Churn_Modelling.csv')
matrix_of_features=dataset.iloc[:,3:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values

Le=LabelEncoder()
matrix_of_features[:,2]=Le.fit_transform(matrix_of_features[:,2])

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
matrix_of_features=np.array(ct.fit_transform(matrix_of_features))

matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.25,random_state=0)

sc=StandardScaler()
matrix_of_features_train=sc.fit_transform(matrix_of_features_train)
matrix_of_features_test=sc.fit_transform(matrix_of_features_test)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=6,activation='softmax'))

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

ann.fit(matrix_of_features_train,dependent_variable_vector_train,batch_size=32,epochs=100)

print(ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])))

dependent_variable_vector_predict=ann.predict(matrix_of_features_test)
dependent_variable_vector_predict=(dependent_variable_vector_predict>0.5)
print(np.concatenate((dependent_variable_vector_predict.reshape(len(dependent_variable_vector_predict),1),dependent_variable_vector_test.reshape(len(dependent_variable_vector_test),1)),1))


acc=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict))