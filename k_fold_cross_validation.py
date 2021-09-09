import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score



dataset=pd.read_csv('Social_Network_Ads.csv')
matrix_of_features=dataset.iloc[:,:-1].values
dependent_variable_vector=dataset.iloc[:,-1].values


matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.25,random_state=0)

sc=StandardScaler()
matrix_of_features_train=sc.fit_transform(matrix_of_features_train)
matrix_of_features_test=sc.transform(matrix_of_features_test)


classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(matrix_of_features_train,dependent_variable_vector_train)
dependent_variable_vector_predict=classifier.predict(matrix_of_features_test)
print(np.concatenate((dependent_variable_vector_predict.reshape(len(dependent_variable_vector_predict),1),dependent_variable_vector_test.reshape(len(dependent_variable_vector_test),1)),1))

acc=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict))

accuracies=cross_val_score(estimator=classifier,X=matrix_of_features_train,y=dependent_variable_vector_train,cv=10)
print("Accuracies:{:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation:{:.2f}%".format(accuracies.std()*100))

from sklearn.model_selection import GridSearchCV

parameters=[{'C':[0.25,0.5,0.75,1.0],'kernel':['linear']},{'C':[0.25,0.5,0.75,1.0],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search=GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy',cv=10,n_jobs=-1)

grid_search.fit(matrix_of_features_train,dependent_variable_vector_train)
best_accuracy=grid_search.best_score_

best_parameters=grid_search.best_params_

print("Best accuracy:{:.2f}%".format(best_accuracy*100))
print("Best Parameters:",best_parameters)