import matplotlib.pyplot as plt
from nltk import stem
import numpy as np
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB


dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#replace everthing that's not a letter by space
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)

cv=CountVectorizer(max_features=1500)
matrix_of_features=cv.fit_transform(corpus).toarray()
dependent_variable_vector=dataset.iloc[:,-1].values

#print(len(matrix_of_features[0]))
matrix_of_features_train,matrix_of_features_test,dependent_variable_vector_train,dependent_variable_vector_test=train_test_split(matrix_of_features,dependent_variable_vector,test_size=0.25,random_state=0)

classifier=GaussianNB()
classifier.fit(matrix_of_features_train,dependent_variable_vector_train)

dependent_variable_vector_predict=classifier.predict(matrix_of_features_test)

print(np.concatenate((dependent_variable_vector_predict.reshape(len(dependent_variable_vector_predict),1),dependent_variable_vector_test.reshape(len(dependent_variable_vector_test),1)),1))

cm=confusion_matrix(dependent_variable_vector_test,dependent_variable_vector_predict)
print(accuracy_score(dependent_variable_vector_test,dependent_variable_vector_predict))



