import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

##Read in dataset
df = pd.read_csv('news.csv')
print(df.head())

labels = df.label

x_train,x_test,y_train,y_test = train_test_split(df['text'],labels,test_size=.2,random_state=7)

##Create vectorize and preprocess to get rid of stop words
tfidf_vect = TfidfVectorizer(stop_words='english',max_df=.7)

tfidf_train = tfidf_vect.fit_transform(x_train)
tfidf_test= tfidf_vect.transform(x_test)

##Create classifier to predict for fake news
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

print(confusion_matrix(y_test,y_pred,labels=['FAKE','REAL']))
