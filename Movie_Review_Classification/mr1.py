import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,ConfusionMatrixDisplay

df = pd.read_csv('Movie_Review_Classification\\moviereviews.csv')

#print(df.isnull().sum())

df = df.dropna()
df = df[~df['review'].str.isspace()]
#df = df.drop(df[df['review'].str.isspace() == True].index)
#print(df[df['review'].str.isspace() == True])
#print(df.count())
#print(df['label'].value_counts())
X = df['review']
y = df['label']

'''
cv = CountVectorizer(stop_words='english')
sparse_matrix = cv.fit_transform(df[df['label']=='neg']['review'])
freq = zip(cv.get_feature_names_out(),sparse_matrix.sum(axis=0).tolist()[0])
print(sorted(freq,key=lambda x: -x[1])[:20])
'''

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
pipe = Pipeline([('tfidf', TfidfVectorizer()),('svc', LinearSVC()),])
pipe.fit(X_train,y_train)
preds = pipe.predict(X_test)
print(classification_report(y_test,preds))
ConfusionMatrixDisplay.from_predictions(y_test,preds)
plt.show()
