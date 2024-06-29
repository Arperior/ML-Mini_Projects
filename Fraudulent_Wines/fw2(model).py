from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature
from sklearn.svm import SVC

df = pd.read_csv('Fraudulent_Wines\\wine_fraud.csv')

df1 = pd.get_dummies(df,columns=['type'])
#print(df1.head())

y = df1['quality']
X = df1.drop('quality',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

svm = SVC(class_weight='balanced')
param_grid = {'C':[0.001, 0.01, 0.1, 0.5, 1],'gamma':['scale','auto']}
grid = GridSearchCV(svm,param_grid)
grid.fit(scaled_X_train,y_train)
y_pred = grid.predict(scaled_X_test)


cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print('Confusion Matrix is ',cm,'\nClassification Report is',cr)