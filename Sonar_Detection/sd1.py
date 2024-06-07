import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report

df = pd.read_csv('Sonar_Detection\\sonar.all-data.csv')

df['Label'] = df['Label'].replace(to_replace='R',value=0)
df['Label'] = df["Label"].replace(to_replace='M',value=1)
#print(df.head())

#sns.heatmap(df.corr(numeric_only=True))
#plt.show()
#print(df.corr().abs().sort_values(by='Label').tail(n=6)['Label'])

X = df.drop('Label',axis=1)
y = df['Label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
knn = KNeighborsClassifier()
scaler = StandardScaler()
k_values = list(range(1,30))
param_grid = {'knn__n_neighbors':k_values}
operations = [('scaler',scaler),('knn',knn)]
pipe = Pipeline(operations)
grid_model = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy',verbose=1)
grid_model.fit(X_train,y_train)
#print(grid_model.best_params_)

'''
dict = grid_model.cv_results_
#print(dict['mean_test_score'])
sns.lineplot(y=dict['mean_test_score'],x=k_values)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
'''

y_pred = grid_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print(cm,cr)
