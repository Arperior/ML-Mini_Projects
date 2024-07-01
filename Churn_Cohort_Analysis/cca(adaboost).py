from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Churn_Cohort_Analysis\\Telco-Customer-Churn.csv')

X = df.drop('Churn',axis=1)
y = df['Churn']
X = X.drop('customerID',axis=1)
X = pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=101)

rfc = AdaBoostClassifier()
param_grid = {'learning_rate':[0.05,0.1,0.2,0.5,1],'n_estimators':[1,10,32,64,100,128]}
grid = GridSearchCV(rfc,param_grid)
grid.fit(X_train,y_train)
preds = grid.predict(X_test)
print(grid.best_params_)
print(classification_report(y_test,preds))
ConfusionMatrixDisplay.from_predictions(y_test,preds)
plt.show()