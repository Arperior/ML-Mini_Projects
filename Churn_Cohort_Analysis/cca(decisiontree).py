import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('Churn_Cohort_Analysis\\Telco-Customer-Churn.csv')

X = df.drop('Churn',axis=1)
y = df['Churn']
X = X.drop('customerID',axis=1)
X = pd.get_dummies(X,drop_first=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=101)

'''
dtc = DecisionTreeClassifier()
param_grid = {'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2']}
grid = GridSearchCV(dtc,param_grid)
grid.fit(X_train,y_train)
preds = grid.predict(X_test)
print(grid.best_params_)
print(classification_report(y_test,preds))
ConfusionMatrixDisplay.from_predictions(y_test,preds)
plt.show()
'''

model = DecisionTreeClassifier(criterion='gini',max_features='sqrt')
model.fit(X_train,y_train)
feat_imp = model.feature_importances_
imp = pd.DataFrame(feat_imp)
imp.columns = ['Feature Importances']
imp.index = X.columns
imp = imp.sort_values(by='Feature Importances')
imp = imp[imp['Feature Importances'] > 0.0005]
sns.barplot(data=imp,x=imp.index,y='Feature Importances')
plt.xticks(rotation=90)
plt.show()
