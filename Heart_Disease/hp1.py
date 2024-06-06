import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
from joblib import dump

df = pd.read_csv('Heart_Disease\\heart.csv')

X = df.drop('target',axis=1)
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=10000,tol=0.0001,solver='saga',multi_class='ovr')
C = np.logspace(0,10,10)
penalty = ['elasticnet','l1','l2']
l1_ratio = np.linspace(0,1,10)
param_grid = {'l1_ratio':l1_ratio,'penalty':penalty,'C':C}
#grid_model = GridSearchCV(param_grid=param_grid,verbose=1,estimator=log_model)
#grid_model.fit(scaled_X_train,y_train)
#print(grid_model.best_estimator_)

final_model = LogisticRegression(max_iter=10000,tol=0.0001,solver='saga',multi_class='ovr',C=1.0, penalty= 'l1')
final_model.fit(scaled_X_train,y_train)
#print(final_model.coef_)

coe = pd.DataFrame(final_model.coef_)
coe = coe.transpose()
column = X.columns
coe['Column'] = column
coe.columns = ['Coef','Attribute']
coe = coe.sort_values(by='Coef')
#print(coe)
#sns.barplot(y='Coef',x='Attribute',data=coe)
#plt.show()

y_pred = final_model.predict(scaled_X_test)
cm = confusion_matrix(y_test,y_pred)
report = classification_report(y_test,y_pred)
#print(cm,'Report :\n',report)

# Performance Curves

#ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

#PrecisionRecallDisplay.from_estimator(final_model,scaled_X_test,y_test)
#plt.ylim(0.7,1.02)

#RocCurveDisplay.from_estimator(final_model,scaled_X_test,y_test)

#plt.show()

dump(final_model,'Heart_Disease\\heartlog.joblib')


