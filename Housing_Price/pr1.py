import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv('Housing_Price\\AMES_Final_DF.csv')

X = df.drop('SalePrice',axis=1)
y = df['SalePrice']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=101)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = ElasticNet()
param_grid = {'alpha':[.1,1,10,100],'l1_ratio':[.1,.7,.9,.95,.99,1]}

grid_model = GridSearchCV(estimator=model,param_grid=param_grid,scoring='neg_mean_squared_error',cv=10,verbose=1)
grid_model.fit(X_train,y_train)
print("Best combination is ",grid_model.best_params_)

final_model = ElasticNet(alpha=100,l1_ratio=1)
final_model.fit(X_train,y_train)
y_pred = final_model.predict(X_test)
MAE = mean_absolute_error(y_pred,y_test)
RMSE = np.sqrt(mean_squared_error(y_pred,y_test))
print("MAE is ",MAE,"RMSE is ",RMSE)