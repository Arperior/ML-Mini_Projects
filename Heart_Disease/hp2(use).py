from joblib import load
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Heart_Disease\\heart.csv')
X = df.drop('target',axis=1)
y = df['target']

model = load('Heart_Disease\\heartlog.joblib')
model.fit(X,y)

patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]

pred = model.predict(patient)
print(pred,'\n',model.predict_proba(patient))