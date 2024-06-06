import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Heart_Disease\\heart.csv')

#print(df.head())
#print(df.describe())

#print(df.isnull().sum())
#sns.countplot(data=df,x='target')
'''
vars = ['age','trestbps', 'chol','thalach']
sns.pairplot(data=df,vars=vars,hue='target')
'''
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()