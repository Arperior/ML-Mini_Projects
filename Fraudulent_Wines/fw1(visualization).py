import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Fraudulent_Wines\\wine_fraud.csv')

#print(df.head())
#print(df['quality'].unique())

#sns.countplot(x='quality',data=df,hue='quality')
#sns.countplot(x='type',data=df,hue='quality')
#plt.show()

'''
red_f = df[(df['quality'] == 'Fraud') & (df['type'] == 'red')].shape[0]
red_l = df[(df['quality'] == 'Legit') & (df['type'] == 'red')].shape[0]
white_f = df[(df['quality'] == 'Fraud') & (df['type'] == 'white')].shape[0]
white_l = df[(df['quality'] == 'Legit') & (df['type'] == 'white')].shape[0]
print('Percentage of white wine fraud is',100*white_f/(white_f+white_l),'\nPercentage of red wine fraud is',100*red_f/(red_f+red_l))
'''

df1 = df.drop('type',axis=1)
df1['quality'] = df1['quality'].map({'Fraud':1,'Legit':0})
#print(df1.corr().sort_values(by=['quality']).iloc[:, -1])
df2 = df1.corr().sort_values(by=['quality']).iloc[:, -1]
df2 = df2.drop('quality',axis=0)
#sns.barplot(data=df2)
sns.clustermap(df1.corr())
plt.show()