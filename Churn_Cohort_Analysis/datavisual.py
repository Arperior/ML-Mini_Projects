import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Cohort_Analysis\\Telco-Customer-Churn.csv')

#print(df.isnull().sum())
#print(df.describe())

#sns.countplot(data=df,x='Churn')
#sns.violinplot(data=df,x='Churn',y='TotalCharges')
#sns.boxplot(data=df,x='Contract',y='TotalCharges',hue='Churn')

'''
corr = df.drop('customerID',axis=1)
df1 = pd.get_dummies(corr,drop_first=True)
corr = df1.corr().sort_values(by=['Churn_Yes']).iloc[:, -1]
sns.barplot(data=corr)
plt.xticks(rotation=90)
plt.show()
'''

#sns.histplot(data=df,x='tenure',bins=69)
#sns.displot(data=df,x='tenure',col='Contract',row='Churn',kind='hist')
#sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn')
#plt.show()

rate = []
df2 = pd.get_dummies(df)
for n in range(1,73):
    yes = df2[(df2['tenure'] == n) & df2['Churn_Yes'] == 1]
    yesm = yes.count()
    no = df2[(df['tenure'] == n) & df2['Churn_No'] == 1]
    nom = no.count()
    r = 100*yesm/(yesm+nom)
    rate.append(r)
df3 = pd.DataFrame(rate)
#print(df3['tenure'])
#plt.plot(range(1,73),df3['tenure'])
#plt.show()

def tenure_cohort(x):
    if x < 13:
        x = '0-12 Months'
    elif x < 25:
        x = '12-24 Months'
    elif x < 49:
        x = '24-48 Months'
    else:
        x = 'Over 48 Months'
    return x
df['Tenure Cohort'] = df['tenure'].apply(tenure_cohort)
#print(df.head())
#sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Tenure Cohort')
#sns.countplot(data=df,x='Tenure Cohort',hue='Churn')
sns.catplot(kind='count',data=df,x='Tenure Cohort',hue='Churn',col='Contract')
plt.show()