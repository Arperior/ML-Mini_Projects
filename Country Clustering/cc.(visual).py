import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Country Clustering\\CIA_Country_Facts.csv')

#print(df.head())
#print(df.info())
#print(df.describe())

#sns.histplot(data=df,x='Population')
#plt.xlim((0,0.5*pow(10,9)))

#sns.barplot(data=df,x='Region',y='GDP ($ per capita)')
#plt.xticks(rotation=90)

#sns.scatterplot(data=df,y='Phones (per 1000)',x='GDP ($ per capita)',hue='Region')
#plt.legend(bbox_to_anchor=(1,0.75))

#sns.scatterplot(data=df,y='Literacy (%)',x='GDP ($ per capita)',hue='Region')

#sns.heatmap(df.corr(numeric_only=True))

sns.clustermap(df.corr(numeric_only=True))
plt.show()