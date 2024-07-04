import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

df = pd.read_csv('Country Clustering\\CIA_Country_Facts.csv')
iso = pd.read_csv('Country Clustering\\country_iso_codes.csv')

#print(df.isnull().sum())

#print(df[df['Agriculture'].isnull()]['Country'])

df['Agriculture'] = df['Agriculture'].fillna(0)
#print(df.isnull().sum())
df['Climate'] = df['Climate'].fillna(df.groupby('Region')['Climate'].transform('mean'))
#print(df.isnull().sum())
df['Literacy (%)'] = df['Literacy (%)'].fillna(df.groupby('Region')['Literacy (%)'].transform('mean'))
df['Industry'] = df['Industry'].fillna(df.groupby('Region')['Industry'].transform('mean'))
df['Service'] = df['Service'].fillna(df.groupby('Region')['Service'].transform('mean'))
#print(df.isnull().sum())
df = df.dropna()

X = pd.get_dummies(df.drop('Country',axis=1))
#print(X.columns)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

ssd = []
for k in range(2,30):
    model = KMeans(n_clusters=k)
    model.fit(scaled_X)
    ssd.append(model.inertia_)

#plt.plot(range(2,30),ssd,'o--')
diff_s = pd.Series(ssd).diff()
#sns.barplot(x=range(2,30),y=diff_s)
#plt.show()

model_final = KMeans(n_clusters=3)
labels = model_final.fit_predict(scaled_X)
X['Cluster'] = labels
#print(X.corr()['Cluster'].iloc[:-1].sort_values())
X['Country'] = df['Country']
df_merged = pd.merge(X,iso,on='Country',how='left')
#print(df_merged)

fig = px.choropleth(df_merged, locations="ISO Code",
                    color="Cluster", 
                    hover_name="Country", 
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.write_html('map.html')