# -*- coding: utf-8 -*-

!pip install --upgrade --ignore-installed kaggle

from google.colab import files
files.upload()

!mkdir ~/.kaggle #created at root folder in colab

#copy
!cp kaggle.json ~/.kaggle

! chmod 600 ~/.kaggle/kaggle.json

countries=!kaggle datasets download -d thomaskranzkowski/using-geodata-in-python

# Sorting countries by population

ax=countries.sort_values('pop_est').plot('pop_est', cmap='plasma', figsize=(30, 20), legend=True)
ax.set_title("worldmap", fontsize=40)

# Visualizing GDP

ax=countries.sort_values('gdp_md_est').plot('gdp_md_est', cmap='viridis', figsize=(30, 20),legend=True)
ax.set_title("worldmap", fontsize = 40)

asia=countries[countries['continent']=='Asia']
asia['name']

# Sorting countries by overall GDP

import warnings
warnings.filterwarnings("ignore")

asia=countries[countries['continent']=='Asia']
asia.sort_values('gdp_md_est').plot('gdp_md_est', cmap='viridis',scheme='equal_interval', k=10, figsize=(15, 10),legend=True)

print('highest overall gdp in aisa:')
asia.loc[asia['gdp_md_est'].idxmax()]

# Add column gdp/capita

asia['gdp/pop'] = (asia.gdp_md_est / asia.pop_est)

asia

asia.sort_values('gdp/pop').plot('gdp/pop', cmap='viridis',scheme='equal_interval', k=10, figsize=(15, 10),legend=True)

print('highest gdp per capita:')
asia.loc[asia['gdp/pop'].idxmax()]

# Agglomorative clustering

import sklearn.cluster as skc
from sklearn.cluster import AgglomerativeClustering
asia_gdpclusters = AgglomerativeClustering(n_clusters=4).fit(asia[['gdp/pop']])
asia.assign(labels=asia_gdpclusters.labels_).plot('labels', cmap='plasma', figsize=(15, 10), legend=True)

# Kmeans clustering

from sklearn.cluster import KMeans 
asia_gdpclusters2 = KMeans(n_clusters=4).fit(asia[['gdp/pop']])
asia.assign(labels=asia_gdpclusters2.labels_).plot('labels', cmap='viridis', figsize=(15, 10), legend=True)

asia_gdpclusters2.score

asia_gdpclusters2.cluster_centers_

asia_gdpclusters2.labels_

cluster = asia_gdpclusters2.labels_

cluster

asia['cluster'] = cluster

asia

clasia = pd.DataFrame(asia)

clus = clasia.loc[asia['cluster']== 1]

clus.name

clasia.name
