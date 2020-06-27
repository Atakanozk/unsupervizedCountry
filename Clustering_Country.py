# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:52:11 2020

@author: ataka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#importing dataset 
dataset = pd.read_csv("Country-data.csv")
dataset_dict = pd.read_csv("data-dictionary.csv")
col_meanings = np.array(dataset_dict["Column Name"])
col_meanings.T
dataset.info()
"""
RangeIndex: 167 entries, 0 to 166
Data columns (total 10 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   country     167 non-null    object 
 1   child_mort  167 non-null    float64
 2   exports     167 non-null    float64
 3   health      167 non-null    float64
 4   imports     167 non-null    float64
 5   income      167 non-null    int64  
 6   inflation   167 non-null    float64
 7   life_expec  167 non-null    float64
 8   total_fer   167 non-null    float64
 9   gdpp        167 non-null    int64  

"""
dataset.head()
"""
             country  child_mort  exports  health  imports  income  inflation  life_expec  total_fer   gdpp
0          Afghanistan        90.2     10.0    7.58     44.9    1610       9.44        56.2       5.82    553
1              Albania        16.6     28.0    6.55     48.6    9930       4.49        76.3       1.65   4090
2              Algeria        27.3     38.4    4.17     31.4   12900      16.10        76.5       2.89   4460
3               Angola       119.0     62.3    2.85     42.9    5900      22.40        60.1       6.16   3530
4  Antigua and Barbuda        10.3     45.5    6.03     58.9   19100       1.44        76.8       2.13  12200
"""
dataset.isnull().values.any()#No NaN values
for i, row in dataset_dict.iterrows():
    print(row['Column Name'], '--->', row['Description'])
"""
country ---> Name of the country
child_mort ---> Death of children under 5 years of age per 1000 live births
exports ---> Exports of goods and services per capita. Given as %age of the GDP per capita
health ---> Total health spending per capita. Given as %age of GDP per capita
imports ---> Imports of goods and services per capita. Given as %age of the GDP per capita
Income ---> Net income per person
Inflation ---> The measurement of the annual growth rate of the Total GDP
life_expec ---> The average number of years a new born child would live if the current mortality patterns are to remain the same
total_fer ---> The number of children that would be born to each woman if the current age-fertility rates remain the same.
gdpp ---> The GDP per capita. Calculated as the Total GDP divided by the total population.
"""

#Replacing country column with Country_ID columns for feature scaling
dataset.insert(0, 'Country_ID', range(0, 166 + 1))
country_data = dataset.iloc[:,1].values
dataset = dataset.drop("country", axis=1)


#feature scealing
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler(feature_range = (0, 1))
dataset = ms.fit_transform(dataset)

#training the set 
from minisom import MiniSom
sm = MiniSom(x = 10, y = 10, input_len = 10, sigma = 1.0, learning_rate = 0.5)
sm.random_weights_init(dataset)
sm.train_random(data = dataset, num_iteration = 100)
#Self orginizng map for clustering
from pylab import bone,pcolor,colorbar,plot,show
plt.title("Self Orginizing Map")
bone()
pcolor(sm.distance_map().T)
colorbar()
markers = ["o", "s", "+", "x"]
colors = ["r", "g", "c", "m"]
for i, x in enumerate(dataset):
    w = sm.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y_pred_pca[i]],
         markeredgecolor = colors[y_pred_pca[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
    
#K-MEANS clustering part
kdata = pd.read_csv("Country-data.csv")
X = kdata.iloc[:,1:10].values

#finding optimal number of clusters 

from sklearn.cluster import KMeans
listt = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
    kmeans.fit(X)
    listt.append(kmeans.inertia_)
plt.plot(range(1,11), listt)
plt.title("Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("listt")
plt.show()    
#optimal number of cluster is 4
#training 
kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 42)
y_pred = kmeans.fit_predict(X)
print(y_pred)
#appliying Priciple component analysis
#StandardScaler 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_pca = sc.fit_transform(X)
#Applying principle component analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_pca)  

#training PCA
kmeans = KMeans(n_clusters = 4, init = "k-means++", random_state = 42)
y_pred_pca = kmeans.fit_predict(X_pca)

  
#visualising the clusters 
plt.scatter(X_pca[y_pred == 0, 0], X_pca[y_pred == 0, 1], s = 50, c = "red", label = "cluster 1" )
plt.scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], s = 50, c = "green", label = "cluster 2" )
plt.scatter(X_pca[y_pred == 2, 0], X_pca[y_pred == 2, 1], s = 50, c = "cyan", label = "cluster 3" )
plt.scatter(X_pca[y_pred == 3, 0], X_pca[y_pred == 3, 1], s = 50, c = "magenta", label = "cluster 4" )
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = "yellow", label = "Centroids" )
plt.title("Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

#Hierarchical Clutering
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X_pca, method  = "ward"))
plt.title("Dendrogram of PCA")
plt.xlabel("Country")
plt.ylabel("Distances")
plt.show()
#I chosed 2 clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = "euclidean", linkage = "ward")
y_pred_hc = hc.fit_predict(X_pca)

#visualising the clusters 
plt.scatter(X_pca[y_pred_hc == 0, 0], X_pca[y_pred_hc == 0, 1], s = 50, c = "red", label = "cluster 1" )
plt.scatter(X_pca[y_pred_hc == 1, 0], X_pca[y_pred_hc == 1, 1], s = 50, c = "green", label = "cluster 2" )
plt.title("Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()

#I chosed 4 clusters
hc = AgglomerativeClustering(n_clusters = 4, affinity = "euclidean", linkage = "ward")
y_pred_hc = hc.fit_predict(X_pca)

#visualising the clusters 
plt.scatter(X_pca[y_pred_hc == 0, 0], X_pca[y_pred_hc == 0, 1], s = 50, c = "red", label = "cluster 1" )
plt.scatter(X_pca[y_pred_hc == 1, 0], X_pca[y_pred_hc == 1, 1], s = 50, c = "green", label = "cluster 2" )
plt.scatter(X_pca[y_pred_hc == 2, 0], X_pca[y_pred_hc == 2, 1], s = 50, c = "cyan", label = "cluster 3" )
plt.scatter(X_pca[y_pred_hc == 3, 0], X_pca[y_pred_hc == 3, 1], s = 50, c = "magenta", label = "cluster 4" )
plt.title("Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()


